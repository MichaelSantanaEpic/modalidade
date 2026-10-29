"""Browser-only ComfyUI deployment on Modal with MOSS-Image-SFT BF16 + Anima Base.

Model downloads and model inference occur in Modal, never on the deployment client.
Source assumptions: Modal web_server/Volume docs and ComfyUI v0.42.1 CLI.
"""

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("moss-image-comfyui")
models = modal.Volume.from_name("moss-image-bf16-models", create_if_missing=True)
userdata = modal.Volume.from_name("moss-image-comfy-data", create_if_missing=True)

# Files to keep on the persistent Modal volume in folders that ComfyUI scans.
DOWNLOAD_SPECS = (
    # MOSS-Image-SFT BF16
    {
        "repo_id": "Comfy-Org/MOSS-Image",
        "filename": "diffusion_models/moss_image_sft_bf16.safetensors",
        "dest": "diffusion_models/moss_image_sft_bf16.safetensors",
    },
    {
        "repo_id": "Comfy-Org/MOSS-Image",
        "filename": "text_encoders/qwen3.5_9b_bf16.safetensors",
        "dest": "text_encoders/qwen3.5_9b_bf16.safetensors",
    },
    {
        "repo_id": "Comfy-Org/MOSS-Image",
        "filename": "vae/flux2-vae.safetensors",
        "dest": "vae/flux2-vae.safetensors",
    },
    # Anima Base v1.0 + requirements
    {
        "repo_id": "circlestone-labs/Anima",
        "filename": "split_files/diffusion_models/anima-base-v1.0.safetensors",
        "dest": "diffusion_models/anima-base-v1.0.safetensors",
    },
    {
        "repo_id": "circlestone-labs/Anima",
        "filename": "split_files/text_encoders/qwen_3_06b_base.safetensors",
        "dest": "text_encoders/qwen_3_06b_base.safetensors",
    },
    {
        "repo_id": "circlestone-labs/Anima",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": "vae/qwen_image_vae.safetensors",
    },
)

REQUIRED_SUBDIRS = ("diffusion_models", "text_encoders", "vae")

# This light CPU image only downloads the weights on Modal; no GPU charges.
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]>=0.35,<2")
    .env({"HF_XET_NUM_CONCURRENT_RANGE_GETS": "4"})
)


@app.function(
    image=download_image,
    cpu=1,
    memory=128,
    timeout=7200,
    volumes={"/mnt/models": models},
)
def download_models():
    """One-time cloud download of MOSS + Anima model components."""
    from huggingface_hub import hf_hub_download

    base = Path("/mnt/models")
    for sub in REQUIRED_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)

    for spec in DOWNLOAD_SPECS:
        dest = base / spec["dest"]
        print(f"Downloading/checking: {spec['repo_id']} :: {spec['filename']}", flush=True)
        cached = Path(
            hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
            )
        )
        if not cached.is_file() or cached.stat().st_size < 1_000_000:
            raise RuntimeError(f"Missing or incomplete cached file: {cached}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != cached.stat().st_size:
            shutil.copy2(cached, dest)
        if not dest.is_file() or dest.stat().st_size < 1_000_000:
            raise RuntimeError(f"Missing/incomplete copied component: {dest}")
        print(f"Verified file present: {dest} ({dest.stat().st_size:,} bytes)", flush=True)
        models.commit()
    print("All requested files are on the persistent Modal Volume.", flush=True)


# Image and dependencies are built on Modal, not your PC.
# v0.42.1 specifically includes the MOSS-Image workflow crash fixes.
comfy_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "nginx", "apache2-utils", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone --depth 1 --branch v0.42.1 "
        "https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI"
    )
    .run_commands(
        "python -m pip install --no-cache-dir "
        "torch==2.10.0 torchvision==0.25.0 "
        "--index-url https://download.pytorch.org/whl/cu130",
        "python -m pip install --no-cache-dir "
        "-r /opt/ComfyUI/requirements.txt",
        "python -c 'import torch; "
        "assert torch.backends.cuda.is_built(), \"CUDA-enabled PyTorch required\"'",
    )
)


def _patch_modal_userdata_routes():
    file = Path("/opt/ComfyUI/app/user_manager.py")
    text = file.read_text(encoding="utf8")
    substitutions = {
        '"/userdata/{file}"': '"/userdata/{file:.+}"',
        '"/userdata/{file}/move/{dest}"': '"/userdata/{file:.+}/move/{dest:.+}"',
    }
    for before, after in substitutions.items():
        if before not in text and after not in text:
            raise RuntimeError(f"ComfyUI changed its route; review the saving patch: {before}")
        text = text.replace(before, after)
    file.write_text(text, encoding="utf8")


def _wait_for_server(port: int, proc: subprocess.Popen, seconds: int = 450):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI terminated at startup, status {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("ComfyUI did not open its port before the startup deadline")


@app.function(
    image=comfy_image,
    gpu="L40S",
    cpu=1,
    memory=128,
    max_containers=1,
    min_containers=0,
    scaledown_window=30,
    timeout=3600,
    volumes={"/mnt/models": models, "/mnt/data": userdata},
    secrets=[modal.Secret.from_name("moss-comfy-password")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=600)
def comfyui():
    if not os.environ.get("COMFY_PASSWORD"):
        raise RuntimeError("Set the COMFY_PASSWORD value in Modal secret moss-comfy-password")

    for sub in ("user", "input", "output"):
        (Path("/mnt/data") / sub).mkdir(parents=True, exist_ok=True)
    for sub in REQUIRED_SUBDIRS:
        (Path("/mnt/models") / sub).mkdir(parents=True, exist_ok=True)
    for spec in DOWNLOAD_SPECS:
        path = Path("/mnt/models") / spec["dest"]
        if not path.is_file():
            raise RuntimeError(f"Missing {path}; run download_models before deploying")

    _patch_modal_userdata_routes()

    Path("/tmp/multi_model_paths.yaml").write_text(
        "persistent_models:\n"
        "  base_path: /mnt/models\n"
        "  diffusion_models: diffusion_models\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n",
        encoding="utf8",
    )

    proc = subprocess.Popen(
        [
            "python", "main.py",
            "--listen", "127.0.0.1", "--port", "8188",
            "--disable-auto-launch", "--fp32-vae",
            "--extra-model-paths-config", "/tmp/multi_model_paths.yaml",
            "--user-directory", "/mnt/data/user",
            "--input-directory", "/mnt/data/input",
            "--output-directory", "/mnt/data/output",
        ],
        cwd="/opt/ComfyUI",
    )
    _wait_for_server(8188, proc)

    htpasswd = subprocess.check_output(
        ["htpasswd", "-nbB", "comfy", os.environ["COMFY_PASSWORD"]],
        text=True,
    )
    Path("/tmp/comfy.htpasswd").write_text(htpasswd, encoding="utf8")
    Path("/tmp/comfy.htpasswd").chmod(0o644)
    Path("/tmp/comfy.nginx.conf").write_text(
        """worker_processes 1;
        events { worker_connections 1024; }
        http {
          map $http_upgrade $connection_upgrade { default upgrade; '' close; }
          server {
            listen 8000;
            server_name _;
            auth_basic \"ComfyUI - Modal\";
            auth_basic_user_file /tmp/comfy.htpasswd;
            client_max_body_size 512m;
            location / {
              proxy_pass http://127.0.0.1:8188;
              proxy_http_version 1.1;
              proxy_set_header Host $host;
              proxy_set_header Upgrade $http_upgrade;
              proxy_set_header Connection $connection_upgrade;
              proxy_set_header Sec-WebSocket-Extensions \"\";
              proxy_read_timeout 3600s;
              proxy_send_timeout 3600s;
              proxy_buffering off;
              proxy_request_buffering off;
            }
          }
        }
        """,
        encoding="utf8",
    )
    subprocess.Popen(["nginx", "-c", "/tmp/comfy.nginx.conf", "-g", "daemon off;"])
    print("Authenticated ComfyUI server launched with MOSS + Anima models.", flush=True)
