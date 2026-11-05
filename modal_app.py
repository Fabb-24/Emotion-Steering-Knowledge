"""
Run the project on Modal GPUs. Data and results live on the "emotion-steering" volume.

modal run modal_app.py --args "layer dataset"   # any command of app/cli.py
"""

import shlex
import modal

APP_DIR = "/root/app"   # project code inside the container
VOLUME_DIR = "/vol"     # the volume: data/, results/ and the Hugging Face cache
GPU = "A100-80GB"
TIMEOUT = 6 * 60 * 60   # seconds

volume = modal.Volume.from_name("emotion-steering")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers==5.20.2",
        "accelerate",
        "bitsandbytes",
        "scikit-learn",
        "pandas",
        "huggingface_hub",
        "matplotlib",
        "tqdm",
        "typer",
        "flash-linear-attention",  # fast kernels of the linear attention layers of Qwen3.8
    )
    .env({
        "PROJECT_ROOT": VOLUME_DIR,           # config.py reads data/ and results/ from here
        "HF_HOME": f"{VOLUME_DIR}/hf_cache",  # model weights are downloaded only once
    })
    .add_local_dir("app", remote_path=APP_DIR)  # the local code, sent at every run
)

app = modal.App("emotion-steering", image=image)


@app.function(gpu=GPU, volumes={VOLUME_DIR: volume}, timeout=TIMEOUT, secrets=[modal.Secret.from_name("huggingface")])  # HF_TOKEN for gated models (Llama)
def cli(args: str):
    """
    Run one command of app/cli.py (e.g. "halluc eval --layer 20") on the GPU.
    """

    import sys
    sys.path.insert(0, APP_DIR)
    from cli import app as cli_app

    try:
        cli_app(shlex.split(args), standalone_mode=False)
    finally:
        volume.commit()  # save data and results on the volume even if the command fails


@app.local_entrypoint()
def main(args: str):
    """
    `modal run modal_app.py --args "<command>"`: run a command of app/cli.py on Modal.
    """
    cli.remote(args)
