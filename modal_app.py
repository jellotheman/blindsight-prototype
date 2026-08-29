"""Modal deployment: the BlindSight `/v1` API and the reference web client, served from one
Modal application so the client has no privileged in-process path -- it calls the same HTTP
interface any other caller would.

Deploy with:

    modal deploy modal_app.py

Requires a Modal secret named `blindsight-api-key` holding `BLINDSIGHT_API_KEY`, created before
the function is declared:

    modal secret create blindsight-api-key BLINDSIGHT_API_KEY=<shared key>

The secret is attached at function declaration below. There is no in-container probe: a missing
secret fails the deployment loudly instead of silently serving without a key.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent

api_key_secret = modal.Secret.from_name("blindsight-api-key")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("fastapi>=0.115", "pydantic>=2.7")
    .add_local_python_source("blindsight")
    .add_local_dir(str(ROOT / "data"), remote_path="/root/data")
    .add_local_dir(str(ROOT / "static"), remote_path="/root/static")
)

app = modal.App("blindsight-api")
capture_state = modal.Dict.from_name("blindsight-capture-state", create_if_missing=True)


@app.function(image=image, secrets=[api_key_secret])
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from blindsight.app import create_app, mount_reference_client
    from blindsight.storage import ModalCaptureStore

    fastapi_app = create_app(
        api_key=os.environ["BLINDSIGHT_API_KEY"],
        manifest_path=Path("/root/data/excerpts/manifest.json"),
        store=ModalCaptureStore(capture_state),
    )
    mount_reference_client(fastapi_app, static_dir=Path("/root/static"))
    return fastapi_app
