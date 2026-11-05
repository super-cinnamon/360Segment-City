import threading
import time

import uvicorn

from src.tasks.api.api import app
from src.tasks.api.llama_cpp_backend import start_llama_cpp_server
from src.tasks.config.utils import apply_vlm_profile, resolve_vlm_profile
from src.tasks.environment import load_world_model


def start_api_in_background(host="127.0.0.1", port=5015, model_name=None):
    """Start the local FastAPI app in a daemon thread."""
    if model_name is None:
        profile = apply_vlm_profile()
    else:
        profile = resolve_vlm_profile(model_name)

    if profile["backend"] == "llama.cpp":
        start_llama_cpp_server(profile)
    else:
        load_world_model(profile["repo_id"])

    def _serve():
        uvicorn.run(app, host=host, port=port)

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()
    print(f"Starting API server on http://{host}:{port}...")
    time.sleep(3)
    return server_thread
