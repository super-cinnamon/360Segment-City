import threading
import time

import uvicorn

from src.tasks.api.api import app
from src.tasks.config.utils import CONFIG

MODEL_NAME = CONFIG["vlm"]["world_model"]["model_name"]


def start_api_in_background(host="127.0.0.1", port=(5015), model_name=MODEL_NAME):
    """Start the local FastAPI app in a daemon thread."""

    def _serve():
        uvicorn.run(app, host=host, port=port)

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()
    print(f"Starting API server on http://{host}:{port}...")
    time.sleep(3)
    return server_thread
