import sys
import time
import threading
from openai import OpenAI

# Import the server launcher created above
from api import run_server

from src.tasks.config.utils import CONFIG

IS_WINDOWS = sys.platform.startswith('win')
MODEL_NAME = CONFIG["vlm"]["world_model"]["model_name"]
API_URL = "http://localhost:8000/v1"

def start_api_in_background():
    """Starts the Windows HF server in a daemon thread."""
    server_thread = threading.Thread(
        target=run_server, 
        kwargs={"host": "127.0.0.1", "port": 8000, "model_name": MODEL_NAME},
        daemon=True
    )
    server_thread.start()
    print("Waiting 10s for API Server initialization...")
    time.sleep(10)
