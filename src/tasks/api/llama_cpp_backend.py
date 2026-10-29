"""Lifecycle helpers for profile-specific llama.cpp GGUF servers."""

import atexit
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict
from urllib.error import URLError
from urllib.request import urlopen

from huggingface_hub import snapshot_download


_SERVERS: Dict[str, tuple[subprocess.Popen, str]] = {}


def _cached_snapshot(profile: dict) -> Path:
    try:
        snapshot_dir = snapshot_download(
            repo_id=profile["repo_id"],
            allow_patterns=profile.get("download_patterns"),
            local_files_only=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"GGUF files for profile '{profile['id']}' are not cached. Download them "
            f"with `python scripts/download_vlm_models.py --model {profile['id']}`."
        ) from exc
    return Path(snapshot_dir)


def _gguf_path(profile: dict, snapshot_dir: Path, filename_key: str) -> Path:
    filename = profile.get(filename_key)
    if not filename:
        raise ValueError(
            f"GGUF profile '{profile['id']}' must configure '{filename_key}'."
        )

    path = snapshot_dir / filename
    if not path.is_file():
        raise RuntimeError(
            f"Required GGUF file '{filename}' is missing from profile "
            f"'{profile['id']}' cache. Download it with "
            f"`python scripts/download_vlm_models.py --model {profile['id']}`."
        )
    return path


def start_llama_cpp_server(profile: dict) -> str:
    """Start and wait for the local llama.cpp server for one GGUF profile."""
    profile_id = profile["id"]
    existing = _SERVERS.get(profile_id)
    if existing is not None:
        process, base_url = existing
        if process.poll() is None:
            return base_url
        _SERVERS.pop(profile_id)

    configured_executable = os.environ.get("LLAMA_SERVER_EXECUTABLE", "llama-server")
    executable = shutil.which(configured_executable)
    if executable is None:
        raise RuntimeError(
            "The GGUF profile requires the llama.cpp `llama-server` executable. "
            "Install a llama.cpp build with multimodal support and ensure "
            "`llama-server` is on PATH, or set LLAMA_SERVER_EXECUTABLE."
        )

    snapshot_dir = _cached_snapshot(profile)
    model_path = _gguf_path(profile, snapshot_dir, "model_file")
    mmproj_path = _gguf_path(profile, snapshot_dir, "mmproj_file")
    port = int(profile.get("server_port", 5016))
    base_url = f"http://127.0.0.1:{port}"
    command = [
        executable,
        "--model", str(model_path),
        "--mmproj", str(mmproj_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(profile.get("context_size", 8192)),
    ]
    process = subprocess.Popen(command)
    _SERVERS[profile_id] = (process, base_url)

    timeout = float(profile.get("startup_timeout_seconds", 180))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _SERVERS.pop(profile_id, None)
            raise RuntimeError(
                f"llama-server exited with status {process.returncode} while starting "
                f"profile '{profile_id}'. Check the server output for details."
            )
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    atexit.register(stop_llama_cpp_servers)
                    print(f"llama.cpp API ready at {base_url}")
                    return base_url
        except (URLError, TimeoutError):
            time.sleep(1)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    _SERVERS.pop(profile_id, None)
    raise RuntimeError(
        f"llama-server did not become ready within {timeout:g} seconds for "
        f"profile '{profile_id}'."
    )


def get_llama_cpp_server_url(profile_id: str) -> str:
    server = _SERVERS.get(profile_id)
    if server is None:
        raise RuntimeError(
            f"llama.cpp server for profile '{profile_id}' was not started. "
            "Start the API with this profile selected."
        )
    process, base_url = server
    if process.poll() is not None:
        raise RuntimeError(
            f"llama.cpp server for profile '{profile_id}' exited with status "
            f"{process.returncode}."
        )
    return base_url


def stop_llama_cpp_servers() -> None:
    for profile_id, (process, _) in list(_SERVERS.items()):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        _SERVERS.pop(profile_id, None)
