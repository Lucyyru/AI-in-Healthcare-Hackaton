"""AccessScan's persistent connection to the independent model environment."""
from __future__ import annotations

import atexit
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock
import time


APP_DIR = Path(__file__).resolve().parent
_RUN_LOCK = Lock()
_WORKER = None
log = logging.getLogger("accessscan")


def backend_directory() -> Path:
    for parent in (APP_DIR, *APP_DIR.parents):
        candidate = parent / "SidewalkDetection"
        if (candidate / "pipeline.py").is_file():
            return candidate
    raise RuntimeError("Keep this app and SidewalkDetection inside the same project folder.")


def backend_python() -> str:
    configured = os.environ.get("SIDEWALK_PYTHON")
    local_config = APP_DIR / "accessscan.local.json"
    if not configured and local_config.is_file():
        configured = json.loads(local_config.read_text(encoding="utf-8"))["backend_python"]
    if configured:
        path = Path(configured).expanduser()
        return str(path if path.is_absolute() else (APP_DIR / path).resolve())
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    candidate = backend_directory() / ".venv" / executable
    return str(candidate) if candidate.is_file() else sys.executable


def stop_worker():
    """Stop only the model process owned by this AccessScan connection."""
    global _WORKER
    worker, _WORKER = _WORKER, None
    if worker is None:
        return
    if worker.stdin:
        try:
            worker.stdin.close()
        except OSError:
            pass
    try:
        worker.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            # A Windows venv launcher may have a Python child process.
            subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            worker.kill()
        worker.wait(timeout=5)


atexit.register(stop_worker)


def _model_worker(backend):
    global _WORKER
    if _WORKER is None or _WORKER.poll() is not None:
        stop_worker()
        output = backend / "output"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "accessscan-worker.log").open("w", encoding="utf-8") as stream:
            _WORKER = subprocess.Popen(
                [backend_python(), "-u", str(APP_DIR / "accessscan_worker.py"), str(backend)],
                cwd=backend, stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
    return _WORKER


def analyze_access_image(image_bytes: bytes, suffix: str, height_cm: float,
                         wheelchair_width_cm: float) -> dict:
    """Reuse loaded models and the last image; inputs are in centimetres."""
    if any(not math.isfinite(value) or value <= 0 for value in (height_cm, wheelchair_width_cm)):
        raise ValueError("Camera height and wheelchair width must be positive finite numbers.")

    with _RUN_LOCK, tempfile.TemporaryDirectory(prefix="accessscan_") as temporary:
        image = Path(temporary) / ("upload" + suffix.lower())
        response = Path(temporary) / "response.json"
        image.write_bytes(image_bytes)
        request = {"image": str(image), "height_m": height_cm / 100,
                   "wheelchair_width_m": wheelchair_width_cm / 100, "response": str(response)}
        worker = _model_worker(backend_directory())
        try:
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            deadline = time.monotonic() + 300
            while not response.exists():
                if worker.poll() is not None:
                    raise RuntimeError("The model process stopped. Please analyse again.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Analysis timed out. Please try again.")
                time.sleep(0.05)
            result = json.loads(response.read_text(encoding="utf-8"))
            if not result["ok"]:
                raise RuntimeError(result["error"])
            return result["report"]
        except (OSError, RuntimeError, ValueError) as error:
            log.exception("AccessScan analysis failed")
            if isinstance(error, (OSError, ValueError)):
                stop_worker()
            raise


def result_html(report):
    """Use explicit colours so this result stays readable in either app theme."""
    if report.get("width_m") is None:
        text, color, background = "Unable to determine", "#475569", "#F1F5F9"
    elif report.get("passability") == "Can pass!":
        text, color, background = "Can pass!", "#0B6E3B", "#EAF7EE"
    else:
        text, color, background = "Cannot pass", "#B91C1C", "#FEF2F2"
    return (
        '<div class="accessscan-result" role="status" aria-live="polite" '
        f'style="color:{color};background:{background};font-size:clamp(40px,6vw,72px);'
        'font-weight:800;line-height:1.2;text-align:center;padding:36px 16px;'
        f'border-radius:20px;margin:20px 0;">{text}</div>'
    )
