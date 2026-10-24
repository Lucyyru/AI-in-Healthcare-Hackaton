"""
store.py - everything SpineMotion keeps on disk, in one small, inspectable place.

* history.json   - numbers only (never video), and only if the user switches it on.
* temp dir       - the face-blurred, audio-free result video; wiped on exit or on Delete.
The original upload is written to temp only while it is analysed, then removed.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("SPINEMOTION_DATA", "spinemotion_data")).resolve()
HISTORY = DATA_DIR / "history.json"
TMP = Path(tempfile.gettempdir()) / "spinemotion_session"


def tmp_dir() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY.read_text())
    except Exception:
        return []


def append_entry(entry: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_history()
    rows.append({"saved_at": datetime.now().isoformat(timespec="seconds"), **entry})
    tmp = HISTORY.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    tmp.replace(HISTORY)


def audit() -> list[tuple[str, int]]:
    """Every file the app currently holds, with size in bytes."""
    files = []
    for base in (TMP, DATA_DIR):
        if base.exists():
            files += [(str(p), p.stat().st_size) for p in base.rglob("*") if p.is_file()]
    return files


def delete_everything() -> int:
    n = len(audit())
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    return n


atexit.register(lambda: shutil.rmtree(TMP, ignore_errors=True))
