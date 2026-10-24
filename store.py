"""
store.py - everything SpyMotion keeps on disk, in one small, inspectable place.

Stored (all inside DATA_DIR, owner-only permissions):
  history.json           movement numbers only (never video), only if the user switches it on
  wellbeing_entries.json the daily wellbeing diary, written only when the user presses Save
Transient (TMP):
  the ORIGINAL upload exists only while it is analysed; the result video is read into memory
  and its file is removed straight away.

Reliability / privacy guarantees (each is covered by test_store.py):
  - Nothing here raises into the UI: a full disk or a permissions error cannot crash the app.
  - A corrupt JSON file is moved aside, never silently overwritten.
  - NaN / numpy values are sanitised so files are always strict, valid JSON.
  - Writes are atomic (temp file + rename): a crash cannot leave half a file.
  - Folders are owner-only (0700), files 0600, and the temp folder name is per-user.
  - Leftover original footage from a crashed run is swept on start-up.
  - DATA_DIR is the per-user app-data folder (outside the project), so health data can never be
    committed to git; data an older build left inside the project folder is migrated there.
  - delete_everything() covers history, diary, temp files and legacy files, and VERIFIES it.
"""
from __future__ import annotations

import atexit
import json
import logging
import math
import numbers
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("spymotion")


def _user_tag() -> str:
    try:
        return str(os.getuid())
    except AttributeError:  # Windows
        return os.environ.get("USERNAME", "user")


_HERE = Path(__file__).resolve().parent


def _default_data_dir() -> Path:
    """Per-user app-data folder OUTSIDE the project, so health data can never be committed to git."""
    try:
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
        return base / "SpyMotion"
    except RuntimeError:  # no home directory (some containers)
        return Path(tempfile.gettempdir()) / f"spymotion_data_{_user_tag()}"


# Override with SPYMOTION_DATA=/some/folder if you want it elsewhere (keep it outside the repository).
DATA_DIR = Path(os.environ.get("SPYMOTION_DATA") or os.environ.get("SPINEMOTION_DATA") or _default_data_dir()).resolve()
HISTORY = DATA_DIR / "history.json"
PROM_NAME = "wellbeing_entries.json"
TMP = Path(tempfile.gettempdir()) / f"spymotion_{_user_tag()}"

# Earlier builds kept the diary next to the source code (easy to commit to git by accident).
LEGACY_PROM_FILES = [_HERE / "spinemotion_prom_entries.json", _HERE / "spymotion_prom_entries.json"]
# Earlier builds kept the whole data folder inside the project folder.
LEGACY_DATA_DIRS = [_HERE / "spymotion_data", _HERE / "spinemotion_data"]

MAX_ROWS = 5000
STALE_PATTERNS = ("in_*", "*.raw.mp4")  # original footage / half-written videos


# ---------------------------------------------------------------- generic helpers
def _private_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def tmp_dir() -> Path:
    return _private_dir(TMP)


def _clean(v):
    """Make a value strictly JSON-safe: numpy -> python, NaN/inf -> None."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_clean(x) for x in v]
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return str(v)


def _read_json(path: Path, want: type) -> tuple[object, bool]:
    """Return (data, ok). ok=False means the file exists but is unreadable / the wrong shape."""
    empty = want()
    if not path.exists():
        return empty, True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, want):
            raise ValueError("unexpected JSON shape")
        return data, True
    except Exception:
        return empty, False


def _quarantine(path: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path.replace(path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}"))
    log.warning("%s was unreadable; moved aside", path.name)


def _write_json_atomic(path: Path, obj) -> bool:
    tmp = None
    try:
        _private_dir(path.parent)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return True
    except Exception:
        log.exception("could not write %s", path.name)
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return False


# ---------------------------------------------------------------- movement history
def load_history() -> list[dict]:
    _migrate_legacy_dirs()
    rows, _ = _read_json(HISTORY, list)
    return [r for r in rows if isinstance(r, dict)]  # type: ignore[union-attr]


def append_entry(entry: dict) -> bool:
    """Append one numeric entry. Returns True on success; never raises."""
    try:
        _migrate_legacy_dirs()
        rows, ok = _read_json(HISTORY, list)
        if not ok:
            _quarantine(HISTORY)
        rows = [r for r in rows if isinstance(r, dict)]  # type: ignore[union-attr]
        rows.append({"saved_at": datetime.now().isoformat(timespec="seconds"), **_clean(entry)})
        return _write_json_atomic(HISTORY, rows[-MAX_ROWS:])
    except Exception:
        log.exception("could not save history entry")
        return False


# ---------------------------------------------------------------- wellbeing diary
def prom_path() -> Path:
    return DATA_DIR / PROM_NAME


def _migrate_legacy_dirs() -> None:
    """Move data an older build kept inside the project folder into the per-user DATA_DIR."""
    try:
        for old in LEGACY_DATA_DIRS:
            if not old.is_dir() or old.resolve() == DATA_DIR:
                continue
            for f in [p for p in old.iterdir() if p.is_file()]:
                dest = DATA_DIR / f.name
                if not dest.exists():
                    _private_dir(DATA_DIR)
                    shutil.move(str(f), str(dest))
            try:
                old.rmdir()  # only succeeds if empty
            except OSError:
                pass
    except Exception:
        log.exception("data migration failed")


def _migrate_legacy_prom() -> None:
    """Move a diary that an older build left next to the source code into DATA_DIR."""
    try:
        for legacy in LEGACY_PROM_FILES:
            if legacy.exists() and not prom_path().exists():
                _private_dir(DATA_DIR)
                shutil.move(str(legacy), str(prom_path()))
                try:
                    os.chmod(prom_path(), 0o600)
                except OSError:
                    pass
                log.info("migrated diary from %s", legacy)
    except Exception:
        log.exception("diary migration failed")


def load_prom() -> tuple[dict, bool]:
    """Return (entries keyed by ISO date, ok). ok=False: the file existed but was unreadable."""
    _migrate_legacy_dirs()
    _migrate_legacy_prom()
    data, ok = _read_json(prom_path(), dict)
    return data, ok  # type: ignore[return-value]


def save_prom(entries: dict) -> bool:
    """Atomically save the whole diary. Returns False (and changes nothing) on failure."""
    try:
        _migrate_legacy_dirs()
        _, ok = _read_json(prom_path(), dict)
        if not ok:
            _quarantine(prom_path())
        return _write_json_atomic(prom_path(), _clean(entries))
    except Exception:
        log.exception("could not save diary")
        return False


def delete_prom() -> bool:
    ok = True
    for p in [prom_path(), *LEGACY_PROM_FILES]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            ok = False
    return ok


# ---------------------------------------------------------------- audit / delete / sweep
def _all_roots() -> list[Path]:
    return [TMP, DATA_DIR, *LEGACY_DATA_DIRS]


def audit() -> list[tuple[str, int]]:
    """Every file the app currently holds, with size in bytes (includes any legacy diary)."""
    files: list[tuple[str, int]] = []
    for base in _all_roots():
        if not base.exists():
            continue
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    files.append((str(p), p.stat().st_size))
            except OSError:  # vanished between listing and stat
                continue
    for p in LEGACY_PROM_FILES:
        try:
            if p.is_file():
                files.append((str(p), p.stat().st_size))
        except OSError:
            continue
    return files


def delete_everything_report() -> dict:
    """Delete all stored data and VERIFY it. Returns {'deleted': int, 'failed': [paths]}."""
    files = [p for p, _ in audit()]
    for base in _all_roots():
        shutil.rmtree(base, ignore_errors=True)
    for p in LEGACY_PROM_FILES:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    failed = []
    for f in files:
        fp = Path(f)
        if fp.exists():
            try:
                fp.unlink()
            except OSError:
                failed.append(f)
    return {"deleted": len(files) - len(failed), "failed": failed}


def delete_everything() -> int:
    """Back-compat wrapper: returns the number of files deleted."""
    rep = delete_everything_report()
    if rep["failed"]:
        log.error("could not delete: %s", rep["failed"])
    return rep["deleted"]


def sweep_stale(max_age_s: float = 120.0, now: float | None = None) -> int:
    """Remove original-footage / half-written files left behind by a crashed run."""
    now = time.time() if now is None else now
    removed = 0
    if not TMP.exists():
        return 0
    for pattern in STALE_PATTERNS:
        for p in TMP.glob(pattern):
            try:
                if p.is_file() and now - p.stat().st_mtime > max_age_s:
                    p.unlink()
                    removed += 1
            except OSError:
                continue
    return removed


try:
    sweep_stale()
except Exception:  # never block start-up
    pass

atexit.register(lambda: shutil.rmtree(TMP, ignore_errors=True))
