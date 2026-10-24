"""Shared pytest set-up: make the project importable and make it impossible for a test to touch real user data."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Every test gets its own empty data + temp folders; the real ones are never read or deleted."""
    import store

    data = tmp_path / "data"
    monkeypatch.setattr(store, "DATA_DIR", data)
    monkeypatch.setattr(store, "HISTORY", data / "history.json")
    monkeypatch.setattr(store, "TMP", tmp_path / "tmp")
    monkeypatch.setattr(store, "LEGACY_PROM_FILES", [tmp_path / "legacy_a.json", tmp_path / "legacy_b.json"])
    monkeypatch.setattr(store, "LEGACY_DATA_DIRS", [tmp_path / "old_project_data", tmp_path / "old_project_data2"])
    yield
