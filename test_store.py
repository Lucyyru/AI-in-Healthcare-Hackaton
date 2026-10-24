"""Storage, history, diary, deletion and privacy tests (all run in an isolated temp folder)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import store


# ---------------------------------------------------------------- history
def test_history_roundtrip_and_order():
    assert store.load_history() == []
    assert store.append_entry({"exercise": "knee", "mean_peak": 80.5})
    assert store.append_entry({"exercise": "hip", "mean_peak": 60})
    rows = store.load_history()
    assert [r["exercise"] for r in rows] == ["knee", "hip"] and "saved_at" in rows[0]


def test_history_is_strict_json_even_with_nan_and_numpy():
    store.append_entry({"mean_peak": float("nan"), "x": np.float64(1.5), "n": np.int64(3), "inf": float("inf")})

    def boom(x):
        raise AssertionError(f"non-strict JSON constant {x}")

    data = json.loads(store.HISTORY.read_text(), parse_constant=boom)
    assert data[0]["mean_peak"] is None and data[0]["x"] == 1.5 and data[0]["n"] == 3 and data[0]["inf"] is None


def test_corrupt_history_is_kept_aside_not_overwritten():
    store.DATA_DIR.mkdir(parents=True)
    store.HISTORY.write_text("{this is not json")
    assert store.load_history() == []  # reading never crashes
    assert store.append_entry({"mean_peak": 70})
    assert [r["mean_peak"] for r in store.load_history()] == [70]
    kept = list(store.DATA_DIR.glob("history.corrupt-*.json"))
    assert len(kept) == 1 and kept[0].read_text() == "{this is not json"


def test_history_with_wrong_shape_is_treated_as_corrupt():
    store.DATA_DIR.mkdir(parents=True)
    store.HISTORY.write_text('{"a": 1}')
    assert store.load_history() == []
    assert store.append_entry({"k": 1}) and len(list(store.DATA_DIR.glob("history.corrupt-*"))) == 1


def test_history_ignores_non_dict_rows():
    store.DATA_DIR.mkdir(parents=True)
    store.HISTORY.write_text(json.dumps([{"a": 1}, 5, "x", None]))
    assert store.load_history() == [{"a": 1}]


def test_append_never_raises_when_storage_is_unwritable(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a folder")
    monkeypatch.setattr(store, "DATA_DIR", blocker / "sub")
    monkeypatch.setattr(store, "HISTORY", blocker / "sub" / "history.json")
    assert store.append_entry({"a": 1}) is False
    assert store.save_prom({"2026-01-01": {"pain": 1}}) is False


def test_failed_write_leaves_existing_history_intact(monkeypatch):
    store.append_entry({"keep": "me"})
    before = store.HISTORY.read_text()

    def fail(*a, **k):
        raise OSError("disk full")

    with monkeypatch.context() as m:  # scoped, so the isolation fixture stays active
        m.setattr(os, "replace", fail)
        assert store.append_entry({"lost": True}) is False
    assert store.HISTORY.read_text() == before
    assert not list(store.DATA_DIR.glob("*.tmp"))


def test_history_is_capped(monkeypatch):
    monkeypatch.setattr(store, "MAX_ROWS", 5)
    for i in range(9):
        store.append_entry({"i": i})
    assert [r["i"] for r in store.load_history()] == [4, 5, 6, 7, 8]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permissions")
def test_files_and_folders_are_owner_only():
    store.append_entry({"a": 1})
    store.tmp_dir()
    assert oct(store.DATA_DIR.stat().st_mode & 0o777) == "0o700"
    assert oct(store.HISTORY.stat().st_mode & 0o777) == "0o600"
    assert oct(store.TMP.stat().st_mode & 0o777) == "0o700"


# ---------------------------------------------------------------- diary
DIARY = {"2026-10-24": {"pain": 3, "fatigue": 4, "pain_location": ["leg"], "activities": {"walk": "easy"},
                        "sleep_hours": 7.5, "notes": "ok"}}


def test_diary_roundtrip_nested():
    assert store.load_prom() == ({}, True)
    assert store.save_prom(DIARY)
    assert store.load_prom() == (DIARY, True)
    assert store.prom_path().parent == store.DATA_DIR  # lives with the other data, not in the code folder


def test_diary_nan_is_sanitised():
    store.save_prom({"2026-10-24": {"sleep_hours": float("nan"), "pain": np.int64(2)}})
    assert store.load_prom()[0]["2026-10-24"] == {"sleep_hours": None, "pain": 2}


def test_corrupt_diary_is_reported_and_preserved():
    store.DATA_DIR.mkdir(parents=True)
    store.prom_path().write_text("garbage")
    entries, ok = store.load_prom()
    assert entries == {} and ok is False
    assert store.save_prom(DIARY)  # the new save must not destroy the evidence
    assert any(p.name.startswith("wellbeing_entries.corrupt-") and p.read_text() == "garbage"
               for p in store.DATA_DIR.iterdir())


def test_legacy_diary_next_to_source_code_is_migrated():
    legacy = store.LEGACY_PROM_FILES[0]
    legacy.write_text(json.dumps(DIARY))
    assert store.load_prom() == (DIARY, True)
    assert not legacy.exists() and store.prom_path().exists()


def test_legacy_diary_does_not_overwrite_newer_data():
    store.save_prom({"2026-10-25": {"pain": 1}})
    legacy = store.LEGACY_PROM_FILES[0]
    legacy.write_text(json.dumps(DIARY))
    assert "2026-10-25" in store.load_prom()[0]
    assert legacy.exists()  # left alone, but still audited and deleted below
    assert str(legacy) in [p for p, _ in store.audit()]


def test_delete_prom_removes_diary_and_legacy():
    store.save_prom(DIARY)
    store.LEGACY_PROM_FILES[1].write_text("{}")
    assert store.delete_prom()
    assert not store.prom_path().exists() and not store.LEGACY_PROM_FILES[1].exists()


# ---------------------------------------------------------------- audit + delete everything
def _populate():
    store.append_entry({"a": 1})
    store.save_prom(DIARY)
    (store.tmp_dir() / "result_x.mp4").write_bytes(b"x" * 10)
    store.LEGACY_PROM_FILES[0].write_text("{}")


def test_audit_lists_everything_held():
    _populate()
    names = {Path(p).name for p, _ in store.audit()}
    assert {"history.json", "wellbeing_entries.json", "result_x.mp4", "legacy_a.json"} <= names


def test_delete_everything_removes_all_and_audit_is_empty():
    _populate()
    n = store.delete_everything()
    assert n == 4 and store.audit() == []
    assert not store.DATA_DIR.exists() and not store.TMP.exists()
    assert not store.LEGACY_PROM_FILES[0].exists()


def test_delete_everything_on_empty_store_is_fine():
    assert store.delete_everything() == 0 and store.audit() == []


def test_delete_everything_reports_files_it_could_not_remove(monkeypatch):
    _populate()
    victim = store.prom_path()
    real_unlink = Path.unlink

    def picky(self, *a, **k):
        if self == victim:
            raise PermissionError("locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", picky)
    monkeypatch.setattr(store.shutil, "rmtree", lambda *a, **k: None)  # force the per-file path
    rep = store.delete_everything_report()
    assert rep["failed"] == [str(victim)] and rep["deleted"] == 3


# ---------------------------------------------------------------- stale original footage
def test_sweep_removes_old_originals_but_not_fresh_or_other_files():
    d = store.tmp_dir()
    old_in, old_raw = d / "in_abc.mp4", d / "result_abc.raw.mp4"
    fresh_in, other = d / "in_fresh.mp4", d / "notes.txt"
    for f in (old_in, old_raw, fresh_in, other):
        f.write_bytes(b"x")
    past = time.time() - 3600
    for f in (old_in, old_raw, other):
        os.utime(f, (past, past))
    assert store.sweep_stale(max_age_s=120) == 2
    assert not old_in.exists() and not old_raw.exists()
    assert fresh_in.exists() and other.exists()


def test_sweep_on_missing_folder_is_fine():
    assert store.sweep_stale() == 0


# ---------------------------------------------------------------- data lives outside the project (no .gitignore needed)
def test_default_data_folder_is_outside_the_project(monkeypatch, tmp_path):
    for env in ("XDG_DATA_HOME", "APPDATA"):
        monkeypatch.setenv(env, str(tmp_path / "appdata"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    d = store._default_data_dir().resolve()
    assert store._HERE.resolve() not in d.parents and d != store._HERE.resolve()
    assert d.name == "SpyMotion"


def test_env_var_overrides_the_data_folder(tmp_path):
    import importlib
    import os

    os.environ["SPYMOTION_DATA"] = str(tmp_path / "custom")
    try:
        fresh = importlib.reload(store)
        assert fresh.DATA_DIR == (tmp_path / "custom").resolve()
    finally:
        del os.environ["SPYMOTION_DATA"]
        importlib.reload(store)


def test_data_an_older_build_left_inside_the_project_is_moved_out():
    old = store.LEGACY_DATA_DIRS[0]
    old.mkdir()
    (old / "history.json").write_text(json.dumps([{"mean_peak": 71}]))
    (old / "wellbeing_entries.json").write_text(json.dumps({"2026-10-20": {"pain": 2}}))
    assert [r["mean_peak"] for r in store.load_history()] == [71]
    assert store.load_prom()[0] == {"2026-10-20": {"pain": 2}}
    assert not old.exists() and store.HISTORY.exists()


def test_migration_never_overwrites_newer_data():
    store.append_entry({"mean_peak": 99})
    old = store.LEGACY_DATA_DIRS[0]
    old.mkdir()
    (old / "history.json").write_text(json.dumps([{"mean_peak": 1}]))
    assert [r["mean_peak"] for r in store.load_history()] == [99]


def test_audit_and_delete_also_cover_old_in_project_folders():
    old = store.LEGACY_DATA_DIRS[1]
    old.mkdir()
    (old / "orphan.json").write_text("{}")
    assert "orphan.json" in {Path(p).name for p, _ in store.audit()}
    store.delete_everything()
    assert not old.exists()
