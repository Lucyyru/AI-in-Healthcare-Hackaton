"""Headless UI tests: every results state renders, failures are contained, privacy controls really work."""
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import core  # noqa: E402
import robust  # noqa: E402
import scenarios  # noqa: E402
import store  # noqa: E402

APP = str(Path(__file__).with_name("app.py"))


def entry(peak, seed, side="right", conf=0.95):
    s, c = scenarios.movement_signal(peak, seed=seed)
    r = core.process_signal(s, c * 0 + conf, 30.0)
    r.meta = {"exercise": "knee", "side": side, "fps": 30.0, "frames": len(s), "duration_s": len(s) / 30, "size": (480, 640)}
    return {"res": r, "video": None, "label": "x", "key": f"k{seed}{peak}{side}"}


def failed(reason="The video file is empty."):
    return {"res": robust.failed_result(reason), "video": None, "label": "x", "key": "failed"}


def launch(analysis=None, medication=False):
    at = AppTest.from_file(APP, default_timeout=90)
    if analysis is not None:
        at.session_state["analysis"] = analysis
    at.run()
    if medication:
        [t for t in at.sidebar.toggle if "medication" in t.label.lower()][0].set_value(True)
        at.run()
    return at


def texts(at):
    return " ".join(str(m.value) for m in at.markdown)


def no_errors(at):
    assert [str(e.value)[:150] for e in at.exception] == []


# ---------------------------------------------------------------- rendering
def test_empty_state_and_tabs():
    at = launch()
    no_errors(at)
    labels = [t.label for t in at.tabs]
    assert labels == ["Check my progress", "My history", "Daily wellbeing", "Data & safety"]


def test_medication_tab_is_optional_and_works_when_on():
    at = launch(medication=True)
    no_errors(at)
    assert "Medication & routines" in [t.label for t in at.tabs]


@pytest.mark.parametrize("name,before,after", [
    ("improved", entry(60, 1), entry(82, 2)),
    ("reduced", entry(82, 1), entry(60, 2)),
    ("similar", entry(70, 1), entry(71, 2)),
    ("single", None, entry(75, 3)),
    ("rejected", None, entry(75, 3, conf=0.2)),
    ("both_rejected", entry(75, 4, conf=0.2), entry(75, 3, conf=0.2)),
])
def test_result_states_render(name, before, after):
    state = {"ex_key": "knee", "after": after}
    if before:
        state["before"] = before
    at = launch(state)
    no_errors(at)
    assert "SpyMotion" in texts(at)


def test_improvement_is_announced_with_uncertainty():
    at = launch({"ex_key": "knee", "before": entry(60, 1), "after": entry(82, 2)})
    t = texts(at)
    assert "Improved" in t and "95% range" in t


def test_different_limbs_cannot_be_compared():
    at = launch({"ex_key": "knee", "before": entry(60, 1, side="left"), "after": entry(82, 2)})
    no_errors(at)
    assert "Different limbs tracked" in texts(at)


@pytest.mark.parametrize("position", ["before", "after", "both"])
def test_a_failed_video_does_not_break_the_page_or_cause_false_warnings(position):
    state = {"ex_key": "knee", "after": entry(80, 2)}
    if position in ("before", "both"):
        state["before"] = failed("The video file is empty.")
    if position in ("after", "both"):
        state["after"] = failed("Could not open the video.")
        if position == "after":
            state["before"] = entry(60, 1)
    at = launch(state)
    no_errors(at)
    t = texts(at)
    assert "We can’t score this video reliably" in t
    assert "Different limbs tracked" not in t


# ---------------------------------------------------------------- privacy controls
def _populate():
    store.append_entry({"exercise": "knee", "mean_peak": 80})
    store.save_prom({date.today().isoformat(): {"pain": 3, "medication": "Taken as prescribed"}})
    (store.tmp_dir() / "result_x.mp4").write_bytes(b"x")


def test_delete_everything_button_wipes_history_diary_and_temp_files():
    _populate()
    assert len(store.audit()) == 3
    at = launch()
    [b for b in at.sidebar.button if "Delete everything" in b.label][0].click().run()
    no_errors(at)
    assert store.audit() == [] and store.load_history() == [] and store.load_prom() == ({}, True)


def test_data_tab_audit_lists_the_diary():
    _populate()
    at = launch()
    cells = " ".join(str(v) for d in at.dataframe for v in d.value.astype(str).to_numpy().ravel())
    assert "wellbeing_entries.json" in cells and "history.json" in cells


def test_data_tab_says_nothing_is_stored_when_clean():
    at = launch()
    assert "No files stored" in texts(at)


def test_legacy_diary_in_the_code_folder_is_moved_into_the_data_folder():
    legacy = store.LEGACY_PROM_FILES[0]
    legacy.write_text(json.dumps({"2026-10-20": {"pain": 2}}))
    at = launch()
    no_errors(at)
    assert not legacy.exists() and store.load_prom()[0] == {"2026-10-20": {"pain": 2}}


def test_app_never_writes_health_data_into_the_code_folder():
    _populate()
    launch()
    code_dir = Path(APP).parent
    stray = [p.name for p in code_dir.iterdir() if "prom" in p.name.lower() or p.suffix == ".mp4"]
    assert stray == []


# ---------------------------------------------------------------- diary + movement on one timeline
def test_history_shows_movement_next_to_how_you_felt():
    d1, d2 = date.today() - timedelta(days=1), date.today()
    store.DATA_DIR.mkdir(parents=True)
    store.HISTORY.write_text(json.dumps([
        {"saved_at": f"{d1}T10:00:00", "exercise": "knee", "side": "right", "label": "a", "mean_peak": 70, "n_reps": 5},
        {"saved_at": f"{d2}T10:00:00", "exercise": "knee", "side": "right", "label": "b", "mean_peak": 78, "n_reps": 5},
    ]))
    store.save_prom({d1.isoformat(): {"pain": 5, "fatigue": 6}, d2.isoformat(): {"pain": 2, "fatigue": 3}})
    at = launch()
    no_errors(at)
    assert "Movement and how you felt" in texts(at)


def test_history_without_overlapping_diary_days_just_hides_the_chart():
    store.append_entry({"exercise": "knee", "side": "right", "label": "a", "mean_peak": 70, "n_reps": 5})
    store.save_prom({"2020-01-01": {"pain": 1}})
    at = launch()
    no_errors(at)
    assert "Movement and how you felt" not in texts(at)


def test_history_tab_survives_incomplete_or_odd_rows():
    store.DATA_DIR.mkdir(parents=True)
    store.HISTORY.write_text(json.dumps([
        {"mean_peak": 70},                                             # no date, no exercise
        {"saved_at": "not a date", "mean_peak": 71, "exercise": "knee"},
        {"saved_at": "2026-10-20T10:00:00", "mean_peak": None},
        {"saved_at": "2026-10-21T10:00:00", "exercise": "knee", "mean_peak": 75},
    ]))
    at = launch()
    no_errors(at)
