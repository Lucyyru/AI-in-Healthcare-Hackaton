"""
Video-level reliability tests: invalid / short / odd videos, MediaPipe losing the person,
crash containment, and the guarantee that original footage is never left on disk.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mp = pytest.importorskip("mediapipe")

import core  # noqa: E402
import pose_pipeline as pp  # noqa: E402
import robust  # noqa: E402
import scenarios  # noqa: E402
import store  # noqa: E402
from core import SignalConfig  # noqa: E402

CFG = SignalConfig()
W, H = 320, 240


def _video(path, n_frames, fps=30, size=(W, H)):
    if not scenarios.write_video(path, n_frames, fps, size):
        pytest.skip("this OpenCV build cannot write mp4 test videos")
    return str(path)


def _assert_refused(res, *needles):
    assert res.status == "rejected" and res.reasons and res.meta.get("failed") is True
    text = " ".join(res.reasons).lower()
    for n in needles:
        assert n.lower() in text, (n, res.reasons)


@pytest.fixture
def synthetic(monkeypatch):
    """Replace MediaPipe inference with supplied landmarks (deterministic, no real person needed)."""
    holder = {}

    def fake(path, fps, n_est, model_complexity, progress):
        return holder["lms"], (W, H)

    monkeypatch.setattr(pp, "extract_landmarks", fake)
    return holder


def _flex_run(tmp_path, lms, fps=30, **kw):
    path = _video(tmp_path / "v.mp4", len(lms), fps)
    out = tmp_path / "o.mp4"
    res = robust.safe_analyze(path, "knee", "auto", CFG, kw.pop("blur", True), 1, str(out))
    return res, out


# ---------------------------------------------------------------- invalid files never crash
def test_zero_byte_file(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    _assert_refused(robust.safe_analyze(str(p), "knee"), "empty")


def test_missing_file(tmp_path):
    _assert_refused(robust.safe_analyze(str(tmp_path / "nope.mp4"), "knee"), "empty")


def test_text_file_pretending_to_be_video(tmp_path):
    p = tmp_path / "fake.mp4"
    p.write_text("hello, I am not a video")
    _assert_refused(robust.safe_analyze(str(p), "knee"), "could not open")


def test_random_bytes(tmp_path):
    p = tmp_path / "noise.mov"
    p.write_bytes(np.random.default_rng(0).bytes(5000))
    res = robust.safe_analyze(str(p), "knee")
    assert res.status == "rejected" and res.reasons


def test_truncated_video(tmp_path):
    good = _video(tmp_path / "good.mp4", 90)
    data = open(good, "rb").read()
    p = tmp_path / "cut.mp4"
    p.write_bytes(data[: len(data) // 3])
    res = robust.safe_analyze(str(p), "knee", model_complexity=1)
    assert res.status == "rejected" and res.reasons


@pytest.mark.parametrize("frames", [1, 5, 30, 50])
def test_too_short_videos_get_a_clear_reason(tmp_path, frames):
    p = _video(tmp_path / "short.mp4", frames)
    _assert_refused(robust.safe_analyze(p, "knee"), "only")


# ---------------------------------------------------------------- real MediaPipe, no person
def test_person_never_found_is_refused_with_a_playable_overlay(tmp_path):
    p = _video(tmp_path / "blank.mp4", 120)  # 4 s of grey
    out = tmp_path / "o.mp4"
    res = robust.safe_analyze(p, "knee", "auto", CFG, True, 1, str(out))
    assert res.status == "rejected" and res.reasons and not res.meta.get("failed")
    cap = cv2.VideoCapture(str(out))
    assert out.stat().st_size > 1024 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 100
    ok, frame = cap.read()
    assert ok and frame.shape[:2] == (H, W)


@pytest.mark.parametrize("size", [(32, 32), (481, 241), (64, 480), (1280, 720)])
def test_odd_resolutions_do_not_crash(tmp_path, size):
    p = _video(tmp_path / "odd.mp4", 90, size=size)
    res = robust.safe_analyze(p, "knee", "auto", CFG, True, 1, str(tmp_path / "o.mp4"))
    assert res.status == "rejected" and res.reasons


@pytest.mark.parametrize("fps", [5, 12, 60])
def test_odd_frame_rates_do_not_crash(tmp_path, fps):
    p = _video(tmp_path / "f.mp4", fps * 4, fps=fps)
    res = robust.safe_analyze(p, "knee", "auto", CFG, True, 1, str(tmp_path / "o.mp4"))
    assert res.status == "rejected" and res.reasons


def test_only_the_bundled_model_is_ever_used_so_nothing_downloads(tmp_path, monkeypatch):
    seen = []
    Original = mp.solutions.pose.Pose

    class Spy(Original):
        def __init__(self, **kw):
            seen.append(kw["model_complexity"])
            super().__init__(**kw)

    monkeypatch.setattr(mp.solutions.pose, "Pose", Spy)
    p = _video(tmp_path / "b.mp4", 90)
    for requested in (0, 1, 2):
        robust.safe_analyze(p, "knee", "auto", CFG, False, requested, None)
    assert seen == [1, 1, 1]


# ---------------------------------------------------------------- MediaPipe loses the person (synthetic skeleton)
def test_clean_synthetic_run_end_to_end(tmp_path, synthetic):
    synthetic["lms"] = scenarios.synth_landmarks(scenarios.flex_curve(85, 660))
    res, out = _flex_run(tmp_path, synthetic["lms"])
    assert res.status in ("good", "fair") and res.n_reps == 5 and abs(res.mean_peak - 85) < 5
    assert res.meta["side"] in ("left", "right") and not res.meta.get("failed")
    cap = cv2.VideoCapture(str(out))
    assert abs(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 660) <= 2


@pytest.mark.parametrize("blur", [True, False])
def test_overlay_renders_with_and_without_face_blur(tmp_path, synthetic, blur):
    synthetic["lms"] = scenarios.synth_landmarks(scenarios.flex_curve(85, 300, n_reps=2))
    res, out = _flex_run(tmp_path, synthetic["lms"], blur=blur)
    assert out.exists() and out.stat().st_size > 1024


def test_person_lost_midway_is_flagged_not_hidden(tmp_path, synthetic):
    lms = scenarios.synth_landmarks(scenarios.flex_curve(85, 660))
    lms[250:420] = np.nan  # MediaPipe finds nobody for ~5.7 s
    synthetic["lms"] = lms
    res, out = _flex_run(tmp_path, lms)
    assert res.status in ("fair", "rejected")
    assert any("could not be tracked" in w for w in res.warnings + res.reasons)
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) >= 658  # overlay still complete


def test_person_lost_at_the_very_start_and_end(tmp_path, synthetic):
    lms = scenarios.synth_landmarks(scenarios.flex_curve(85, 660))
    lms[:60] = np.nan
    lms[-60:] = np.nan
    synthetic["lms"] = lms
    res, _ = _flex_run(tmp_path, lms)
    assert res.status in ("good", "fair") and res.n_reps >= 4


def test_flickering_confidence(tmp_path, synthetic):
    lms = scenarios.synth_landmarks(scenarios.flex_curve(85, 660))
    drop = np.random.default_rng(0).random(660) < 0.3
    lms[drop, :, 2] = 0.1
    synthetic["lms"] = lms
    res, _ = _flex_run(tmp_path, lms)
    assert res.status in ("good", "fair", "rejected") and np.isfinite(res.coverage)


def test_auto_side_picks_the_limb_that_is_visible(tmp_path, synthetic):
    flex = scenarios.flex_curve(85, 400)
    for visible in ("right", "left"):  # the other limb is (almost) invisible to the camera
        synthetic["lms"] = scenarios.synth_landmarks(flex, side=visible, other_side_vis=0.1)
        res, _ = _flex_run(tmp_path, synthetic["lms"])
        assert res.meta["side"] == visible


def test_front_on_camera_is_warned(tmp_path, synthetic):
    synthetic["lms"] = scenarios.synth_landmarks(scenarios.flex_curve(85, 660), front_on=True)
    res, _ = _flex_run(tmp_path, synthetic["lms"])
    assert any("side-on" in w for w in res.warnings)


# ---------------------------------------------------------------- crash containment
def test_unexpected_pipeline_exception_is_contained_and_not_leaked(tmp_path, monkeypatch):
    out = tmp_path / "o.mp4"

    def explode(*a, **k):
        out.with_suffix(".raw.mp4").write_bytes(b"half written")
        raise RuntimeError("secret internal detail /home/patient/video.mp4")

    monkeypatch.setattr(robust, "analyze_video", explode)
    p = _video(tmp_path / "v.mp4", 120)
    res = robust.safe_analyze(p, "knee", "auto", CFG, True, 1, str(out))
    assert res.status == "rejected" and res.reasons == [robust.GENERIC_REASON]
    assert "secret" not in " ".join(res.reasons)
    assert not out.with_suffix(".raw.mp4").exists()


def test_a_value_error_with_unfriendly_text_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(robust, "analyze_video", lambda *a, **k: (_ for _ in ()).throw(ValueError("zero-size array")))
    res = robust.safe_analyze(_video(tmp_path / "v.mp4", 120), "knee")
    assert res.reasons == [robust.GENERIC_REASON]


# ---------------------------------------------------------------- privacy: original footage never survives
def _tmp_files():
    return [p for p in store.TMP.rglob("*") if p.is_file()] if store.TMP.exists() else []


def test_upload_success_leaves_no_files_behind(tmp_path, synthetic):
    synthetic["lms"] = scenarios.synth_landmarks(scenarios.flex_curve(85, 660))
    data = open(_video(tmp_path / "v.mp4", 660), "rb").read()
    res, video = robust.analyze_upload(data, ".mp4", "abc", "knee", "auto", CFG, True)
    assert res.status in ("good", "fair") and video and len(video) > 1024
    assert _tmp_files() == []


def test_upload_of_garbage_leaves_no_files_behind():
    res, video = robust.analyze_upload(b"not a video at all", ".mp4", "abc", "knee", "auto", CFG, True)
    assert res.status == "rejected" and video is None
    assert _tmp_files() == []


class _StreamlitStop(BaseException):
    """Streamlit interrupts a running script with a BaseException (rerun/stop)."""


def test_original_is_removed_even_if_the_run_is_interrupted(monkeypatch, tmp_path):
    seen = {}

    def interrupt(src, *a, **k):
        seen["existed_during"] = (store.TMP / "in_abc.mp4").exists()
        raise _StreamlitStop()

    monkeypatch.setattr(robust, "safe_analyze", interrupt)
    with pytest.raises(_StreamlitStop):
        robust.analyze_upload(b"secret footage", ".mp4", "abc", "knee", "auto", CFG, True)
    assert seen["existed_during"] is True  # it did exist while analysing...
    assert _tmp_files() == []  # ...and is gone afterwards


def test_disk_full_while_storing_the_upload_is_handled(monkeypatch):
    def full(self, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(type(store.tmp_dir() / "x"), "write_bytes", full)
    res, video = robust.analyze_upload(b"x" * 100, ".mp4", "abc", "knee", "auto", CFG, True)
    assert res.status == "rejected" and "disk" in " ".join(res.reasons).lower() and video is None


def test_hostile_names_cannot_escape_the_temp_folder(monkeypatch):
    seen = []

    monkeypatch.setattr(robust, "safe_analyze", lambda src, *a, **k: seen.append(src) or robust.failed_result("x"))
    robust.analyze_upload(b"x", "/../../evil.sh", "../../etc/passwd", "knee", "auto", CFG, True)
    from pathlib import Path

    assert len(seen) == 1 and Path(seen[0]).parent == store.TMP and Path(seen[0]).suffix == ".mp4"
    assert ".." not in Path(seen[0]).name
