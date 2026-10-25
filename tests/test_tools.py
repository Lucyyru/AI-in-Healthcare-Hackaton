"""validate.py and stress_test.py: they must survive bad inputs and always produce a report."""
import sys

import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from spymotion import core  # noqa: E402
from spymotion import robust  # noqa: E402
from spymotion import scenarios  # noqa: E402
from tests import stress_test  # noqa: E402
import validate  # noqa: E402


def _good_result(peak=85.0):
    s, c = scenarios.movement_signal(peak, seed=1)
    r = core.process_signal(s, c, 30.0)
    r.meta = {"side": "right", "duration_s": 22.0}
    return r


def test_validate_reports_errors_and_refusals_without_crashing(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    labels = pd.DataFrame([
        {"file": "a.mp4", "exercise": "knee", "manual_peak_deg": 85, "condition": "standing"},
        {"file": "b.mp4", "exercise": "knee", "manual_peak_deg": 80, "condition": "wheelchair"},
        {"file": "missing.mp4", "exercise": "knee", "manual_peak_deg": 80, "condition": "wheelchair"},
        {"file": "c.mp4", "exercise": "knee", "manual_peak_deg": 80, "condition": None},
    ])
    labels.to_csv(tmp_path / "labels.csv", index=False)

    def fake(path, exercise, side, **k):
        if path.endswith("missing.mp4"):
            raise ValueError("Could not open the video.")
        if path.endswith("b.mp4"):
            return robust.failed_result("refused")
        return _good_result(85.0 if path.endswith("a.mp4") else 78.0)

    monkeypatch.setattr(validate, "analyze_video", fake)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--csv", str(tmp_path / "labels.csv"),
                                      "--videos", str(clips), "--out", str(tmp_path)])
    validate.main()
    md = (tmp_path / "validation_report.md").read_text(encoding="utf-8")
    assert "wheelchair" in md and "unspecified" in md and "Mean absolute error" in md
    out = pd.read_csv(tmp_path / "validation_report.csv")
    statuses = set(out.status)
    assert "error" in statuses and "rejected" in statuses and statuses & {"good", "fair"}


def test_validate_rejects_a_labels_file_missing_required_columns(tmp_path, monkeypatch):
    pd.DataFrame([{"file": "a.mp4"}]).to_csv(tmp_path / "l.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--csv", str(tmp_path / "l.csv"), "--videos", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        validate.main()
    assert "manual_peak_deg" in str(e.value)


def test_stress_test_plumbing_and_report(tmp_path):
    clip = tmp_path / "clip.mp4"
    assert scenarios.write_video(clip, 90, 30, (320, 240))
    calls = []

    def fake(path, ex, side, cfg, blur, cx, overlay):
        calls.append(path)
        return _good_result(85.0 if len(calls) == 1 else 85.0 + 8.0 * (len(calls) % 3))  # some drift, some agree

    rows, report = stress_test.run_stress(clip, "knee", "auto", tmp_path, analyze=fake)
    assert len(rows) == 1 + len(stress_test.PERTURBATIONS)
    text = report.read_text(encoding="utf-8")
    assert "Robustness stress test" in text and "mirrored" in text and "DIFFERENT NUMBER" in text


def test_stress_test_refuses_a_baseline_the_app_cannot_score(tmp_path):
    clip = tmp_path / "clip.mp4"
    scenarios.write_video(clip, 90, 30, (320, 240))
    with pytest.raises(SystemExit) as e:
        stress_test.run_stress(clip, "knee", "auto", tmp_path, analyze=lambda *a, **k: robust.failed_result("nope"))
    assert "baseline" in str(e.value)


def test_stress_test_variants_are_valid_videos(tmp_path):
    clip = tmp_path / "clip.mp4"
    scenarios.write_video(clip, 60, 30, (321, 241))  # odd size on purpose
    for name, fn in stress_test.PERTURBATIONS.items():
        dst = tmp_path / "v.mp4"
        n = stress_test._write_variant(clip, dst, fn)
        assert n > 0 and dst.stat().st_size > 0, name
        dst.unlink()
