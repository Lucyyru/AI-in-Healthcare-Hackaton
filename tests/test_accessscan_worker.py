"""The image cache must preserve geometry and never reuse another photograph."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from spymotion.accessscan.accessscan_worker import ImageMeasurements

@pytest.fixture
def detector(tmp_path, monkeypatch):
    from spymotion.accessscan import accessscan

    monkeypatch.syspath_prepend(str(accessscan.backend_directory()))
    left = np.array([[200, 800], [350, 550], [400, 466.6666666667]])
    right = np.array([[800, 800], [650, 550], [600, 466.6666666667]])
    calibration = {"image_size": [1000, 1000], "K": [[500, 0, 500], [0, 500, 300], [0, 0, 1]],
                   "up_camera": [0, -1, 0]}
    engine = ImageMeasurements(tmp_path / "output")
    with patch.object(engine, "recognize", return_value=((left, right), calibration,
                                                        {"sam2_s": 1, "geocalib_s": 1})) as recognize:
        yield engine, recognize


def request(tmp_path, content=b"first photo", height=1, wheelchair=0.65):
    image = tmp_path / "photo.jpg"
    image.write_bytes(content)
    return {"image": str(image), "height_m": height, "wheelchair_width_m": wheelchair}


def test_wheelchair_change_reuses_image_and_updates_decision(detector, tmp_path):
    engine, recognize = detector
    first = engine.analyze(request(tmp_path))
    second = engine.analyze(request(tmp_path, wheelchair=1.3))
    assert first["width_m"] == pytest.approx(1.2)
    assert second["width_m"] == pytest.approx(first["width_m"])
    assert first["passability"] == "Can pass!"
    assert second["passability"] == "Cannot pass"
    assert first["image_cache_hit"] is False
    assert second["image_cache_hit"] is True
    assert second["timings_s"]["sam2_s"] == second["timings_s"]["geocalib_s"] == 0
    recognize.assert_called_once()


def test_height_change_recalculates_width_without_recognition(detector, tmp_path):
    engine, recognize = detector
    first = engine.analyze(request(tmp_path))
    second = engine.analyze(request(tmp_path, height=1.5))
    assert second["width_m"] == pytest.approx(first["width_m"] * 1.5)
    assert second["image_cache_hit"] is True
    recognize.assert_called_once()


def test_new_photo_invalidates_image_cache(detector, tmp_path):
    engine, recognize = detector
    engine.analyze(request(tmp_path))
    result = engine.analyze(request(tmp_path, content=b"a different photo"))
    assert result["image_cache_hit"] is False
    assert recognize.call_count == 2


def test_outputs_are_overwritten_without_stale_visualizations(detector, tmp_path):
    engine, _ = detector
    for name in ("sidewalk_mask.png", "boundaries.png", "measurement.png"):
        (engine.output / name).write_bytes(b"old picture")
    original = engine.output / "keep.txt"
    original.write_text("unrelated file", encoding="utf-8")
    engine.analyze(request(tmp_path))
    result = engine.analyze(request(tmp_path, wheelchair=1.3))
    stored = json.loads((engine.output / "measurement.json").read_text(encoding="utf-8"))
    assert stored["passability"] == result["passability"] == "Cannot pass"
    assert not list(engine.output.glob("*.png"))
    assert stored["visualizations_generated"] is False
    assert original.read_text(encoding="utf-8") == "unrelated file"
