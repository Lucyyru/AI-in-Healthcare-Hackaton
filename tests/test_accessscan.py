"""AccessScan's real integration contract and result states."""
from io import BytesIO
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock, patch

from PIL import Image
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import accessscan


def image_bytes():
    output = BytesIO()
    Image.new("RGB", (20, 30), "white").save(output, format="PNG")
    return output.getvalue()


def test_backend_directory_at_repository_root(tmp_path, monkeypatch):
    backend = tmp_path / "SidewalkDetection"
    backend.mkdir()
    (backend / "pipeline.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(accessscan, "APP_DIR", tmp_path)
    assert accessscan.backend_directory() == backend


def test_model_environment_is_found_without_personal_configuration(tmp_path, monkeypatch):
    backend = tmp_path / "SidewalkDetection"
    executable = backend / ".venv" / ("Scripts/python.exe" if accessscan.os.name == "nt" else "bin/python")
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    (backend / "pipeline.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(accessscan, "APP_DIR", tmp_path)
    monkeypatch.delenv("SIDEWALK_PYTHON", raising=False)
    assert Path(accessscan.backend_python()) == executable


def test_relative_model_configuration_is_resolved_from_app_folder(tmp_path, monkeypatch):
    (tmp_path / "accessscan.local.json").write_text(
        json.dumps({"backend_python": "SidewalkDetection/.venv/Scripts/python.exe"}), encoding="utf-8"
    )
    monkeypatch.setattr(accessscan, "APP_DIR", tmp_path)
    monkeypatch.delenv("SIDEWALK_PYTHON", raising=False)
    assert Path(accessscan.backend_python()) == (tmp_path / "SidewalkDetection/.venv/Scripts/python.exe").resolve()


def test_bridge_converts_units_reads_results_and_removes_upload(tmp_path):
    paths = []
    worker = Mock()
    worker.poll.return_value = None

    def send(line):
        request = json.loads(line)
        paths.append(Path(request["image"]))
        assert paths[0].read_bytes() == b"photo"
        assert request["height_m"] == 1.24
        assert request["wheelchair_width_m"] == 0.65
        Path(request["response"]).write_text(
            json.dumps({"ok": True, "report": {"width_m": 1.58, "passability": "Can pass!"}}),
            encoding="utf-8",
        )

    worker.stdin.write.side_effect = send
    with patch.object(accessscan, "backend_directory", return_value=tmp_path), \
         patch.object(accessscan, "_model_worker", return_value=worker):
        report = accessscan.analyze_access_image(b"photo", ".jpg", 124, 65)
    assert report["width_m"] == 1.58
    assert not paths[0].exists()


def test_failed_model_does_not_return_previous_result(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "measurement.json").write_text('{"width_m": 9}', encoding="utf-8")
    failed = Mock()
    failed.poll.return_value = 1
    with patch.object(accessscan, "backend_directory", return_value=tmp_path), \
         patch.object(accessscan, "_model_worker", return_value=failed):
        with pytest.raises(RuntimeError, match="model process stopped"):
            accessscan.analyze_access_image(b"photo", ".jpg", 124, 65)


def test_timeout_has_a_readable_message(tmp_path):
    worker = Mock()
    worker.poll.return_value = None
    with patch.object(accessscan, "backend_directory", return_value=tmp_path), \
         patch.object(accessscan, "_model_worker", return_value=worker), \
         patch.object(accessscan.time, "monotonic", side_effect=[0, 301]), \
         patch.object(accessscan, "stop_worker") as stop:
        with pytest.raises(TimeoutError, match="timed out"):
            accessscan.analyze_access_image(b"photo", ".jpg", 124, 65)
        stop.assert_called_once()


def test_model_process_is_kept_for_the_next_request(tmp_path):
    worker = Mock()
    worker.poll.return_value = None
    with patch.object(accessscan, "_WORKER", None), \
         patch.object(accessscan.subprocess, "Popen", return_value=worker) as start:
        first = accessscan._model_worker(tmp_path)
        second = accessscan._model_worker(tmp_path)
        assert first is second is worker
        start.assert_called_once()


@pytest.mark.parametrize("height,width", [(0, 65), (124, -1), (float("nan"), 65)])
def test_invalid_measurements_are_rejected(height, width):
    with pytest.raises(ValueError, match="positive finite"):
        accessscan.analyze_access_image(b"photo", ".jpg", height, width)


def uploaded_photo(*args, **kwargs):
    if kwargs.get("key") == "accessscan_image":
        photo = BytesIO(image_bytes())
        photo.name = "photo.png"
        return photo
    return None


@pytest.mark.parametrize("width,verdict", [
    (1.5848951751805072, "Can pass!"),
    (0.65, "Cannot pass"),
    (None, "Unable to determine: no valid passage width."),
])
def test_ui_result_states_and_changed_inputs(width, verdict):
    report = {"width_m": width, "passability": verdict, "warnings": []}
    app_path = str(Path(__file__).with_name("app.py"))
    with patch.object(st, "file_uploader", side_effect=uploaded_photo), \
         patch.object(accessscan, "analyze_access_image", return_value=report) as analyze:
        app = AppTest.from_file(app_path, default_timeout=60).run()
        assert not app.exception
        app.number_input(key="accessscan_wheelchair_width").set_value(65.0).run()
        analyze.assert_not_called()
        app.button(key="accessscan_analyse").click().run()
        assert not app.exception
        analyze.assert_called_once_with(image_bytes(), ".png", 124.0, 65.0)
        results = [element.value for element in app.markdown if 'class="accessscan-result"' in element.value]
        assert len(results) == 1
        assert (verdict if width is not None else "Unable to determine") in results[0]
        assert "font-size:clamp(40px,6vw,72px)" in results[0]
        assert "Passage width" not in [element.label for element in app.metric]
        assert len(app.get("imgs")) == 1  # Original upload only; no boundary visualization.
        assert not any("How AccessScan works" in element.value for element in app.markdown)
        app.number_input(key="accessscan_camera_height").set_value(125.0).run()
        assert not app.exception
        assert not any('class="accessscan-result"' in element.value for element in app.markdown)
        analyze.assert_called_once()


def test_ui_model_failure_is_shown_without_a_result():
    with patch.object(st, "file_uploader", side_effect=uploaded_photo), \
         patch.object(accessscan, "analyze_access_image", side_effect=RuntimeError("No passage detected")):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60).run()
        app.button(key="accessscan_analyse").click().run()
        assert not app.exception
        results = [element.value for element in app.markdown if 'class="accessscan-result"' in element.value]
        assert len(results) == 1 and "Unable to determine" in results[0]
        assert "No passage detected" not in results[0]
        assert "Passage width" not in [element.label for element in app.metric]
