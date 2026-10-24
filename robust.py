"""
robust.py - the safety net between the UI and the video pipeline.

* safe_analyze()   never raises for bad input: it returns a 'rejected' SignalResult with a
                   plain-language reason (empty / unreadable / too short / pipeline failure).
* analyze_upload() writes the upload to a private temp file, analyses it, and ALWAYS removes the
                   original footage and any half-written video, even if the analysis crashes or
                   Streamlit interrupts the run. Returns (result, result_video_bytes_or_None).
"""
from __future__ import annotations

import logging
from pathlib import Path

import store
from core import CFG, SignalConfig, SignalResult
from pose_pipeline import analyze_video, probe_fps

log = logging.getLogger("spymotion")

MIN_SECONDS = 2.0
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
_FRIENDLY_PREFIXES = ("Could not open the video", "The video contained no readable frames")
GENERIC_REASON = (
    "Something went wrong while analysing this video. Try re-saving it as an .mp4 and filming "
    "again with your whole body in view."
)


def failed_result(reason: str, side: str = "auto", fps: float = 30.0) -> SignalResult:
    """A rejected result the UI can render like any other refusal (no score, clear reason)."""
    r = SignalResult(status="rejected", fps=float(fps))
    r.reasons.append(reason)
    r.meta = {"exercise": None, "side": side, "fps": float(fps), "frames": 0,
              "duration_s": 0.0, "size": (0, 0), "failed": True}
    return r


def _too_short(seconds: float) -> str:
    return (f"The video is only {seconds:.1f} s long. Record at least {MIN_SECONDS:.0f} s "
            "(ideally 15\u201330 s) covering 3\u20135 repetitions.")


def safe_analyze(
    video_path: str,
    exercise_key: str,
    side: str = "auto",
    cfg: SignalConfig = CFG,
    blur_face: bool = True,
    model_complexity: int = 1,
    overlay_path: str | None = None,
    progress=None,
) -> SignalResult:
    p = Path(video_path)
    try:
        if not p.exists() or p.stat().st_size == 0:
            return failed_result("The video file is empty. Please choose the file again.", side)
        try:
            fps, n_frames = probe_fps(str(p))
        except ValueError as e:
            return failed_result(str(e), side)
        if n_frames > 0 and n_frames / fps < MIN_SECONDS:
            return failed_result(_too_short(n_frames / fps), side, fps)

        res = analyze_video(str(p), exercise_key, side, cfg, blur_face, model_complexity,
                            overlay_path, progress)

        dur = float(res.meta.get("duration_s", 0.0))
        if dur < MIN_SECONDS:  # container metadata can lie about length
            res.status = "rejected"
            res.reasons.insert(0, _too_short(dur))
        return res
    except ValueError as e:
        msg = str(e)
        return failed_result(msg if msg.startswith(_FRIENDLY_PREFIXES) else GENERIC_REASON, side)
    except Exception:  # noqa: BLE001 - one broken video must never take the app down
        log.exception("analysis failed")
        return failed_result(GENERIC_REASON, side)
    finally:
        if overlay_path:  # a crash can leave a half-written video behind
            Path(overlay_path).with_suffix(".raw.mp4").unlink(missing_ok=True)


def analyze_upload(
    data: bytes,
    suffix: str,
    token: str,
    exercise_key: str,
    side: str,
    cfg: SignalConfig,
    blur_face: bool,
    model_complexity: int = 1,
    progress=None,
) -> tuple[SignalResult, bytes | None]:
    """Analyse uploaded bytes. The original footage never outlives this call."""
    suffix = suffix.lower() if suffix.lower() in ALLOWED_SUFFIXES else ".mp4"
    token = "".join(ch for ch in str(token) if ch.isalnum())[:16] or "x"
    tmp = store.tmp_dir()
    src = tmp / f"in_{token}{suffix}"
    out = tmp / f"result_{token}.mp4"
    try:
        try:
            src.write_bytes(data)
        except OSError:
            log.exception("could not write temp file")
            return failed_result("Could not store the video temporarily (disk full or no permission).", side), None
        res = safe_analyze(str(src), exercise_key, side, cfg, blur_face, model_complexity, str(out), progress)
        video = None
        if out.exists():
            blob = out.read_bytes()
            video = blob if len(blob) > 1024 else None  # an empty/garbage file is not a video
        return res, video
    finally:
        # privacy: original footage and the result file are removed no matter what happened
        for f in (src, out, out.with_suffix(".raw.mp4")):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                log.error("could not remove %s", f)
