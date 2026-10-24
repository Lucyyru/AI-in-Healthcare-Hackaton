"""
pose_pipeline.py - video in, SignalResult + privacy-preserving overlay video out.

Runs entirely on-device with MediaPipe Pose. Only the standard model (model_complexity=1)
ships inside the pip package. The 'lite' (0) and 'heavy' (2) models are downloaded from Google
Cloud Storage the first time they are used, which would break the on-device / offline promise,
so any other value is forced to 1. Inference runs once per frame; the overlay is rendered from
the stored landmarks, so no second pass of the model is needed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

import core
from core import CFG, EXERCISES, SignalConfig, SignalResult

MAX_W = 640  # frames are downscaled for speed; angles are scale-invariant
MAX_SECONDS = 90
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, ORANGE, WHITE, GREY = (90, 200, 60), (0, 150, 255), (255, 255, 255), (200, 200, 200)

Progress = Optional[Callable[[float, str], None]]
BUNDLED_COMPLEXITY = 1  # the only pose model that needs no download


# ---------------------------------------------------------------- video helpers
def probe_fps(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video. Try an .mp4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 30.0
    return float(fps), int(n or 0)


def _even(x: int) -> int:
    return max(2, x - (x % 2))


def iter_frames(path: str, max_frames: int):
    """Yield frames downscaled to MAX_W with even dimensions (needed for H.264)."""
    cap = cv2.VideoCapture(path)
    size = None
    i = 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if size is None:
            h, w = frame.shape[:2]
            scale = min(1.0, MAX_W / w)
            size = (_even(int(round(w * scale))), _even(int(round(h * scale))))
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        yield frame
        i += 1
    cap.release()


def _to_h264(src: str, dst: str) -> str:
    """Re-encode to browser-playable H.264 and strip audio (voices are personal data)."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [exe, "-y", "-i", src, "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "24", "-movflags", "+faststart", dst],
            check=True, capture_output=True,
        )
        Path(src).unlink(missing_ok=True)
        return dst
    except Exception:
        shutil.move(src, dst)
        return dst


# ---------------------------------------------------------------- pose extraction
def extract_landmarks(path: str, fps: float, n_est: int, model_complexity: int, progress: Progress):
    """Return landmarks (N, 33, 3) = x, y (normalised), visibility; and the frame size."""
    import mediapipe as mp

    model_complexity = BUNDLED_COMPLEXITY  # never trigger a network download
    mp_pose = mp.solutions.pose
    max_frames = int(fps * MAX_SECONDS)
    out, size = [], None
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for i, frame in enumerate(iter_frames(path, max_frames)):
            if size is None:
                size = (frame.shape[1], frame.shape[0])
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                out.append([[lm.x, lm.y, lm.visibility] for lm in res.pose_landmarks.landmark])
            else:
                out.append(np.full((33, 3), np.nan))
            if progress and i % 10 == 0 and n_est:
                progress(0.7 * min(1.0, i / n_est), "Tracking your movement")
    if not out:
        raise ValueError("The video contained no readable frames.")
    return np.asarray(out, float), size


def compute_angles(lms: np.ndarray, ex: core.Exercise, side: str, size: tuple[int, int]):
    """Angle in pixel space (so non-square frames are not distorted) + per-frame confidence."""
    ids = core.landmark_ids(ex, side)
    w, h = size
    pts = lms[:, ids, :2] * np.array([w, h])
    raw = core.joint_angle(pts[:, 0], pts[:, 1], pts[:, 2])
    conf = np.nan_to_num(lms[:, ids, 2], nan=0.0).min(axis=1)
    raw = np.where(conf > 0, raw, np.nan)
    return core.movement_value(raw, ex), conf


def view_ratio(lms: np.ndarray, size: tuple[int, int]) -> float | None:
    """Shoulder-width / torso-length. Near 0 = filmed side-on; near 1 = filmed from the front."""
    w, h = size
    ids = [11, 12, 23, 24]
    ok = np.nan_to_num(lms[:, ids, 2], nan=0.0).min(axis=1) > 0.5
    if ok.sum() < 10:
        return None
    p = lms[ok][:, :, :2] * np.array([w, h])
    sep = np.abs(p[:, 11, 0] - p[:, 12, 0])
    torso = np.linalg.norm(p[:, [11, 12]].mean(1) - p[:, [23, 24]].mean(1), axis=1)
    return float(np.median(sep / np.maximum(torso, 1e-6)))


# ---------------------------------------------------------------- overlay rendering
def _pixelate_face(frame: np.ndarray, lm: np.ndarray, last_box):
    h, w = frame.shape[:2]
    pts = lm[:11]
    ok = np.isfinite(pts[:, 0]) & (pts[:, 2] > 0.3)
    box = last_box
    if ok.sum() >= 2:
        xs, ys = pts[ok, 0] * w, pts[ok, 1] * h
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.9 + 14
        cx, cy = xs.mean(), ys.mean()
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    if box is None:
        return last_box
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x1 - x0 > 4 and y1 - y0 > 4:
        roi = frame[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 10), max(1, (y1 - y0) // 10)), interpolation=cv2.INTER_LINEAR)
        frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return box


def render_overlay(path, out_path, lms, ex, side, size, fps, res: SignalResult, blur_face, progress: Progress):
    import mediapipe as mp

    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    ids = core.landmark_ids(ex, side)
    w, h = size
    raw_path = str(Path(out_path).with_suffix(".raw.mp4"))
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rep_frames = np.array([r["frame"] for r in res.reps], int)
    best = np.maximum.accumulate(np.nan_to_num(res.smooth, nan=0.0))
    last_box = None
    n = len(lms)
    for i, frame in enumerate(iter_frames(path, n)):
        if i >= n:
            break
        lm = lms[i]
        have = np.isfinite(lm[:, 0]).any()
        if blur_face and have:
            last_box = _pixelate_face(frame, lm, last_box)
        ok = bool(res.valid[i]) if i < len(res.valid) else False
        col = GREEN if ok else ORANGE
        if have:
            px = (lm[:, :2] * np.array([w, h])).astype(int)
            for a, b in connections:
                if lm[a, 2] > 0.5 and lm[b, 2] > 0.5:
                    cv2.line(frame, tuple(px[a]), tuple(px[b]), GREY, 1, cv2.LINE_AA)
            a, b, c = ids
            cv2.line(frame, tuple(px[a]), tuple(px[b]), col, 5, cv2.LINE_AA)
            cv2.line(frame, tuple(px[b]), tuple(px[c]), col, 5, cv2.LINE_AA)
            for k in ids:
                cv2.circle(frame, tuple(px[k]), 7, WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(px[k]), 7, col, 2, cv2.LINE_AA)
            if ok and np.isfinite(res.smooth[i]):
                tx, ty = px[b][0] + 12, px[b][1] - 12
                cv2.putText(frame, f"{res.smooth[i]:.0f} deg", (tx, ty), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"{res.smooth[i]:.0f} deg", (tx, ty), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
        # HUD
        reps_so_far = int((rep_frames <= i).sum())
        lines = [
            (f"Reps: {reps_so_far}", WHITE),
            (f"Best so far: {best[i]:.0f} deg" if i < len(best) else "", WHITE),
            ("Tracking OK" if ok else ("Low confidence - ignored" if have else "No person found"), col),
        ]
        ov = frame.copy()
        cv2.rectangle(ov, (6, 6), (270, 12 + 28 * len(lines) + 6), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        for k, (txt, c_) in enumerate(lines):
            cv2.putText(frame, txt, (14, 34 + 28 * k), FONT, 0.65, c_, 2, cv2.LINE_AA)
        writer.write(frame)
        if progress and i % 10 == 0:
            progress(0.7 + 0.3 * min(1.0, i / max(1, n)), "Drawing your result video")
    writer.release()
    return _to_h264(raw_path, out_path)


# ---------------------------------------------------------------- public entry point
def analyze_video(
    video_path: str,
    exercise_key: str,
    side: str = "auto",
    cfg: SignalConfig = CFG,
    blur_face: bool = True,
    model_complexity: int = 1,
    overlay_path: str | None = None,
    progress: Progress = None,
) -> SignalResult:
    ex = EXERCISES[exercise_key]
    fps, n_est = probe_fps(video_path)
    lms, size = extract_landmarks(video_path, fps, n_est, model_complexity, progress)

    angles = {s: compute_angles(lms, ex, s, size) for s in ("left", "right")}
    if side == "auto":
        side = max(("left", "right"), key=lambda s: float(((angles[s][1] >= cfg.conf_thresh)).mean()))
    movement, conf = angles[side]
    vr = view_ratio(lms, size)

    res = core.process_signal(movement, conf, fps, cfg, view_ratio=vr)
    res.meta = {"exercise": exercise_key, "side": side, "fps": fps, "frames": len(lms),
                "duration_s": len(lms) / fps, "size": size}
    if overlay_path:
        render_overlay(video_path, overlay_path, lms, ex, side, size, fps, res, blur_face, progress)
    if progress:
        progress(1.0, "Done")
    return res
