"""
scenarios.py - deterministic synthetic data for tests and reproducible demos.

Everything here is generated from fixed seeds, so results are identical on every machine.
Run `python scenarios.py` to print the regression table (what each scenario must produce).
"""
from __future__ import annotations

import numpy as np

FPS = 30.0


def movement_signal(peak, n_reps=5, seconds=22.0, noise=2.0, seed=0, fps=FPS, period=3.0, rest=1.5):
    """Raised-cosine repetitions of height `peak` degrees, plus Gaussian noise. Returns (signal, conf)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    s = np.zeros_like(t)
    for k in range(n_reps):
        t0 = rest + k * period
        m = (t >= t0) & (t < t0 + period)
        s[m] = peak * (1 - np.cos(2 * np.pi * (t[m] - t0) / period)) / 2
    return s + rng.normal(0, noise, len(t)), np.full(len(t), 0.95)


def flex_curve(peak, n_frames, fps=FPS, n_reps=5, period=3.0, rest=1.5):
    t = np.arange(n_frames) / fps
    f = np.zeros(n_frames)
    for k in range(n_reps):
        t0 = rest + k * period
        m = (t >= t0) & (t < t0 + period)
        f[m] = peak * (1 - np.cos(2 * np.pi * (t[m] - t0) / period)) / 2
    return f


def synth_landmarks(flex, side="right", vis=0.95, other_side_vis=None, front_on=False):
    """MediaPipe-shaped landmarks (N,33,3) for a person whose knee flexes by `flex` degrees."""
    n = len(flex)
    lms = np.zeros((n, 33, 3))
    lms[:, :, 0] = 0.5
    lms[:, :, 1] = 0.5
    lms[:, :, 2] = vis
    for i in range(11):  # face landmarks, so face blurring has something to find
        lms[:, i, 0] = 0.5 + 0.01 * (i % 3 - 1)
        lms[:, i, 1] = 0.15 + 0.01 * (i % 4)
    lx, rx = (0.35, 0.65) if front_on else (0.5, 0.5)
    lms[:, 11, :2] = [lx, 0.30]
    lms[:, 12, :2] = [rx, 0.30]
    hx_l, hx_r = (0.40, 0.60) if front_on else (0.5, 0.5)
    lms[:, 23, :2] = [hx_l, 0.55]
    lms[:, 24, :2] = [hx_r, 0.55]
    a = np.radians(flex)
    for knee, ankle in ((25, 27), (26, 28)):
        lms[:, knee, 0], lms[:, knee, 1] = 0.5, 0.75
        lms[:, ankle, 0] = 0.5 - 0.2 * np.sin(a)
        lms[:, ankle, 1] = 0.75 + 0.2 * np.cos(a)
    if other_side_vis is not None:
        other = [12, 24, 26, 28, 14, 16] if side == "left" else [11, 23, 25, 27, 13, 15]
        lms[:, other, 2] = other_side_vis
    return lms


def write_video(path, n_frames, fps=30, size=(320, 240), gray=90):
    """Write a plain mp4 (no person). Returns True if a non-empty file was produced."""
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frame = np.full((size[1], size[0], 3), gray, np.uint8)
    for _ in range(n_frames):
        w.write(frame)
    w.release()
    from pathlib import Path

    return Path(path).exists() and Path(path).stat().st_size > 0


# ------------------------------------------------------------------ regression scenarios
def _dropout(conf, a, b, val=0.1):
    conf = conf.copy()
    conf[int(a * FPS): int(b * FPS)] = val
    return conf


SCENARIOS = {
    # name: (signal kwargs, confidence edit, view_ratio, expected)
    "clean_80": (dict(peak=80), None, None, dict(status={"good"}, reps=(5, 5), peak=(76, 84))),
    "noisy_80": (dict(peak=80, noise=4, seed=3), None, None, dict(status={"good", "fair"}, reps=(5, 5), peak=(74, 86))),
    "three_reps": (dict(peak=70, n_reps=3, seconds=14), None, None, dict(status={"good"}, reps=(3, 3), peak=(66, 74))),
    "two_reps_only": (dict(peak=70, n_reps=2, seconds=10), None, None, dict(status={"fair"}, reps=(2, 2), peak=(66, 74))),
    "one_rep": (dict(peak=70, n_reps=1, seconds=6), None, None, dict(status={"rejected"})),
    "tiny_motion": (dict(peak=5, n_reps=6), None, None, dict(status={"rejected"})),
    "low_confidence": (dict(peak=80), lambda c: c * 0 + 0.2, None, dict(status={"rejected"})),
    "half_lost": (dict(peak=80), lambda c: _dropout(c, 0, 11), None, dict(status={"rejected"})),
    "person_lost_4s": (dict(peak=80), lambda c: _dropout(c, 10, 14), None, dict(status={"fair"}, reps=(3, 4))),
    "front_on_camera": (dict(peak=80), None, 0.9, dict(status={"fair"}, reps=(5, 5))),
    "flat_line": (dict(peak=0), None, None, dict(status={"rejected"})),
}


def build(name):
    kw, conf_edit, view, _ = SCENARIOS[name]
    s, c = movement_signal(**kw)
    if conf_edit:
        c = conf_edit(c)
    return s, c, view


def run(name):
    import core

    s, c, view = build(name)
    return core.process_signal(s, c, FPS, view_ratio=view)


def check(name):
    """Return a list of problems (empty = scenario behaves as specified)."""
    exp, res, problems = SCENARIOS[name][3], run(name), []
    if res.status not in exp["status"]:
        problems.append(f"status {res.status!r} not in {exp['status']}")
    if "reps" in exp and not (exp["reps"][0] <= res.n_reps <= exp["reps"][1]):
        problems.append(f"{res.n_reps} reps not in {exp['reps']}")
    if "peak" in exp and not (exp["peak"][0] <= res.mean_peak <= exp["peak"][1]):
        problems.append(f"peak {res.mean_peak:.1f} not in {exp['peak']}")
    return problems


if __name__ == "__main__":
    print(f"{'scenario':18s} {'status':9s} {'reps':>4s} {'peak':>6s}  verdict")
    bad = 0
    for n in SCENARIOS:
        r, p = run(n), check(n)
        bad += bool(p)
        print(f"{n:18s} {r.status:9s} {r.n_reps:4d} {r.mean_peak:6.1f}  {'OK' if not p else 'FAIL ' + '; '.join(p)}")
    raise SystemExit(1 if bad else 0)
