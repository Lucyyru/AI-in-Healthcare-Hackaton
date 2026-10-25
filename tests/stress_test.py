"""
stress_test.py - robustness / fairness evidence from ONE good clip.

Takes a clip the app scores well, then re-analyses degraded copies (dark, blurry, noisy, low
resolution, legs cropped out, mirrored...). For each it reports whether SpyMotion still agrees
with the original, refuses to score, or (the case to worry about) gives a different number.

    python tests/stress_test.py --video clips/knee_standing_01.mp4 --exercise knee [--side right]

Writes stress_report.md (shown in the app's Data & safety tab). Copies live in a temporary
folder that is deleted afterwards. This tests robustness to image quality; it does NOT replace
testing on real people in different postures (wheelchair, seated, assisted).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from spymotion.core import CFG
from spymotion.robust import safe_analyze

AGREE_WITHIN = 5.0  # degrees: the app's own threshold for a "meaningful" change


def _darken(f, k):
    return np.clip(f.astype(np.float32) * k, 0, 255).astype(np.uint8)


def _noise(f, sigma):
    return np.clip(f.astype(np.float32) + np.random.default_rng(0).normal(0, sigma, f.shape), 0, 255).astype(np.uint8)


def _lowres(f, width):
    h, w = f.shape[:2]
    small = cv2.resize(f, (width, max(2, int(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _lowcontrast(f):
    return np.clip(127 + (f.astype(np.float32) - 127) * 0.4, 0, 255).astype(np.uint8)


def _crop_bottom(f, frac=0.30):
    return f[: int(f.shape[0] * (1 - frac))]


PERTURBATIONS = {
    "dim light (x0.5)": lambda f: _darken(f, 0.5),
    "very dim light (x0.25)": lambda f: _darken(f, 0.25),
    "over-exposed (x1.8)": lambda f: _darken(f, 1.8),
    "low contrast": _lowcontrast,
    "blur (moderate)": lambda f: cv2.GaussianBlur(f, (0, 0), 3),
    "blur (heavy)": lambda f: cv2.GaussianBlur(f, (0, 0), 8),
    "sensor noise (sigma 25)": lambda f: _noise(f, 25),
    "low resolution (160 px)": lambda f: _lowres(f, 160),
    "legs cropped out (bottom 30%)": _crop_bottom,
    "mirrored left-right": lambda f: cv2.flip(f, 1),
}


def _even(n):
    return max(2, n - n % 2)


def _write_variant(src, dst, fn, max_seconds=90):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    n = 0
    while n < fps * max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        out = fn(frame)
        out = out[: _even(out.shape[0]), : _even(out.shape[1])]
        if writer is None:
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(out)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    return n


def _row(name, res, base):
    if res.status == "rejected":
        why = res.reasons[0] if res.reasons else "refused"
        return {"name": name, "outcome": "refused to score", "peak": None, "delta": None, "note": why[:70]}
    delta = res.mean_peak - base
    outcome = "agrees" if abs(delta) <= AGREE_WITHIN else "DIFFERENT NUMBER"
    note = "; ".join(res.warnings)[:70] if res.warnings else ""
    return {"name": name, "outcome": outcome, "peak": res.mean_peak, "delta": delta, "note": note}


def run_stress(video, exercise, side="auto", out_dir=".", analyze=safe_analyze):
    base_res = analyze(str(video), exercise, side, CFG, False, 1, None)
    if base_res.status == "rejected":
        raise SystemExit("The original clip is refused by the app, so it cannot be a baseline:\n  - "
                         + "\n  - ".join(base_res.reasons))
    side = base_res.meta.get("side", side)  # use the same limb for every variant
    rows = [{"name": "original", "outcome": "baseline", "peak": base_res.mean_peak, "delta": 0.0, "note": ""}]
    with tempfile.TemporaryDirectory(prefix="spymotion_stress_") as tmp:
        for name, fn in PERTURBATIONS.items():
            dst = Path(tmp) / "variant.mp4"
            if _write_variant(video, dst, fn) == 0:
                continue
            rows.append(_row(name, analyze(str(dst), exercise, side, CFG, False, 1, None), base_res.mean_peak))
            dst.unlink(missing_ok=True)
    variants = rows[1:]
    agree = sum(r["outcome"] == "agrees" for r in variants)
    refused = sum(r["outcome"] == "refused to score" for r in variants)
    wrong = sum(r["outcome"] == "DIFFERENT NUMBER" for r in variants)
    lines = [
        "### Robustness stress test", "",
        f"One clip ({exercise}, {side} side, baseline {base_res.mean_peak:.0f}\u00b0) re-analysed under "
        f"{len(variants)} degradations. 'Agrees' = within {AGREE_WITHIN:.0f}\u00b0 of the original.", "",
        f"- Agrees: **{agree}**  \u00b7  Refused to score: **{refused}**  \u00b7  Gave a different number: **{wrong}**", "",
        "| Condition | Outcome | Average peak | Change | Note |", "|---|---|---|---|---|",
    ]
    for r in rows:
        peak = "\u2013" if r["peak"] is None else f"{r['peak']:.0f}\u00b0"
        delta = "\u2013" if r["delta"] is None else f"{r['delta']:+.1f}\u00b0"
        lines.append(f"| {r['name']} | {r['outcome']} | {peak} | {delta} | {r['note']} |")
    lines += ["", "Refusing to score is the safe outcome. 'Different number' is the failure to investigate.",
              "This tests image-quality robustness on one clip; it does not replace testing on different bodies and postures."]
    report = Path(out_dir) / "stress_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return rows, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--exercise", required=True, choices=["knee", "hip", "shoulder", "elbow"])
    ap.add_argument("--side", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    rows, report = run_stress(a.video, a.exercise, a.side, a.out)
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
