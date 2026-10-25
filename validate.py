"""
validate.py - accuracy + fairness evidence for the pitch.

Usage:
    python validate.py --csv data/labels.csv --videos clips/

labels.csv columns:
    file             clip filename inside --videos
    exercise         knee | hip | shoulder | elbow
    manual_peak_deg  your reference measurement of the peak joint angle (see README)
    condition        free-text tag, e.g. standing, seated, wheelchair, dim_light, dark_clothes
    side             (optional) left | right | auto

Outputs validation_report.csv and validation_report.md (the app shows the .md in 'Data & safety').
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spymotion.pose_pipeline import analyze_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = {"file", "exercise", "manual_peak_deg"} - set(df.columns)
    if missing:
        raise SystemExit(f"labels.csv is missing required column(s): {sorted(missing)}")
    if "condition" not in df.columns:
        df["condition"] = "unspecified"
    df["condition"] = df["condition"].fillna("unspecified")
    rows = []
    for r in df.to_dict("records"):
        side = r.get("side") if isinstance(r.get("side"), str) else "auto"
        path = Path(args.videos) / r["file"]
        print(f"analysing {path.name} ...")
        try:
            res = analyze_video(str(path), r["exercise"], side, blur_face=False)
            ok = res.status != "rejected"
            rows.append({**r, "status": res.status, "app_peak_deg": res.mean_peak if ok else np.nan,
                         "n_reps": res.n_reps, "coverage": res.coverage, "confidence": res.confidence,
                         "reason": "; ".join(res.reasons)})
        except Exception as e:  # keep going: a crash is a failure we want to report
            rows.append({**r, "status": "error", "app_peak_deg": np.nan, "n_reps": 0, "coverage": 0,
                         "confidence": 0, "reason": str(e)})
    out = pd.DataFrame(rows)
    out["error_deg"] = out["app_peak_deg"] - out["manual_peak_deg"]
    out.to_csv(Path(args.out) / "validation_report.csv", index=False)

    acc = out.dropna(subset=["error_deg"])
    lines = ["### Validation against reference measurements", ""]
    if len(acc):
        bias, sd = acc.error_deg.mean(), acc.error_deg.std(ddof=1) if len(acc) > 1 else 0.0
        lines += [
            f"- Clips scored: **{len(acc)}** of {len(out)} ({len(out) - len(acc)} refused/failed)",
            f"- Mean absolute error: **{acc.error_deg.abs().mean():.1f}\u00b0**",
            f"- Bias (app minus reference): **{bias:+.1f}\u00b0**; 95% limits of agreement "
            f"{bias - 1.96 * sd:+.1f}\u00b0 to {bias + 1.96 * sd:+.1f}\u00b0",
        ]
        if len(acc) >= 3:
            lines.append(f"- Correlation with reference: r = {np.corrcoef(acc.app_peak_deg, acc.manual_peak_deg)[0, 1]:.2f}")
    lines += ["", "### Fairness: performance by condition", "",
              "| Condition | Clips | Refused / failed | Mean abs. error (\u00b0) | Mean tracked frames |",
              "|---|---|---|---|---|"]
    for cond, g in out.groupby("condition"):
        refused = int((g.status.isin(["rejected", "error"])).sum())
        mae = g.error_deg.abs().mean()
        lines.append(f"| {cond} | {len(g)} | {refused} ({refused / len(g):.0%}) | "
                     f"{'n/a' if np.isnan(mae) else f'{mae:.1f}'} | {g.coverage.mean():.0%} |")
    lines += ["", "Where the app refuses to score, it is behaving as designed rather than guessing.",
              "Reference = manual measurement; this checks the pose estimate, not a clinical goniometer."]
    (Path(args.out) / "validation_report.md").write_text(
    "\n".join(lines),
    encoding="utf-8",
)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
