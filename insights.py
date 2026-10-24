"""Longitudinal movement insights for SpyMotion.

Pure functions only: no Streamlit and no file access.  Keeping the calculations here makes the
progress claims easy to test and audit.  Insights describe measured movement patterns; they do
not diagnose causes or prescribe treatment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProgressSummary:
    exercise: str
    side: str
    total_sessions: int
    included_sessions: int
    excluded_sessions: int
    start_mean: float
    recent_mean: float
    delta: float
    slope_per_session: float
    first_consistency_sd: float | None
    recent_consistency_sd: float | None
    first_units: float | None
    recent_units: float | None
    confidence_mean: float | None
    direction: str


def history_frame(history: Iterable[dict]) -> pd.DataFrame:
    """Normalise saved rows and keep only rows usable for movement-history displays."""
    df = pd.DataFrame(list(history))
    if df.empty:
        return df
    for col in ("exercise", "side", "status", "label", "saved_at"):
        if col not in df.columns:
            df[col] = None
    for col in ("mean_peak", "sd_peak", "mean_units", "confidence", "coverage"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["when"] = pd.to_datetime(df["saved_at"], errors="coerce")
    return df.dropna(subset=["when", "mean_peak"]).sort_values("when").reset_index(drop=True)


def sessions_for(df: pd.DataFrame, exercise: str, side: str | None = None, limit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (included, excluded) recent sessions for a comparable exercise/side.

    Only sessions marked ``good`` drive a progress claim.  ``fair`` sessions remain visible to
    the user but are excluded so a low-quality recording cannot create a false recovery trend.
    Older rows without a status are treated as included for backwards compatibility.
    """
    if df.empty:
        return df.copy(), df.copy()
    q = df[df["exercise"] == exercise].copy()
    if side and side != "auto":
        q = q[q["side"] == side]
    q = q.tail(max(1, int(limit)))
    status = q["status"].fillna("legacy")
    included = q[status.isin(["good", "legacy"])].copy()
    excluded = q[~status.isin(["good", "legacy"])].copy()
    return included, excluded


def _finite_mean(values) -> float | None:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(x.mean()) if len(x) else None


def progress_summary(included: pd.DataFrame, excluded: pd.DataFrame, meaningful_change: float = 5.0) -> ProgressSummary | None:
    """Compare early vs recent windows and estimate a simple session-by-session trend."""
    if len(included) < 4:
        return None
    vals = included["mean_peak"].astype(float).to_numpy()
    split = max(2, len(vals) // 2)
    early = included.iloc[:split]
    recent = included.iloc[split:]
    if recent.empty:
        return None
    start_mean = float(early["mean_peak"].mean())
    recent_mean = float(recent["mean_peak"].mean())
    delta = recent_mean - start_mean
    slope = float(np.polyfit(np.arange(len(vals), dtype=float), vals, 1)[0]) if len(vals) >= 2 else 0.0
    direction = "improving" if delta >= meaningful_change else "reduced" if delta <= -meaningful_change else "stable"
    return ProgressSummary(
        exercise=str(included.iloc[-1]["exercise"]),
        side=str(included.iloc[-1]["side"]),
        total_sessions=len(included) + len(excluded),
        included_sessions=len(included),
        excluded_sessions=len(excluded),
        start_mean=start_mean,
        recent_mean=recent_mean,
        delta=delta,
        slope_per_session=slope,
        first_consistency_sd=_finite_mean(early["sd_peak"]),
        recent_consistency_sd=_finite_mean(recent["sd_peak"]),
        first_units=_finite_mean(early["mean_units"]),
        recent_units=_finite_mean(recent["mean_units"]),
        confidence_mean=_finite_mean(included["confidence"]),
        direction=direction,
    )


def insight_messages(summary: ProgressSummary, meaningful_change: float = 5.0) -> list[tuple[str, str, str]]:
    """Return (kind, title, text) observations; never infer a diagnosis or treatment."""
    out: list[tuple[str, str, str]] = []
    if summary.direction == "improving":
        out.append(("positive", "Range is trending upward",
                    f"Recent sessions average {summary.delta:+.1f}° compared with your earlier sessions in this window."))
    elif summary.direction == "reduced":
        out.append(("caution", "Range is lower in recent sessions",
                    f"Recent sessions average {abs(summary.delta):.1f}° lower. This is an observation, not a diagnosis; consider sharing the trend with your clinician if it persists."))
    else:
        out.append(("neutral", "Range is broadly stable",
                    f"The difference between earlier and recent sessions is {summary.delta:+.1f}°, below the {meaningful_change:.0f}° threshold used for a clear change."))

    if summary.first_consistency_sd is not None and summary.recent_consistency_sd is not None:
        d = summary.recent_consistency_sd - summary.first_consistency_sd
        if d <= -1.5:
            out.append(("positive", "Repetitions are becoming more consistent",
                        f"Rep-to-rep spread fell from about {summary.first_consistency_sd:.1f}° to {summary.recent_consistency_sd:.1f}°."))
        elif d >= 1.5:
            out.append(("caution", "Repetitions are more variable",
                        f"Rep-to-rep spread increased from about {summary.first_consistency_sd:.1f}° to {summary.recent_consistency_sd:.1f}°. Try to keep the same camera setup and compare this pattern over more sessions."))

    if summary.first_units is not None and summary.recent_units is not None:
        d = summary.recent_units - summary.first_units
        if d <= -0.35:
            out.append(("positive", "Movement looks smoother",
                        "The lifting phase contains fewer separate movement bursts than in your earlier sessions."))
        elif d >= 0.35:
            out.append(("caution", "Movement looks more stepwise",
                        "The lifting phase contains more separate movement bursts than in your earlier sessions. This describes the video signal only and does not identify the cause."))

    if summary.excluded_sessions:
        out.append(("neutral", "Quality checks protected this trend",
                    f"{summary.excluded_sessions} of the last {summary.total_sessions} comparable session(s) were excluded from the trend because tracking quality was only fair."))
    return out
