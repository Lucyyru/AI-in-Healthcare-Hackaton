import pandas as pd

from spymotion.insights import (
    history_frame,
    sessions_for,
    progress_summary,
    insight_messages,
)


def _rows(peaks, statuses=None):
    statuses = statuses or ["good"] * len(peaks)
    return [
        {"saved_at": f"2026-10-{i+1:02d}T10:00:00", "exercise": "shoulder", "side": "right",
         "label": "Latest video", "status": statuses[i], "mean_peak": p, "sd_peak": 5-i*0.2,
         "mean_units": 2-i*0.05, "confidence": .9}
        for i, p in enumerate(peaks)
    ]


def test_progress_uses_window_means_not_first_last_only():
    df = history_frame(_rows([80, 82, 81, 96, 100, 98]))
    included, excluded = sessions_for(df, "shoulder", "right", 10)
    s = progress_summary(included, excluded, 5)
    assert s is not None
    assert s.direction == "improving"
    assert s.delta > 10


def test_fair_session_is_visible_but_excluded_from_claim():
    df = history_frame(_rows([80, 82, 140, 98, 100], ["good", "good", "fair", "good", "good"]))
    included, excluded = sessions_for(df, "shoulder", "right", 10)
    assert len(included) == 4
    assert len(excluded) == 1
    s = progress_summary(included, excluded, 5)
    assert s is not None and s.excluded_sessions == 1
    assert any("Quality checks" in title for _, title, _ in insight_messages(s))


def test_needs_four_good_sessions():
    df = history_frame(_rows([80, 82, 84]))
    included, excluded = sessions_for(df, "shoulder", "right", 10)
    assert progress_summary(included, excluded) is None
