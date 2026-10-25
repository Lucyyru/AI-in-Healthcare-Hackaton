"""Engine tests. Run with:  python test_core.py   (or pytest -q). No video, camera or model needed."""
import json

import numpy as np

from spymotion import core
from spymotion import scenarios

FPS = 30.0


def make_signal(peak, n_reps=5, seconds=22, noise=2.0, seed=0, period=3.0, rest=1.5):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FPS)
    s = np.zeros_like(t)
    for k in range(n_reps):
        t0 = rest + k * period
        m = (t >= t0) & (t < t0 + period)
        s[m] = peak * (1 - np.cos(2 * np.pi * (t[m] - t0) / period)) / 2
    return s + rng.normal(0, noise, len(t)), np.full(len(t), 0.95)


def test_rep_count_and_peak():
    s, c = make_signal(80)
    r = core.process_signal(s, c, FPS)
    assert r.status in ("good", "fair"), r.reasons
    assert r.n_reps == 5
    assert abs(r.mean_peak - 80) < 4, r.mean_peak
    assert r.cv < 0.1


def test_short_gaps_bridged_long_gaps_flagged():
    s, c = make_signal(80)
    c[100:110] = 0.1  # 0.33 s gap - bridged
    c[300:420] = 0.1  # 4 s gap - unreliable
    r = core.process_signal(s, c, FPS)
    assert r.reliable[105] and not r.reliable[360]
    assert r.n_reps in (3, 4)  # reps inside the long gap are excluded, not guessed


def test_low_confidence_is_refused():
    s, _ = make_signal(80)
    c = np.full(len(s), 0.2)
    r = core.process_signal(s, c, FPS)
    assert r.status == "rejected" and r.reasons


def test_single_rep_is_refused():
    s, c = make_signal(80, n_reps=1, seconds=6)
    r = core.process_signal(s, c, FPS)
    assert r.status == "rejected"


def test_noise_below_prominence_not_counted():
    s, c = make_signal(5, n_reps=6)  # 5 degree wiggles
    r = core.process_signal(s, c, FPS)
    assert r.status == "rejected"


def test_compare_directions():
    a = core.process_signal(*make_signal(60, seed=1), FPS)
    b = core.process_signal(*make_signal(80, seed=2), FPS)
    same = core.process_signal(*make_signal(61, seed=3), FPS)
    assert core.compare(a, b).direction == "improved"
    assert core.compare(b, a).direction == "reduced"
    assert core.compare(a, same).direction == "similar"


def test_noise_alone_is_not_called_change():
    a = core.process_signal(*make_signal(70, seed=4, noise=4), FPS)
    b = core.process_signal(*make_signal(72, seed=5, noise=4), FPS)
    assert core.compare(a, b).direction == "similar"


def test_tips_name_their_rules():
    ex = core.EXERCISES["knee"]
    a = core.process_signal(*make_signal(60, seed=1), FPS)
    b = core.process_signal(*make_signal(80, seed=2), FPS)
    tips = core.build_tips(b, ex, a, core.compare(a, b))
    assert tips[0][0] == "R1"
    assert core.build_tips(b, ex)[0][0] == "R4"


def test_joint_angle_geometry():
    a, b, c = np.array([[0., 1.]]), np.array([[0., 0.]]), np.array([[1., 0.]])
    assert abs(core.joint_angle(a, b, c)[0] - 90) < 1e-6
    assert abs(core.joint_angle(np.array([[0., 1.]]), b, np.array([[0., -1.]]))[0] - 180) < 1e-6


# ---------------------------------------------------------------- added: reliability tests
def _signal_from_peaks(peaks, period=3.0, rest=1.5, noise=1.0, seed=0, fps=FPS):
    rng = np.random.default_rng(seed)
    t = np.arange(0, rest + period * len(peaks) + 1.5, 1 / fps)
    s = np.zeros_like(t)
    for k, p in enumerate(peaks):
        t0 = rest + k * period
        m = (t >= t0) & (t < t0 + period)
        s[m] = p * (1 - np.cos(2 * np.pi * (t[m] - t0) / period)) / 2
    return s + rng.normal(0, noise, len(t)), np.full(len(t), 0.95)


def _stepwise_signal(peak=80, n_reps=5, period=4.0, rest=1.5, fps=FPS):
    """Each lift happens in two jerky steps with a pause between them."""
    t = np.arange(0, rest + period * n_reps + 1.5, 1 / fps)
    s = np.zeros_like(t)
    for k in range(n_reps):
        t0 = rest + k * period
        knots_t = [t0, t0 + 0.6, t0 + 1.0, t0 + 1.4, t0 + 1.8, t0 + period]
        knots_v = [0, peak / 2, peak / 2, peak, peak, 0]
        m = (t >= t0) & (t <= t0 + period)
        s[m] = np.interp(t[m], knots_t, knots_v)
    return s + np.random.default_rng(1).normal(0, 0.5, len(t)), np.full(len(t), 0.95)


def test_all_scenarios_behave_as_specified():
    for name in scenarios.SCENARIOS:
        assert scenarios.check(name) == [], name


def test_degenerate_inputs_never_raise():
    cases = {
        "empty": (np.array([]), np.array([])),
        "all_nan": (np.full(300, np.nan), np.full(300, 0.9)),
        "all_inf": (np.full(300, np.inf), np.full(300, 0.9)),
        "constant": (np.full(300, 45.0), np.full(300, 0.9)),
        "tiny": (np.arange(12.0), np.full(12, 0.9)),
        "nan_conf": (np.sin(np.arange(300) / 10) * 40 + 40, np.full(300, np.nan)),
    }
    for name, (s, c) in cases.items():
        r = core.process_signal(s, c, FPS)
        assert r.status == "rejected", name
        assert r.reasons, name


def test_spike_is_not_a_repetition_or_peak():
    s, c = make_signal(60, seed=7)
    s = s.copy()
    s[200] = 170  # single-frame tracking glitch
    r = core.process_signal(s, c, FPS)
    assert r.n_reps == 5 and r.mean_peak < 70


def test_frame_rate_extremes():
    for fps in (10, 15, 24, 60, 120):
        s, c = scenarios.movement_signal(80, fps=fps, seed=2)
        r = core.process_signal(s, c, float(fps))
        assert r.status in ("good", "fair") and r.n_reps == 5, (fps, r.status, r.n_reps, r.reasons)
        assert abs(r.mean_peak - 80) < 6, (fps, r.mean_peak)


def test_deterministic():
    a = core.process_signal(*make_signal(75, seed=9), FPS)
    b = core.process_signal(*make_signal(75, seed=9), FPS)
    assert a.summary() == b.summary()


def test_summary_is_strict_json():
    r = core.process_signal(*make_signal(75, seed=9), FPS)
    json.dumps(r.summary(), allow_nan=False)


def test_front_on_camera_is_warned_not_silent():
    r = core.process_signal(*make_signal(80), FPS, view_ratio=0.95)
    assert r.status == "fair" and any("side-on" in w for w in r.warnings)


def test_lost_stretch_is_reported():
    s, c = make_signal(80)
    c[300:420] = 0.1  # 4 s with no person: a repetition vanishes
    r = core.process_signal(s, c, FPS)
    assert r.status == "fair" and any("could not be tracked" in w for w in r.warnings)


def test_compare_refuses_rejected_results():
    ok = core.process_signal(*make_signal(80), FPS)
    bad = core.process_signal(*make_signal(80, n_reps=1, seconds=6), FPS)
    assert core.compare(ok, bad) is None and core.compare(bad, ok) is None


def test_compare_zero_variance():
    a = core.process_signal(*scenarios.movement_signal(60, noise=0.0), FPS)
    b = core.process_signal(*scenarios.movement_signal(80, noise=0.0), FPS)
    c = core.compare(a, b)
    assert c.direction == "improved" and np.isfinite(c.ci)


def test_every_tip_rule_is_reachable():
    ex = core.EXERCISES["knee"]
    seen = set()
    wobbly = core.process_signal(*_signal_from_peaks([60, 95, 55, 100, 60]), FPS)
    steady = core.process_signal(*make_signal(80, noise=0.5), FPS)
    fatigue = core.process_signal(*_signal_from_peaks([95, 90, 70, 60, 50]), FPS)
    stepwise = core.process_signal(*_stepwise_signal(), FPS)
    fair = core.process_signal(*make_signal(80), FPS, view_ratio=0.9)
    up = core.compare(core.process_signal(*make_signal(60, seed=1), FPS), steady)
    down = core.compare(steady, core.process_signal(*make_signal(55, seed=2), FPS))
    same = core.compare(steady, core.process_signal(*make_signal(80, seed=5, noise=0.5), FPS))
    for res, prev, cmp in [(wobbly, None, None), (steady, None, None), (fatigue, None, None), (stepwise, None, None),
                           (fair, None, None), (steady, None, up), (steady, None, down), (steady, None, same)]:
        for rid, text in core.build_tips(res, ex, prev, cmp):
            assert text.strip()
            seen.add(rid)
    assert {"R1", "R2", "R3", "R4", "R5a", "R5b", "R6", "R7", "R8"} <= seen, sorted(seen)


def test_percent_of_typical_and_landmarks():
    knee = core.EXERCISES["knee"]
    assert abs(core.percent_of_typical(67.5, knee) - 50.0) < 1e-9
    assert core.landmark_ids(knee, "left") == (23, 25, 27) and core.landmark_ids(knee, "right") == (24, 26, 28)
    shoulder = core.EXERCISES["shoulder"]
    assert core.movement_value(np.array([30.0]), shoulder)[0] == 30.0
    assert core.movement_value(np.array([30.0]), knee)[0] == 150.0


def test_joint_angle_degenerate_geometry_is_finite():
    p = np.array([[1.0, 1.0]])
    assert np.isfinite(core.joint_angle(p, p, p)).all()  # all three points identical
    assert np.isfinite(core.joint_angle(p, np.array([[0.0, 0.0]]), p)).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} tests passed")
