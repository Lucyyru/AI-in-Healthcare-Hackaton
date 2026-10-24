"""Run with:  python test_core.py   (or pytest). No video, camera or model needed."""
import numpy as np

import core

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} tests passed")
