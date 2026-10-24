"""
core.py - SpyMotion analysis engine.

Pure NumPy/SciPy: no video, no UI, no network. Everything here is deterministic
and unit-tested, so every number the patient sees can be traced back to a rule.

Pipeline:  joint angle per frame -> confidence gating -> gap handling ->
           spike removal -> smoothing -> per-repetition peak detection ->
           quality gate (refuse to score when unreliable) -> comparison with
           statistical uncertainty -> auditable rule-based tips.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, medfilt, savgol_filter

# --------------------------------------------------------------------------
# Exercises
# --------------------------------------------------------------------------
# MediaPipe Pose landmark indices as (left, right)
LM = {
    "shoulder": (11, 12),
    "elbow": (13, 14),
    "wrist": (15, 16),
    "hip": (23, 24),
    "knee": (25, 26),
    "ankle": (27, 28),
}


@dataclass(frozen=True)
class Exercise:
    key: str
    label: str
    noun: str  # used in sentences, e.g. "knee bend"
    joints: tuple  # (a, vertex, c) joint names
    flexion_from_straight: bool  # True: movement = 180 - joint angle
    typical_range: float  # approximate textbook active ROM (degrees), context only
    setup: str


EXERCISES: dict[str, Exercise] = {
    "knee": Exercise(
        "knee", "Knee bend", "knee bend", ("hip", "knee", "ankle"), True, 135.0,
        "Film from the side, with the leg you are exercising closest to the camera. "
        "Bend your knee as far as is comfortable, then straighten it. Repeat 3-5 times.",
    ),
    "hip": Exercise(
        "hip", "Leg / knee raise", "hip raise", ("shoulder", "hip", "knee"), True, 120.0,
        "Film from the side (seated or standing). Lift your knee towards your chest as far "
        "as is comfortable, then lower it. Repeat 3-5 times.",
    ),
    "shoulder": Exercise(
        "shoulder", "Arm raise (forwards)", "arm raise", ("hip", "shoulder", "elbow"), False, 180.0,
        "Film from the side, with the arm you are exercising closest to the camera. "
        "Raise your arm forwards and up as far as is comfortable, then lower it. Repeat 3-5 times.",
    ),
    "elbow": Exercise(
        "elbow", "Elbow bend", "elbow bend", ("shoulder", "elbow", "wrist"), True, 145.0,
        "Film from the side. Bend your elbow to bring your hand towards your shoulder, then "
        "straighten it. Repeat 3-5 times.",
    ),
}


def landmark_ids(ex: Exercise, side: str) -> tuple[int, int, int]:
    i = 0 if side == "left" else 1
    return tuple(LM[name][i] for name in ex.joints)  # type: ignore[return-value]


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b (degrees) for arrays of 2D points, shape (N, 2)."""
    ba, bc = a - b, c - b
    denom = np.maximum(np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1), 1e-9)
    cos = np.clip((ba * bc).sum(-1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def movement_value(raw_angle: np.ndarray, ex: Exercise) -> np.ndarray:
    """Convert a raw joint angle to a 'how far has the joint moved' value (0 = neutral)."""
    return 180.0 - raw_angle if ex.flexion_from_straight else raw_angle


# --------------------------------------------------------------------------
# Configuration & result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalConfig:
    conf_thresh: float = 0.5  # per-frame landmark visibility gate
    max_gap_s: float = 0.5  # longest gap we are willing to bridge
    smooth_s: float = 0.3  # Savitzky-Golay window
    min_prominence: float = 8.0  # degrees; smaller wiggles are not counted as reps
    min_rep_gap_s: float = 0.7  # minimum time between rep peaks
    min_coverage: float = 0.60  # below this: refuse to score
    good_coverage: float = 0.80
    min_reps: int = 2  # below this: refuse to score
    good_reps: int = 3
    max_view_ratio: float = 0.50  # shoulder-width / torso-length; high = not side-on
    meaningful_change: float = 5.0  # degrees; floor for calling a change "real"


CFG = SignalConfig()


@dataclass
class SignalResult:
    status: str = "rejected"  # good | fair | rejected
    reasons: list = field(default_factory=list)  # why it was rejected
    warnings: list = field(default_factory=list)
    fps: float = 30.0
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    raw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    smooth: np.ndarray = field(default_factory=lambda: np.zeros(0))
    valid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    reliable: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    reps: list = field(default_factory=list)
    coverage: float = 0.0
    mean_conf: float = 0.0
    confidence: float = 0.0
    view_ratio: float | None = None
    mean_peak: float = float("nan")
    sd_peak: float = float("nan")
    cv: float = float("nan")
    best_peak: float = float("nan")
    mean_excursion: float = float("nan")
    mean_units: float = float("nan")
    rep_period: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def n_reps(self) -> int:
        return len(self.reps)

    @property
    def peaks(self) -> np.ndarray:
        return np.array([r["peak"] for r in self.reps], float)

    def summary(self) -> dict:
        return {
            "status": self.status,
            "mean_peak": round(float(self.mean_peak), 1),
            "sd_peak": round(float(self.sd_peak), 1),
            "n_reps": self.n_reps,
            "coverage": round(self.coverage, 3),
            "confidence": round(self.confidence, 3),
            "mean_excursion": round(float(self.mean_excursion), 1),
            "mean_units": round(float(self.mean_units), 2),
        }


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------
def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def process_signal(
    movement: np.ndarray,
    conf: np.ndarray,
    fps: float,
    cfg: SignalConfig = CFG,
    view_ratio: float | None = None,
) -> SignalResult:
    """Turn a per-frame movement angle + confidence into per-rep metrics and a quality verdict."""
    movement = np.asarray(movement, float)
    conf = np.nan_to_num(np.asarray(conf, float), nan=0.0)
    n = len(movement)
    idx = np.arange(n)
    res = SignalResult(
        fps=float(fps),
        t=idx / fps,
        raw=movement.copy(),
        smooth=np.full(n, np.nan),
        valid=np.isfinite(movement) & (conf >= cfg.conf_thresh),
        reliable=np.zeros(n, bool),
        view_ratio=view_ratio,
    )
    valid = res.valid
    res.coverage = float(valid.mean()) if n else 0.0
    res.mean_conf = float(conf[valid].mean()) if valid.any() else 0.0
    res.confidence = res.coverage * res.mean_conf

    if valid.sum() < max(10, int(fps)):
        res.reasons.append(
            "Your body was not tracked reliably (too few usable frames). "
            "Make sure your whole body is in view, with good light."
        )
        return res

    # 1) bridge short gaps only; mark long gaps as unreliable
    vi = idx[valid]
    filled = np.interp(idx, vi, movement[valid])
    pos = np.searchsorted(vi, idx)
    left = vi[np.clip(pos - 1, 0, len(vi) - 1)]
    right = vi[np.clip(pos, 0, len(vi) - 1)]
    dist = np.minimum(np.abs(idx - left), np.abs(idx - right))
    reliable = (dist * 2) <= int(cfg.max_gap_s * fps)
    res.reliable = reliable

    # 2) spike removal then smoothing
    win = max(5, _odd(int(round(cfg.smooth_s * fps))))
    if win > n:
        win = n if n % 2 else n - 1
    if win < 5:
        res.reasons.append("The video is too short to analyse.")
        return res
    despiked = medfilt(filled, kernel_size=5)
    smooth = savgol_filter(despiked, win, 2, mode="interp")
    res.smooth = smooth

    # 3) repetition peaks
    peaks, _ = find_peaks(
        smooth,
        height=cfg.min_prominence,  # a rep must also reach a real angle, not just wiggle
        prominence=cfg.min_prominence,
        distance=max(1, int(cfg.min_rep_gap_s * fps)),
    )
    reps = []
    dropped = 0
    for k, p in enumerate(peaks):
        if not reliable[p]:
            dropped += 1
            continue
        lo = peaks[k - 1] if k > 0 else 0
        hi = peaks[k + 1] if k + 1 < len(peaks) else n - 1
        lv = lo + int(np.argmin(smooth[lo : p + 1]))
        rv = p + int(np.argmin(smooth[p : hi + 1]))
        peak = float(smooth[p])
        excursion = float(peak - 0.5 * (smooth[lv] + smooth[rv]))
        units = float("nan")
        seg = smooth[lv : p + 1]
        if len(seg) >= 5:  # movement units on the lifting phase: 1 = single smooth movement
            speed = np.abs(np.gradient(seg, 1.0 / fps))
            sp, _ = find_peaks(speed, prominence=0.15 * speed.max(), distance=max(1, int(0.15 * fps)))
            units = float(max(1, len(sp)))
        reps.append(
            {"rep": len(reps) + 1, "frame": int(p), "t": float(p / fps), "peak": peak,
             "excursion": excursion, "units": units}
        )
    res.reps = reps

    # 4) quality gate
    if res.coverage < cfg.min_coverage:
        res.reasons.append(
            f"The joints were clearly visible in only {res.coverage:.0%} of frames "
            f"(we need at least {cfg.min_coverage:.0%})."
        )
    if len(reps) < cfg.min_reps:
        res.reasons.append(
            f"We found {len(reps)} clear repetition(s); at least {cfg.min_reps} are needed to "
            "measure how consistent your movement is."
        )
    if dropped:
        res.warnings.append(f"{dropped} repetition(s) were ignored because tracking was lost at the top of the movement.")
    lost_s = float((~reliable).sum() / fps)
    if lost_s >= 1.0:  # a lost stretch can hide whole repetitions, so never stay silent about it
        res.warnings.append(
            f"About {lost_s:.0f} s of the video could not be tracked and was ignored; "
            "any repetition in that stretch is not counted."
        )
    if res.coverage < cfg.good_coverage and res.coverage >= cfg.min_coverage:
        res.warnings.append(f"Tracking covered {res.coverage:.0%} of frames; results are less certain.")
    if cfg.min_reps <= len(reps) < cfg.good_reps:
        res.warnings.append(f"Only {len(reps)} repetitions were found; 3 or more give a more dependable result.")
    if view_ratio is not None and view_ratio > cfg.max_view_ratio:
        res.warnings.append(
            "The camera does not look side-on to your body, so angles may be underestimated. "
            "Re-film from the side."
        )

    if res.reasons:
        res.status = "rejected"
        return res

    # 5) summary metrics
    peaks_arr = res.peaks
    res.mean_peak = float(peaks_arr.mean())
    res.sd_peak = float(peaks_arr.std(ddof=1)) if len(peaks_arr) > 1 else 0.0
    res.cv = float(res.sd_peak / res.mean_peak) if res.mean_peak > 1e-6 else float("nan")
    res.best_peak = float(peaks_arr.max())
    res.mean_excursion = float(np.mean([r["excursion"] for r in reps]))
    units = [r["units"] for r in reps if np.isfinite(r["units"])]
    res.mean_units = float(np.mean(units)) if units else float("nan")
    res.rep_period = float(np.mean(np.diff([r["t"] for r in reps]))) if len(reps) > 1 else float("nan")
    res.status = "good" if not res.warnings else "fair"
    return res


# --------------------------------------------------------------------------
# Comparison with uncertainty
# --------------------------------------------------------------------------
@dataclass
class Comparison:
    mean_a: float
    mean_b: float
    delta: float
    ci: float  # half-width of 95% interval from rep-to-rep variation (Welch t)
    threshold: float  # max(ci, floor)
    direction: str  # improved | reduced | similar


def compare(a: SignalResult, b: SignalResult, cfg: SignalConfig = CFG) -> Comparison | None:
    """Compare earlier (a) vs latest (b). Returns None if either was rejected."""
    if a.status == "rejected" or b.status == "rejected":
        return None
    pa, pb = a.peaks, b.peaks
    na, nb = len(pa), len(pb)
    va, vb = pa.var(ddof=1), pb.var(ddof=1)
    sa, sb = va / na, vb / nb
    se = float(np.sqrt(sa + sb))
    if se < 1e-9:
        ci = 0.0
    else:
        df = (sa + sb) ** 2 / ((sa**2) / (na - 1) + (sb**2) / (nb - 1))
        ci = float(stats.t.ppf(0.975, df) * se)
    delta = float(pb.mean() - pa.mean())
    real = abs(delta) >= cfg.meaningful_change and abs(delta) > ci
    direction = ("improved" if delta > 0 else "reduced") if real else "similar"
    return Comparison(float(pa.mean()), float(pb.mean()), delta, ci, max(ci, cfg.meaningful_change), direction)


def percent_of_typical(mean_peak: float, ex: Exercise) -> float:
    return 100.0 * mean_peak / ex.typical_range


# --------------------------------------------------------------------------
# Rule-based tips (deterministic, auditable, no LLM)
# --------------------------------------------------------------------------
SAFETY_NOTE = (
    "SpyMotion is a self-monitoring aid, not a medical device or a diagnosis. "
    "Stop if you feel pain, dizziness or anything unusual. If you have a spinal cord injury and "
    "get a sudden pounding headache, flushing or sweating, these can be signs of autonomic "
    "dysreflexia: stop and seek urgent medical help."
)


def build_tips(
    cur: SignalResult,
    ex: Exercise,
    prev: SignalResult | None = None,
    cmp: Comparison | None = None,
) -> list[tuple[str, str]]:
    """Return [(rule_id, text)]. Every tip names the rule that produced it."""
    tips: list[tuple[str, str]] = []
    pct = percent_of_typical(cur.mean_peak, ex)

    if cmp is not None:
        if cmp.direction == "improved":
            tips.append(("R1", f"Your average peak {ex.noun} is {abs(cmp.delta):.0f}\u00b0 higher than in your earlier "
                               f"video (95% range \u00b1{cmp.ci:.0f}\u00b0). That is bigger than the measurement noise, so it "
                               "looks like a real improvement. Keep your routine steady and talk to your physio about new targets."))
        elif cmp.direction == "reduced":
            tips.append(("R3", f"Your average peak {ex.noun} is {abs(cmp.delta):.0f}\u00b0 lower than in your earlier video. "
                               "Tiredness, a different camera angle, clothing or lighting can all cause this, so first re-film "
                               "with the same set-up. If it stays lower, or you notice new symptoms, tell your physio or doctor."))
        else:
            tips.append(("R2", f"No clear change: the difference of {cmp.delta:+.0f}\u00b0 is within the measurement noise "
                               f"(\u00b1{cmp.threshold:.0f}\u00b0). Day-to-day variation is normal. Repeat under the same conditions "
                               "to see the trend."))
    else:
        tips.append(("R4", f"This is your baseline: an average peak of {cur.mean_peak:.0f}\u00b0 "
                           f"(about {pct:.0f}% of a typical full range). Record again in 2-4 weeks from the same position "
                           "to see how you are progressing."))

    if cur.n_reps >= 3 and np.isfinite(cur.cv):
        if cur.cv > 0.15:
            tips.append(("R5a", f"Your repetitions vary by about {cur.cv:.0%} from one to the next. Try to aim for the same "
                                "height each time and move at a steady pace."))
        elif cur.cv <= 0.08:
            tips.append(("R5b", f"Very consistent repetitions (only {cur.cv:.0%} variation). Nice control."))

    if np.isfinite(cur.mean_units) and cur.mean_units >= 1.6:
        tips.append(("R6", "Your lifting movement looks stepwise rather than smooth. Slow, controlled movement often helps. "
                           "If this is new for you, mention it to your physio."))

    peaks = cur.peaks
    if len(peaks) >= 4:
        half = len(peaks) // 2
        if peaks[:half].mean() - peaks[-half:].mean() >= 8:
            tips.append(("R7", "Your range dropped during the session, which can be a sign of fatigue. Consider resting "
                               "between sets."))

    if cur.status == "fair":
        tips.append(("R8", "Tracking quality was only fair. Re-filming from the side with the whole body in view and "
                           "good light will give a more dependable result."))
    return tips
