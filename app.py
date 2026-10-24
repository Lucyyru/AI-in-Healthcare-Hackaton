"""SpineMotion - patient-owned, on-device movement progress tracking.  Run: streamlit run app.py"""
from __future__ import annotations

import hashlib
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core
import store
from core import CFG, EXERCISES, SAFETY_NOTE, SignalConfig, build_tips, compare, percent_of_typical
from pose_pipeline import analyze_video

st.set_page_config(page_title="SpineMotion", page_icon="\U0001F9B4", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 18px; }
.stButton > button { min-height: 3.2rem; font-size: 1.1rem; font-weight: 600; }
:focus-visible { outline: 3px solid #0B6E3B !important; outline-offset: 2px; }
.headline { padding: 1rem 1.25rem; border-radius: 8px; background: #F2F6F3; font-size: 1.3rem; line-height: 1.5; }
.headline b { font-size: 2.4rem; }
.badge { display:inline-block; padding:.25rem .7rem; margin:0 .4rem .4rem 0; border:2px solid #0B6E3B;
         border-radius: 999px; font-weight:600; font-size:.95rem; color:#0B6E3B; background:#fff; }
.rule { font-family: monospace; font-size:.8rem; padding:1px 6px; border-radius:4px; background:#E3E8E5; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("SpineMotion")
    st.caption("Your movement. Your data. Your device.")
    blur_face = st.toggle("Blur my face in the result video", value=True)
    save_history = st.toggle("Save my numbers on this device (never video)", value=False)
    with st.expander("Advanced settings"):
        conf_thresh = st.slider("Minimum tracking confidence per frame", 0.3, 0.9, CFG.conf_thresh, 0.05,
                                help="Frames below this are ignored, not guessed.")
        min_prom = st.slider("Smallest movement counted as a repetition (\u00b0)", 4, 20, int(CFG.min_prominence))
        complexity = st.selectbox("Pose model size", [0, 1, 2], index=1,
                                  help="0 = fastest, 2 = most accurate (slower).")
    st.divider()
    if st.button("\U0001F5D1  Delete everything now", use_container_width=True):
        removed = store.delete_everything()
        st.session_state.clear()
        st.success(f"Deleted {removed} file(s) and cleared this session.")
    st.caption("Wipes saved numbers, result videos and everything held in memory.")

cfg = SignalConfig(conf_thresh=float(conf_thresh), min_prominence=float(min_prom))

st.title("SpineMotion")
st.markdown(
    '<span class="badge">100% on-device</span><span class="badge">No uploads</span>'
    '<span class="badge">No accounts</span><span class="badge">Original video never saved</span>'
    '<span class="badge">Refuses to score when unsure</span>',
    unsafe_allow_html=True,
)
tab_main, tab_hist, tab_data = st.tabs(["Check my progress", "My history", "Data & safety"])


# ------------------------------------------------------------------ helpers
def run_one(upl, ex_key: str, side: str, label: str) -> dict:
    data = upl.getvalue()
    key = hashlib.sha256(data + repr((ex_key, side, cfg, blur_face, complexity)).encode()).hexdigest()[:20]
    cache = st.session_state.setdefault("cache", {})
    if key in cache:
        return cache[key]
    tmp = store.tmp_dir()
    src = tmp / f"in_{key[:8]}{Path(upl.name).suffix or '.mp4'}"
    out = tmp / f"result_{key[:8]}.mp4"
    src.write_bytes(data)
    bar = st.progress(0.0, text=f"{label}: starting")
    try:
        res = analyze_video(
            str(src), ex_key, side, cfg, blur_face, complexity, str(out),
            progress=lambda f, t: bar.progress(min(f, 1.0), text=f"{label}: {t}"),
        )
    finally:
        src.unlink(missing_ok=True)  # original footage is never kept
        bar.empty()
    video = out.read_bytes() if out.exists() else None
    entry = {"res": res, "video": video, "label": label, "key": key}
    cache[key] = entry
    if save_history and res.status != "rejected":
        saved = st.session_state.setdefault("saved", set())
        if key not in saved:
            store.append_entry({"exercise": ex_key, "side": res.meta["side"], "label": label, **res.summary()})
            saved.add(key)
    return entry


def banner(res):
    if res.status == "rejected":
        st.error(
            "**We can't score this video reliably.**\n\n"
            + "\n".join(f"- {r}" for r in res.reasons)
            + "\n\nRather than guess, SpineMotion shows no score. Check the filming checklist and try again."
        )
    elif res.status == "fair":
        st.warning("**Usable, with caution.**\n\n" + "\n".join(f"- {w}" for w in res.warnings))
    else:
        st.success("**Good tracking quality.**")


def signal_chart(res, ymax, color="#0B6E3B"):
    ang = np.where(res.reliable, res.smooth, np.nan)  # gaps stay visible as gaps
    df = pd.DataFrame({"t": res.t, "angle": ang})
    base = alt.Chart(df).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X("t:Q", title="Time (seconds)"),
        y=alt.Y("angle:Q", title="Joint movement (\u00b0)", scale=alt.Scale(domain=[0, ymax])),
    )
    layers = [base]
    if res.reps:
        pk = pd.DataFrame(res.reps)
        layers.append(alt.Chart(pk).mark_point(size=140, filled=True, color="#C2410C", opacity=1).encode(
            x="t:Q", y="peak:Q", tooltip=["rep", alt.Tooltip("peak:Q", format=".0f")]))
    if res.status != "rejected":
        layers.append(alt.Chart(pd.DataFrame({"m": [res.mean_peak]})).mark_rule(
            strokeDash=[6, 4], color="#333").encode(y="m:Q"))
    return alt.layer(*layers).properties(height=250)


def metrics_row(res, ex):
    c = st.columns(4)
    c[0].metric("Average peak", f"{res.mean_peak:.0f}\u00b0", help="Mean of the highest angle reached in each repetition.")
    c[1].metric("Repetitions", res.n_reps)
    c[2].metric("Consistency", f"\u00b1{res.sd_peak:.0f}\u00b0", help="Rep-to-rep spread (standard deviation). Smaller = steadier.")
    c[3].metric("Tracking confidence", f"{res.confidence:.0%}", help="Share of frames tracked \u00d7 average landmark confidence.")


def per_rep_table(res):
    df = pd.DataFrame(res.reps)[["rep", "t", "peak", "excursion", "units"]]
    df.columns = ["Rep", "Time (s)", "Peak (\u00b0)", "Excursion (\u00b0)", "Movement units"]
    st.dataframe(df.round(1), hide_index=True, use_container_width=True)


def result_panel(entry, ex, ymax, title):
    res = entry["res"]
    st.subheader(title)
    banner(res)
    if entry["video"]:
        st.video(entry["video"])
        st.caption("Result video: skeleton overlay, live angle, rep counter. Face blurred and audio removed."
                   if blur_face else "Result video: skeleton overlay, live angle, rep counter. Audio removed.")
    if res.status != "rejected":
        metrics_row(res, ex)
    st.altair_chart(signal_chart(res, ymax), use_container_width=True)
    st.caption("Line = smoothed joint movement. Orange dots = detected repetition peaks. Dashed = average peak. "
               "Line gaps = frames we did not trust.")
    if res.status != "rejected":
        with st.expander("How this was calculated (every number)"):
            st.write(f"Side tracked: **{res.meta['side']}** \u00b7 {res.meta['duration_s']:.1f} s at "
                     f"{res.fps:.0f} fps \u00b7 frames with usable tracking: {res.coverage:.0%}")
            per_rep_table(res)
            st.caption("Excursion = peak minus the start/end position of that rep. Movement units: 1 = one smooth lift; "
                       "higher = stepwise movement.")


def y_limit(*results):
    peaks = [r.peaks.max() for r in results if r.n_reps]
    return float(max(peaks + [30]) + 20)


# ------------------------------------------------------------------ main tab
with tab_main:
    st.header("1 \u00b7 Pick your exercise")
    ex_key = st.radio("Exercise", list(EXERCISES), format_func=lambda k: EXERCISES[k].label,
                      horizontal=True, label_visibility="collapsed")
    ex = EXERCISES[ex_key]
    st.info(ex.setup)
    side_label = st.radio("Which limb are you moving?", ["Auto-detect", "Left", "Right"], horizontal=True,
                          help="For before/after comparisons, choose Left or Right so both videos track the same limb.")
    side = {"Auto-detect": "auto", "Left": "left", "Right": "right"}[side_label]

    with st.expander("\U0001F4D0  Filming checklist: 2D video is sensitive to camera position", expanded=True):
        st.markdown(
            "- Camera **side-on**, at hip height, about 2\u20133 m away, held still (prop it on something).\n"
            "- **Whole body in view**, including the joint you are moving, from start to finish.\n"
            "- Good light, from the front or side; avoid strong backlight (windows behind you).\n"
            "- Close-fitting clothing helps tracking. Use the **same spot and set-up** every time.\n"
            "- 3\u20135 slow repetitions, about 15\u201330 seconds."
        )

    st.header("2 \u00b7 Add your videos")
    c1, c2 = st.columns(2)
    types = ["mp4", "mov", "m4v", "avi", "webm"]
    with c1:
        up_before = st.file_uploader("Earlier video (optional, for comparison)", type=types, key="up_before")
    with c2:
        up_after = st.file_uploader("Latest video", type=types, key="up_after")

    st.header("3 \u00b7 See your progress")
    go = st.button("Analyse on this device", type="primary", disabled=up_after is None, use_container_width=True)
    if go:
        entries = {}
        if up_before is not None:
            entries["before"] = run_one(up_before, ex_key, side, "Earlier video")
        entries["after"] = run_one(up_after, ex_key, side, "Latest video")
        st.session_state["analysis"] = {"ex_key": ex_key, **entries}

    A = st.session_state.get("analysis")
    if A:
        ex = EXERCISES[A["ex_key"]]
        after = A["after"]
        before = A.get("before")
        ra = after["res"]
        rb = before["res"] if before else None

        if before and rb.meta["side"] != ra.meta["side"]:
            st.warning("The two videos tracked different limbs, so they can't be compared fairly. "
                       "Choose Left or Right above and analyse again.")
            before_ok = False
        else:
            before_ok = True
        cmp = compare(rb, ra, cfg) if (before and before_ok) else None

        if cmp:
            colour, icon, word = {"improved": ("#0B6E3B", "\u25B2", "Improved"),
                                  "reduced": ("#9A2A00", "\u25BC", "Lower than before"),
                                  "similar": ("#333333", "\u25CF", "No clear change")}[cmp.direction]
            st.markdown(
                f'<div class="headline" style="border-left:12px solid {colour}">{icon} {word}<br>'
                f'<b>{cmp.delta:+.0f}\u00b0</b> in average peak {ex.noun} '
                f'({cmp.mean_a:.0f}\u00b0 \u2192 {cmp.mean_b:.0f}\u00b0) \u00b7 95% range \u00b1{cmp.ci:.0f}\u00b0 '
                f'\u00b7 changes under {cfg.meaningful_change:.0f}\u00b0 are treated as noise</div>',
                unsafe_allow_html=True,
            )
            st.write("")

        ymax = y_limit(*[r for r in (ra, rb) if r is not None])
        if before and before_ok:
            left, right = st.columns(2)
            with left:
                result_panel(before, ex, ymax, "Earlier")
            with right:
                result_panel(after, ex, ymax, "Latest")
        else:
            result_panel(after, ex, ymax, "Your result")
            if before:
                result_panel(before, ex, ymax, "Earlier video")

        if ra.status != "rejected":
            st.subheader("Where this puts you")
            pct = percent_of_typical(ra.mean_peak, ex)
            st.progress(min(pct / 100, 1.0), text=f"Latest: {ra.mean_peak:.0f}\u00b0 \u2248 {pct:.0f}% of a typical full range ({ex.typical_range:.0f}\u00b0)")
            if cmp:
                pb_ = percent_of_typical(rb.mean_peak, ex)
                st.progress(min(pb_ / 100, 1.0), text=f"Earlier: {rb.mean_peak:.0f}\u00b0 \u2248 {pb_:.0f}%")
            st.caption("Typical full range is an approximate textbook value for adults without injury. Your own goal "
                       "may differ \u2014 ask your physio.")

            if cmp:
                st.subheader("Repetition by repetition")
                rows = [{"Rep": r["rep"], "Peak": r["peak"], "Video": "Earlier"} for r in rb.reps] + \
                       [{"Rep": r["rep"], "Peak": r["peak"], "Video": "Latest"} for r in ra.reps]
                st.altair_chart(
                    alt.Chart(pd.DataFrame(rows)).mark_line(point=alt.OverlayMarkDef(size=90)).encode(
                        x=alt.X("Rep:O", title="Repetition"), y=alt.Y("Peak:Q", title="Peak angle (\u00b0)",
                                                                       scale=alt.Scale(domain=[0, ymax])),
                        color=alt.Color("Video:N", scale=alt.Scale(range=["#777777", "#0B6E3B"])),
                        strokeDash="Video:N").properties(height=240),
                    use_container_width=True)

            st.subheader("What this means")
            for rid, text in build_tips(ra, ex, rb if cmp else None, cmp):
                st.markdown(f"{text}  <span class='rule'>rule {rid}</span>", unsafe_allow_html=True)
            st.caption("Tips come from fixed, readable rules (shown beside each tip), not a generative model. "
                       "They would be reviewed by a clinician before real use.")
        st.warning(SAFETY_NOTE)

# ------------------------------------------------------------------ history tab
with tab_hist:
    st.header("My history")
    hist = store.load_history()
    if not hist:
        st.info("Nothing saved yet. Switch on **Save my numbers on this device** in the sidebar. "
                "Only numbers are saved \u2014 never video.")
    else:
        df = pd.DataFrame(hist)
        df["when"] = pd.to_datetime(df["saved_at"])
        st.altair_chart(
            alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=90)).encode(
                x=alt.X("when:T", title="Date"), y=alt.Y("mean_peak:Q", title="Average peak (\u00b0)"),
                color="exercise:N", tooltip=["exercise", "side", "mean_peak", "n_reps", "label"]).properties(height=300),
            use_container_width=True)
        st.dataframe(df.drop(columns=["when"]), hide_index=True, use_container_width=True)
        st.download_button("Download my data (CSV)", df.drop(columns=["when"]).to_csv(index=False),
                           "spinemotion_history.csv", "text/csv")
        st.caption("This file lives only in the folder shown in the Data & safety tab. It is yours.")

# ------------------------------------------------------------------ data & safety tab
with tab_data:
    st.header("Data & safety")
    st.info("**Try it:** switch off Wi-Fi and run an analysis. It works identically, because nothing leaves this device.")
    a, b = st.columns(2)
    with a:
        st.subheader("Privacy")
        st.markdown(
            "- All analysis runs on this device. No uploads, accounts, cloud services or API calls.\n"
            "- Your **original video is never saved**: it is analysed and discarded.\n"
            "- The result video is kept only temporarily, with the **face blurred** and the **audio removed**.\n"
            "- Numbers are saved only if you switch that on; **Delete everything** wipes it all in one click."
        )
        st.subheader("Security")
        st.markdown(
            "- No server and no network service means no remote attack surface and no data in transit.\n"
            "- Files live in one folder on your device. The list below shows exactly what is held right now.\n"
            "- Honest limit: saved history is not encrypted; protect your device with a passcode."
        )
        st.subheader("Data provenance")
        st.markdown("The pose model is Google MediaPipe Pose (pre-trained, bundled, no download). "
                    "All test footage used to build and check this prototype was recorded with consent from the team.")
    with b:
        st.subheader("Fairness")
        st.markdown(
            "- Pose models are mostly trained on able-bodied people, so they can be **less accurate for seated, "
            "wheelchair, atypical or assisted postures**, other body shapes, dark clothing and poor light.\n"
            "- So SpineMotion **checks its own reliability** (landmark confidence, rep count, camera angle) and "
            "**refuses to score** instead of guessing.\n"
            "- We tested across conditions and publish where it fails (below)."
        )
        st.subheader("Transparency")
        st.markdown(
            "- Every number can be expanded: per-rep peaks, tracking coverage, confidence.\n"
            "- Tips use fixed rules, each labelled with its rule ID. No generative AI, no hallucinations.\n"
            "- Changes are called 'real' only if they exceed rep-to-rep noise **and** 5\u00b0."
        )
        st.subheader("Known limitations")
        st.markdown(
            "- Single 2D camera: angles are projections and depend on camera position. A side-on view is required.\n"
            "- Not a medical device; not validated against clinical goniometry in a trial.\n"
            "- Face blur is best-effort: if the face is not detected in a frame it may not be blurred.\n"
            "- Typical-range values are textbook averages for uninjured adults."
        )
    st.subheader("What this app holds right now")
    files = store.audit()
    if files:
        st.dataframe(pd.DataFrame(files, columns=["File", "Bytes"]), hide_index=True, use_container_width=True)
    else:
        st.success("No files stored.")
    st.caption(f"Data folder: {store.DATA_DIR} \u00b7 Temp folder: {store.TMP}")
    report = Path(__file__).with_name("validation_report.md")
    if report.exists():
        st.subheader("Validation & fairness testing results")
        st.markdown(report.read_text())
    else:
        st.caption("Run `python validate.py` on your labelled clips to show validation results here.")
    st.warning(SAFETY_NOTE)
