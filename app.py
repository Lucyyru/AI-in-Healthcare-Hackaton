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

# ====================================================================== design system
# System fonts only: loading web fonts would make a network call and break the on-device claim.
CSS = """
<style>
:root{
  --ink:#0F2A20; --muted:#556860; --line:#D9E6DD; --mint:#E8F4EB; --mint2:#F4F9F5;
  --g900:#0B3D2E; --g700:#0B6E3B; --g500:#16A34A; --amber:#B45309; --amberbg:#FEF6E0;
  --red:#9A2A00; --redbg:#FDEDE8; --card:#FFFFFF;
  --shadow:0 1px 2px rgba(15,42,32,.06), 0 8px 24px rgba(15,42,32,.07);
}
html, body, [class*="css"], .stApp { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(1100px 420px at 85% -80px, #D9F0E0 0%, rgba(217,240,224,0) 70%), linear-gradient(180deg,#F3F8F4 0,#FFFFFF 520px); }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
h1,h2,h3 { letter-spacing: -0.02em; color: var(--ink); }
p, li { line-height: 1.6; }
:focus-visible { outline: 3px solid var(--g700) !important; outline-offset: 2px; border-radius: 8px; }

/* hero */
.hero{ background: linear-gradient(135deg,#0B3D2E 0%,#0B6E3B 60%,#16A34A 120%); color:#fff; border-radius:28px; padding:2.2rem 2.4rem 1.8rem; box-shadow: var(--shadow); position:relative; overflow:hidden; margin-bottom:1.4rem;}
.hero:after{ content:""; position:absolute; right:-70px; top:-70px; width:300px; height:300px; border-radius:50%; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.22), rgba(255,255,255,0) 70%);}
.brand{ display:flex; align-items:center; gap:.7rem; font-weight:700; font-size:1.05rem; letter-spacing:.02em; opacity:.95;}
.hero h1{ color:#fff; font-size:2.7rem; line-height:1.1; margin:.9rem 0 .5rem; font-weight:800; max-width:760px;}
.hero p{ color:#E3F4E8; font-size:1.15rem; max-width:680px; margin:0 0 1.2rem;}
.pills{ display:flex; flex-wrap:wrap; gap:.5rem; position:relative; z-index:1;}
.pill{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35); color:#fff; padding:.35rem .85rem; border-radius:999px; font-size:.92rem; font-weight:600; backdrop-filter: blur(4px);}

/* steps */
.step{ display:flex; align-items:center; gap:.9rem; margin:2rem 0 .8rem;}
.step .n{ flex:0 0 auto; width:2.4rem; height:2.4rem; border-radius:50%; background:var(--g700); color:#fff; font-weight:800; display:grid; place-items:center; box-shadow:0 4px 12px rgba(11,110,59,.35);}
.step h2{ margin:0; font-size:1.55rem; font-weight:800; padding:0;}
.step small{ display:block; color:var(--muted); font-size:1rem; font-weight:500; margin-top:.1rem;}

/* callouts */
.callout{ background:var(--mint2); border:1px solid var(--line); border-left:6px solid var(--g500); border-radius:14px; padding:.9rem 1.1rem; margin:.4rem 0 1rem; }
.notice{ border-radius:16px; padding:.95rem 1.2rem; margin:.5rem 0 1rem; border:1px solid; }
.notice b.t{ display:block; font-size:1.05rem; margin-bottom:.2rem;}
.notice ul{ margin:.2rem 0 0; padding-left:1.2rem;}
.notice.ok{ background:#EAF7EE; border-color:#B5E0C1; color:#0B3D2E;}
.notice.warn{ background:var(--amberbg); border-color:#F3D58C; color:#5A3A00;}
.notice.bad{ background:var(--redbg); border-color:#F2B8A5; color:#6B1D00;}
.safety{ background:var(--amberbg); border:1px solid #F3D58C; border-radius:16px; padding:1rem 1.2rem; color:#5A3A00; margin-top:1.4rem; font-size:.97rem;}

/* verdict */
.verdict{ border-radius:24px; padding:1.6rem 1.8rem; display:flex; gap:1.6rem; align-items:center; flex-wrap:wrap; box-shadow:var(--shadow); border:1px solid var(--line); background:#fff; margin:1.2rem 0 .6rem;}
.verdict .big{ font-size:4rem; font-weight:800; line-height:1; letter-spacing:-0.04em;}
.verdict .lab{ font-size:.85rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.35rem;}
.verdict .txt{ flex:1 1 320px; font-size:1.12rem;}
.verdict .txt .sub{ color:var(--muted); font-size:.98rem; margin-top:.3rem;}
.verdict.up{ background:linear-gradient(135deg,#EAF7EE,#FFFFFF 70%); border-color:#B5E0C1;} .verdict.up .big,.verdict.up .lab{ color:var(--g700);}
.verdict.down{ background:linear-gradient(135deg,#FDEDE8,#FFFFFF 70%); border-color:#F2B8A5;} .verdict.down .big,.verdict.down .lab{ color:var(--red);}
.verdict.flat{ background:linear-gradient(135deg,#EEF2EF,#FFFFFF 70%);} .verdict.flat .big,.verdict.flat .lab{ color:#33433B;}

/* panels & stats */
.ptitle{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.25rem; margin-bottom:.4rem;}
.tag{ font-size:.75rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; background:var(--mint); color:var(--g700); padding:.2rem .6rem; border-radius:999px;}
.stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:.7rem; margin:.7rem 0 .4rem;}
.stat{ background:var(--mint2); border:1px solid var(--line); border-radius:16px; padding:.8rem .95rem;}
.stat .k{ font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);}
.stat .v{ font-size:2rem; font-weight:800; letter-spacing:-0.03em; line-height:1.15; margin-top:.15rem; color:var(--g900);}
.stat .h{ font-size:.82rem; color:var(--muted); margin-top:.1rem;}

/* range bars */
.bar{ margin:.55rem 0 1rem;}
.bar .row{ display:flex; justify-content:space-between; font-weight:600; font-size:.98rem; margin-bottom:.3rem;}
.bar .track{ height:16px; background:#E6EEE8; border-radius:999px; overflow:hidden;}
.bar .fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,#16A34A,#0B6E3B);}
.bar .fill.old{ background:linear-gradient(90deg,#A8B8AE,#7C8E83);}

/* tips */
.tip{ display:flex; gap:.9rem; align-items:flex-start; background:#fff; border:1px solid var(--line); border-radius:16px; padding:.9rem 1.1rem; margin:.55rem 0; box-shadow:0 1px 2px rgba(15,42,32,.04);}
.tip p{ margin:0; font-size:1.02rem;}
.rule{ flex:0 0 auto; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.78rem; font-weight:700; padding:.2rem .55rem; border-radius:8px; background:var(--mint); color:var(--g700); margin-top:.1rem;}

/* how-it-works + info cards */
.grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:1rem; margin:.6rem 0;}
.icard{ background:#fff; border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.3rem; box-shadow:var(--shadow);}
.icard .ic{ width:2.6rem; height:2.6rem; border-radius:14px; background:var(--mint); display:grid; place-items:center; font-size:1.3rem; margin-bottom:.6rem;}
.icard h3{ margin:.1rem 0 .4rem; font-size:1.15rem; font-weight:800;}
.icard ul{ margin:0; padding-left:1.1rem; color:#26382F;} .icard li{ margin:.25rem 0;}
.icard p{ margin:0; color:#26382F;}

/* streamlit widgets */
.stButton > button{ min-height:3.3rem; border-radius:14px; font-size:1.08rem; font-weight:700; border:2px solid var(--line); transition:transform .12s ease, box-shadow .12s ease;}
.stButton > button:hover{ transform:translateY(-1px); border-color:var(--g700); color:var(--g700);}
.stButton > button[kind="primary"]{ background:linear-gradient(135deg,#0B6E3B,#16A34A); color:#fff; border:none; box-shadow:0 8px 20px rgba(11,110,59,.28);}
.stButton > button[kind="primary"]:hover{ color:#fff; box-shadow:0 10px 26px rgba(11,110,59,.38);}
.stButton > button:disabled{ opacity:.55; transform:none; box-shadow:none;}
[data-testid="stFileUploaderDropzone"]{ border:2px dashed #8CC3A0; border-radius:20px; background:#fff; padding:1.3rem;}
[data-testid="stFileUploaderDropzone"]:hover{ background:var(--mint2); border-color:var(--g700);}
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:16px; background:#fff;}
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:22px;}
[data-testid="stVideo"] video, video{ border-radius:16px;}
[data-testid="stDataFrame"]{ border-radius:14px; overflow:hidden;}
[data-testid="stSidebar"]{ background:#EEF5F0; border-right:1px solid var(--line);}
[data-testid="stSidebar"] .brand{ color:var(--g900); }

/* pill tabs */
div[data-baseweb="tab-list"]{ gap:.5rem; border-bottom:none; background:var(--mint); padding:.35rem; border-radius:999px; width:fit-content;}
button[data-baseweb="tab"]{ border-radius:999px; padding:.55rem 1.3rem; height:auto; font-weight:700; color:var(--g900);}
button[data-baseweb="tab"][aria-selected="true"]{ background:#fff; color:var(--g700); box-shadow:0 2px 8px rgba(15,42,32,.12);}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"]{ display:none;}

/* chip radios */
div[role="radiogroup"]{ gap:.5rem;}
label[data-baseweb="radio"]{ background:#fff; border:2px solid var(--line); border-radius:999px; padding:.5rem 1.15rem; margin:0; transition:all .12s ease; cursor:pointer;}
label[data-baseweb="radio"] > div:first-child{ display:none;}
label[data-baseweb="radio"]:hover{ border-color:var(--g700);}
label[data-baseweb="radio"]:has(input:checked){ background:var(--g700); border-color:var(--g700); box-shadow:0 4px 12px rgba(11,110,59,.3);}
label[data-baseweb="radio"]:has(input:checked) *{ color:#fff !important; font-weight:700;}
label[data-baseweb="radio"]:has(input:focus-visible){ outline:3px solid var(--g700); outline-offset:2px;}

@media (max-width:760px){ .hero{padding:1.5rem 1.3rem;} .hero h1{font-size:2rem;} .verdict .big{font-size:3rem;} }
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO = (
    '<svg width="34" height="34" viewBox="0 0 34 34" role="img" aria-label="SpineMotion logo">'
    '<rect width="34" height="34" rx="10" fill="rgba(255,255,255,.16)"/>'
    '<rect x="12" y="5" width="10" height="5" rx="2.5" fill="#fff"/>'
    '<rect x="11" y="11.5" width="12" height="5" rx="2.5" fill="#fff" opacity=".9"/>'
    '<rect x="12" y="18" width="10" height="5" rx="2.5" fill="#fff" opacity=".8"/>'
    '<rect x="13" y="24.5" width="8" height="4.5" rx="2.2" fill="#fff" opacity=".7"/></svg>'
)


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def step(n: int, title: str, sub: str = "") -> None:
    s = f"<small>{sub}</small>" if sub else ""
    html(f'<div class="step"><div class="n">{n}</div><div><h2>{title}</h2>{s}</div></div>')


def icard(icon: str, title: str, items: list[str] | None = None, text: str = "") -> str:
    body = "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else f"<p>{text}</p>"
    return f'<div class="icard"><div class="ic">{icon}</div><h3>{title}</h3>{body}</div>'


def themed(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure(background="transparent")
        .configure_axis(labelColor="#3B4A42", titleColor="#3B4A42", labelFontSize=13, titleFontSize=13,
                        titleFontWeight=600, gridColor="#E3ECE6", domainColor="#C5D6CB", tickColor="#C5D6CB")
        .configure_legend(labelFontSize=13, titleFontSize=13, orient="top", title=None, symbolSize=160)
    )


# ====================================================================== sidebar
with st.sidebar:
    html(f'<div class="brand">{LOGO.replace("rgba(255,255,255,.16)", "#0B6E3B")}<span>SpineMotion</span></div>')
    st.caption("Your movement. Your data. Your device.")
    blur_face = st.toggle("Blur my face in the result video", value=True)
    save_history = st.toggle("Save my numbers on this device (never video)", value=False)
    with st.expander("Advanced settings"):
        conf_thresh = st.slider("Minimum tracking confidence per frame", 0.3, 0.9, CFG.conf_thresh, 0.05,
                                help="Frames below this are ignored, not guessed.")
        min_prom = st.slider("Smallest movement counted as a repetition (°)", 4, 20, int(CFG.min_prominence))
        complexity = st.selectbox("Pose model size", [0, 1, 2], index=1,
                                  help="0 = fastest, 2 = most accurate (slower).")
    st.divider()
    if st.button("\U0001F5D1  Delete everything now", use_container_width=True):
        removed = store.delete_everything()
        st.session_state.clear()
        st.success(f"Deleted {removed} file(s) and cleared this session.")
    st.caption("Wipes saved numbers, result videos and everything held in memory.")

cfg = SignalConfig(conf_thresh=float(conf_thresh), min_prominence=float(min_prom))

# ====================================================================== hero
html(
    f'<div class="hero"><div class="brand">{LOGO}<span>SpineMotion</span></div>'
    "<h1>See your recovery, privately.</h1>"
    "<p>Film an exercise. Get your range of motion for every repetition and a clear before-and-after, "
    "calculated entirely on this device.</p>"
    '<div class="pills"><span class="pill">\U0001F512 100% on-device</span><span class="pill">☁️ No uploads</span>'
    '<span class="pill">\U0001F464 No accounts</span><span class="pill">\U0001F3AC Original video never saved</span>'
    '<span class="pill">\U0001F6D1 Refuses to score when unsure</span></div></div>'
)
tab_main, tab_hist, tab_data = st.tabs(["Check my progress", "My history", "Data & safety"])


# ====================================================================== helpers
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


def banner(res) -> None:
    if res.status == "rejected":
        items = "".join(f"<li>{r}</li>" for r in res.reasons)
        html(f'<div class="notice bad" role="alert"><b class="t">We can’t score this video reliably</b><ul>{items}</ul>'
             "<div style='margin-top:.5rem'>Rather than guess, SpineMotion shows no score. "
             "Check the filming checklist and try again.</div></div>")
    elif res.status == "fair":
        items = "".join(f"<li>{w}</li>" for w in res.warnings)
        html(f'<div class="notice warn" role="status"><b class="t">Usable, with caution</b><ul>{items}</ul></div>')
    else:
        html('<div class="notice ok" role="status"><b class="t">✓ Good tracking quality</b>'
             "Your joints were clearly visible throughout.</div>")


def signal_chart(res, ymax, color="#0B6E3B"):
    ang = np.where(res.reliable, res.smooth, np.nan)  # gaps stay visible as gaps
    df = pd.DataFrame({"t": res.t, "angle": ang})
    x = alt.X("t:Q", title="Time (seconds)")
    y = alt.Y("angle:Q", title="Joint movement (°)", scale=alt.Scale(domain=[0, ymax]))
    layers = [
        alt.Chart(df).mark_area(color=color, opacity=0.12, line=False).encode(x=x, y=y),
        alt.Chart(df).mark_line(color=color, strokeWidth=3, interpolate="monotone").encode(x=x, y=y),
    ]
    if res.status != "rejected":
        layers.append(alt.Chart(pd.DataFrame({"m": [res.mean_peak]})).mark_rule(
            strokeDash=[6, 5], color="#33433B", opacity=.7).encode(y="m:Q"))
    if res.reps:
        pk = pd.DataFrame(res.reps)
        layers.append(alt.Chart(pk).mark_point(size=170, filled=True, color="#EA580C", opacity=1,
                                               stroke="#fff", strokeWidth=2).encode(
            x="t:Q", y="peak:Q", tooltip=["rep", alt.Tooltip("peak:Q", format=".0f")]))
    return themed(alt.layer(*layers).properties(height=250))


def stat_cards(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div><div class="h">{h}</div></div>'
                    for k, v, h in items)
    html(f'<div class="stats">{cells}</div>')


def per_rep_table(res) -> None:
    df = pd.DataFrame(res.reps)[["rep", "t", "peak", "excursion", "units"]]
    df.columns = ["Rep", "Time (s)", "Peak (°)", "Excursion (°)", "Movement units"]
    st.dataframe(df.round(1), hide_index=True, use_container_width=True)


def result_panel(entry, ex, ymax, title, tag) -> None:
    res = entry["res"]
    with st.container(border=True):
        html(f'<div class="ptitle"><span class="tag">{tag}</span>{title}</div>')
        banner(res)
        if entry["video"]:
            st.video(entry["video"])
            st.caption("Skeleton overlay, live angle and rep counter. "
                       + ("Face blurred, audio removed." if blur_face else "Audio removed."))
        if res.status != "rejected":
            stat_cards([
                ("Average peak", f"{res.mean_peak:.0f}°", "highest angle per rep"),
                ("Repetitions", f"{res.n_reps}", "clearly detected"),
                ("Consistency", f"±{res.sd_peak:.0f}°", "rep-to-rep spread"),
                ("Tracking", f"{res.confidence:.0%}", "confidence"),
            ])
        st.altair_chart(signal_chart(res, ymax), use_container_width=True)
        st.caption("Green line = smoothed joint movement · orange dots = detected repetition peaks · "
                   "dashed = average peak · gaps = frames we did not trust.")
        if res.status != "rejected":
            with st.expander("How this was calculated (every number)"):
                st.write(f"Side tracked: **{res.meta['side']}** · {res.meta['duration_s']:.1f} s at "
                         f"{res.fps:.0f} fps · frames with usable tracking: {res.coverage:.0%}")
                per_rep_table(res)
                st.caption("Excursion = peak minus the start/end position of that rep. Movement units: "
                           "1 = one smooth lift; higher = stepwise movement.")


def range_bar(label: str, mean_peak: float, ex, old: bool = False) -> None:
    pct = percent_of_typical(mean_peak, ex)
    w = max(2.0, min(pct, 100.0))
    html(f'<div class="bar"><div class="row"><span>{label}</span><span>{mean_peak:.0f}° · {pct:.0f}% of typical</span></div>'
         f'<div class="track" role="img" aria-label="{label} {pct:.0f} percent of typical range">'
         f'<div class="fill{" old" if old else ""}" style="width:{w}%"></div></div></div>')


def y_limit(*results) -> float:
    peaks = [r.peaks.max() for r in results if r.n_reps]
    return float(max(peaks + [30]) + 20)


# ====================================================================== main tab
with tab_main:
    step(1, "Pick your exercise", "Choose the movement you will film")
    ex_key = st.radio("Exercise", list(EXERCISES), format_func=lambda k: EXERCISES[k].label,
                      horizontal=True, label_visibility="collapsed")
    ex = EXERCISES[ex_key]
    html(f'<div class="callout"><b>How to do it:</b> {ex.setup}</div>')
    st.markdown("**Which limb are you moving?**")
    side_label = st.radio("Which limb are you moving?", ["Auto-detect", "Left", "Right"], horizontal=True,
                          label_visibility="collapsed",
                          help="For before/after comparisons, choose Left or Right so both videos track the same limb.")
    side = {"Auto-detect": "auto", "Left": "left", "Right": "right"}[side_label]

    with st.expander("\U0001F4D0  Filming checklist — 2D video is sensitive to camera position", expanded=False):
        st.markdown(
            "- Camera **side-on**, at hip height, about 2–3 m away, held still (prop it on something).\n"
            "- **Whole body in view**, including the joint you are moving, from start to finish.\n"
            "- Good light from the front or side; avoid strong backlight (windows behind you).\n"
            "- Close-fitting clothing helps tracking. Use the **same spot and set-up** every time.\n"
            "- 3–5 slow repetitions, about 15–30 seconds."
        )

    step(2, "Add your videos", "Add an earlier one to see how far you have come")
    c1, c2 = st.columns(2)
    types = ["mp4", "mov", "m4v", "avi", "webm"]
    with c1:
        up_before = st.file_uploader("Earlier video (optional, for comparison)", type=types, key="up_before")
    with c2:
        up_after = st.file_uploader("Latest video", type=types, key="up_after")

    step(3, "See your progress", "Everything is calculated here, on this device")
    go = st.button("Analyse on this device", type="primary", disabled=up_after is None, use_container_width=True)
    if up_after is None:
        st.caption("Add your latest video above to continue.")
    if go:
        entries = {}
        if up_before is not None:
            entries["before"] = run_one(up_before, ex_key, side, "Earlier video")
        entries["after"] = run_one(up_after, ex_key, side, "Latest video")
        st.session_state["analysis"] = {"ex_key": ex_key, **entries}

    A = st.session_state.get("analysis")
    if not A:
        html('<div class="grid">'
             + icard("\U0001F3AC", "1 · Film", text="Record 3–5 slow repetitions from the side with your whole body in view.")
             + icard("\U0001F9E0", "2 · Analyse", text="On-device pose tracking measures the peak angle of every repetition.")
             + icard("\U0001F4C8", "3 · Understand", text="See your change with its uncertainty, and tips from simple, readable rules.")
             + "</div>")
    else:
        ex = EXERCISES[A["ex_key"]]
        after = A["after"]
        before = A.get("before")
        ra = after["res"]
        rb = before["res"] if before else None

        before_ok = True
        if before and rb.meta["side"] != ra.meta["side"]:
            html('<div class="notice warn" role="alert"><b class="t">Different limbs tracked</b>'
                 "The two videos tracked different limbs, so they can’t be compared fairly. "
                 "Choose Left or Right above and analyse again.</div>")
            before_ok = False
        cmp = compare(rb, ra, cfg) if (before and before_ok) else None

        if cmp:
            cls, lab, icon = {"improved": ("up", "Improved", "▲"),
                              "reduced": ("down", "Lower than before", "▼"),
                              "similar": ("flat", "No clear change", "●")}[cmp.direction]
            html(f'<div class="verdict {cls}"><div><div class="lab">{icon} {lab}</div>'
                 f'<div class="big">{cmp.delta:+.0f}°</div></div>'
                 f'<div class="txt">Average peak {ex.noun} went from <b>{cmp.mean_a:.0f}°</b> to <b>{cmp.mean_b:.0f}°</b>.'
                 f'<div class="sub">95% range ±{cmp.ci:.0f}° · changes under {cfg.meaningful_change:.0f}° are treated as noise, '
                 "so you are never told you improved when it could just be wobble.</div></div></div>")

        ymax = y_limit(*[r for r in (ra, rb) if r is not None])
        if before and before_ok:
            left, right = st.columns(2)
            with left:
                result_panel(before, ex, ymax, "Your earlier video", "Earlier")
            with right:
                result_panel(after, ex, ymax, "Your latest video", "Latest")
        else:
            result_panel(after, ex, ymax, "Your result", "Latest")
            if before:
                result_panel(before, ex, ymax, "Your earlier video", "Earlier")

        if ra.status != "rejected":
            html('<div class="step"><div class="n">★</div><div><h2>Where this puts you</h2>'
                 "<small>Compared with a typical full range</small></div></div>")
            range_bar("Latest", ra.mean_peak, ex)
            if cmp:
                range_bar("Earlier", rb.mean_peak, ex, old=True)
            st.caption(f"Typical full range ({ex.typical_range:.0f}°) is an approximate textbook value for adults without "
                       "injury. Your own goal may differ — ask your physio.")

            if cmp:
                html('<div class="step"><div class="n">≋</div><div><h2>Repetition by repetition</h2>'
                     "<small>Each dot is one repetition</small></div></div>")
                rows = [{"Rep": r["rep"], "Peak": r["peak"], "Video": "Earlier"} for r in rb.reps] + \
                       [{"Rep": r["rep"], "Peak": r["peak"], "Video": "Latest"} for r in ra.reps]
                st.altair_chart(themed(
                    alt.Chart(pd.DataFrame(rows)).mark_line(point=alt.OverlayMarkDef(size=110), strokeWidth=3).encode(
                        x=alt.X("Rep:O", title="Repetition", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Peak:Q", title="Peak angle (°)", scale=alt.Scale(domain=[0, ymax])),
                        color=alt.Color("Video:N", scale=alt.Scale(domain=["Earlier", "Latest"], range=["#8A9A91", "#0B6E3B"])),
                        strokeDash=alt.StrokeDash("Video:N", legend=None)).properties(height=240)),
                    use_container_width=True)

            html('<div class="step"><div class="n">✦</div><div><h2>What this means</h2>'
                 "<small>Plain-language tips from fixed rules</small></div></div>")
            for rid, text in build_tips(ra, ex, rb if cmp else None, cmp):
                html(f'<div class="tip"><span class="rule">{rid}</span><p>{text}</p></div>')
            st.caption("Each tip shows the rule that produced it. No generative AI is used. "
                       "Tips would be reviewed by a clinician before real use.")
        html(f'<div class="safety" role="note"><b>Safety.</b> {SAFETY_NOTE}</div>')

# ====================================================================== history tab
with tab_hist:
    step(1, "My history", "Numbers only — never video")
    hist = store.load_history()
    if not hist:
        html('<div class="grid">' + icard("\U0001F4D2", "Nothing saved yet",
             text="Switch on <b>Save my numbers on this device</b> in the sidebar. Each analysis then adds one row of "
                  "numbers here. You can download or delete it at any time.") + "</div>")
    else:
        df = pd.DataFrame(hist)
        df["when"] = pd.to_datetime(df["saved_at"])
        st.altair_chart(themed(
            alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=110), strokeWidth=3).encode(
                x=alt.X("when:T", title="Date"), y=alt.Y("mean_peak:Q", title="Average peak (°)"),
                color=alt.Color("exercise:N", scale=alt.Scale(range=["#0B6E3B", "#EA580C", "#2563EB", "#7C3AED"])),
                tooltip=["exercise", "side", "mean_peak", "n_reps", "label"]).properties(height=300)),
            use_container_width=True)
        st.dataframe(df.drop(columns=["when"]), hide_index=True, use_container_width=True)
        st.download_button("Download my data (CSV)", df.drop(columns=["when"]).to_csv(index=False),
                           "spinemotion_history.csv", "text/csv")
        st.caption("This file lives only in the folder shown in the Data & safety tab. It is yours.")

# ====================================================================== data & safety tab
with tab_data:
    step(1, "Data & safety", "What happens to your information, in plain words")
    html('<div class="callout"><b>Try it:</b> switch off Wi-Fi and run an analysis. It works identically, '
         "because nothing leaves this device.</div>")
    html('<div class="grid">'
         + icard("\U0001F512", "Privacy", [
             "All analysis runs on this device: no uploads, accounts, cloud or API calls.",
             "Your <b>original video is never saved</b>; it is analysed and discarded.",
             "The result video is temporary, with the <b>face blurred</b> and <b>audio removed</b>.",
             "Numbers are saved only if you switch that on. <b>Delete everything</b> wipes it all."])
         + icard("\U0001F6E1️", "Security", [
             "No server and no network service: no remote attack surface and no data in transit.",
             "Files live in one folder; the list below shows exactly what is held right now.",
             "Honest limit: saved history is not encrypted, so protect your device with a passcode."])
         + icard("⚖️", "Fairness", [
             "Pose models are mostly trained on able-bodied people, so they can be <b>less accurate for seated, "
             "wheelchair or assisted postures</b>, other body shapes, dark clothing and poor light.",
             "So SpineMotion checks its own reliability and <b>refuses to score</b> instead of guessing.",
             "We tested across conditions and publish where it fails (below)."])
         + icard("\U0001F50D", "Transparency", [
             "Every number can be expanded: per-rep peaks, tracking coverage, confidence.",
             "Tips use fixed rules, each labelled with its rule ID. No generative AI.",
             "A change is called real only if it exceeds rep-to-rep noise <b>and</b> 5°."])
         + icard("\U0001F9EC", "Data provenance", [
             "Pose model: Google MediaPipe Pose, pre-trained and bundled, no download.",
             "All test footage was recorded with consent from the team."])
         + icard("\U0001F9ED", "Known limitations", [
             "Single 2D camera: angles are projections and need a side-on view.",
             "Not a medical device; not validated against clinical goniometry in a trial.",
             "Face blur is best-effort: it may miss frames where the face isn’t detected.",
             "Typical-range values are textbook averages for uninjured adults."])
         + "</div>")

    st.markdown("### What this app holds right now")
    files = store.audit()
    if files:
        st.dataframe(pd.DataFrame(files, columns=["File", "Bytes"]), hide_index=True, use_container_width=True)
    else:
        html('<div class="notice ok" role="status"><b class="t">✓ No files stored</b>Nothing is being kept on this device.</div>')
    st.caption(f"Data folder: {store.DATA_DIR} · Temp folder: {store.TMP}")

    report = Path(__file__).with_name("validation_report.md")
    st.markdown("### Validation & fairness testing")
    if report.exists():
        st.markdown(report.read_text())
    else:
        st.caption("Run `python validate.py` on your labelled clips to show validation results here.")
    html(f'<div class="safety" role="note"><b>Safety.</b> {SAFETY_NOTE}</div>')
