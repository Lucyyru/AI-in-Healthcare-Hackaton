"""SpyMotion - patient-owned, on-device movement progress tracking.  Run: streamlit run app.py"""

# CODE MAP / QUICK NAVIGATION
# 1. Imports and app setup              -> dependencies and Streamlit page settings
# 2. VISUALS: CSS design system         -> colours, layout, cards, buttons, tabs, ratings
# 3. Visual helper functions            -> HTML snippets, charts, stats, result panels
# 4. Sidebar                             -> privacy toggles and advanced settings
# 5. Hero and navigation tabs           -> top banner and the four main app areas
# 6. Exercise analysis                  -> upload, analyse, compare, results and tips
# 7. Calendar history                   -> saved movement-analysis history and chart
# 8. Data & safety                      -> data inventory, validation and limitations
# 9. PROM storage helpers               -> daily wellbeing JSON read/write and averages
# 10. CALENDAR VISUALS + DATE SELECTOR  -> month navigation and selectable daily buttons
# 11. DAILY CHECK-IN FORM               -> symptoms, sleep, medication, routines, activities
# 12. INSIGHTS / CHARTS                 -> weekly/monthly averages and patterns
# 13. EXPORT / DELETE                   -> CSV/JSON downloads and diary deletion
# Tip: use Ctrl+F to search for one of the ALL-CAPS section labels below.

from __future__ import annotations

import hashlib
import calendar as pycalendar
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core
import store
from core import CFG, EXERCISES, SAFETY_NOTE, SignalConfig, build_tips, compare, percent_of_typical
from robust import analyze_upload

st.set_page_config(
    page_title="SpyMotion",
    page_icon="\U0001F9B4",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================================================================
# VISUALS AREA 1/2: GLOBAL DESIGN SYSTEM
# Edit this CSS block for app-wide colours, typography, spacing, cards, tabs and buttons.
# Keep CSS changes here rather than mixing visual styling into data-processing code.
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
.block-container {
    max-width: 600px;
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-bottom: 4rem;
    margin: 0 auto;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
h1,h2,h3 { letter-spacing: -0.02em; color: var(--ink); }
p, li { line-height: 1.6; }
:focus-visible { outline: 3px solid var(--g700) !important; outline-offset: 2px; border-radius: 8px; }

/* VISUALS: main hero/banner at the top of the app */
.hero{
    background: linear-gradient(135deg,#0B3D2E 0%,#0B6E3B 60%,#16A34A 120%);
    color:#fff;
    border-radius:20px;
    padding:1.4rem 1.2rem;
    box-shadow:var(--shadow);
    position:relative;
    overflow:hidden;
    margin-bottom:1rem;
}
.hero:after{ content:""; position:absolute; right:-70px; top:-70px; width:300px; height:300px; border-radius:50%; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.22), rgba(255,255,255,0) 70%);}
.brand{ display:flex; align-items:center; gap:.7rem; font-weight:700; font-size:1.05rem; letter-spacing:.02em; opacity:.95;}
.hero h1{
    color:#fff;
    font-size:2rem;
    line-height:1.1;
    margin:.7rem 0 .4rem;
    font-weight:800;
}
.hero p{
    color:#E3F4E8;
    font-size:1rem;
    margin:0 0 1rem;
}
.pills{ display:flex; flex-wrap:wrap; gap:.5rem; position:relative; z-index:1;}
.pill{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35); color:#fff; padding:.35rem .85rem; border-radius:999px; font-size:.92rem; font-weight:600; backdrop-filter: blur(4px);}

/* VISUALS: numbered section headings */
.step{ display:flex; align-items:center; gap:.9rem; margin:2rem 0 .8rem;}
.step .n{ flex:0 0 auto; width:2.4rem; height:2.4rem; border-radius:50%; background:var(--g700); color:#fff; font-weight:800; display:grid; place-items:center; box-shadow:0 4px 12px rgba(11,110,59,.35);}
.step h2{ margin:0; font-size:1.55rem; font-weight:800; padding:0;}
.step small{ display:block; color:var(--muted); font-size:1rem; font-weight:500; margin-top:.1rem;}

/* VISUALS: notices, safety callouts and status messages */
.callout{ background:var(--mint2); border:1px solid var(--line); border-left:6px solid var(--g500); border-radius:14px; padding:.9rem 1.1rem; margin:.4rem 0 1rem; }
.notice{ border-radius:16px; padding:.95rem 1.2rem; margin:.5rem 0 1rem; border:1px solid; }
.notice b.t{ display:block; font-size:1.05rem; margin-bottom:.2rem;}
.notice ul{ margin:.2rem 0 0; padding-left:1.2rem;}
.notice.ok{ background:#EAF7EE; border-color:#B5E0C1; color:#0B3D2E;}
.notice.warn{ background:var(--amberbg); border-color:#F3D58C; color:#5A3A00;}
.notice.bad{ background:var(--redbg); border-color:#F2B8A5; color:#6B1D00;}
.safety{ background:var(--amberbg); border:1px solid #F3D58C; border-radius:16px; padding:1rem 1.2rem; color:#5A3A00; margin-top:1.4rem; font-size:.97rem;}

/* VISUALS: movement-analysis verdict panel */
.verdict{ border-radius:24px; padding:1.6rem 1.8rem; display:flex; gap:1.6rem; align-items:center; flex-wrap:wrap; box-shadow:var(--shadow); border:1px solid var(--line); background:#fff; margin:1.2rem 0 .6rem;}
.verdict .big{ font-size:4rem; font-weight:800; line-height:1; letter-spacing:-0.04em;}
.verdict .lab{ font-size:.85rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.35rem;}
.verdict .txt{ flex:1 1 320px; font-size:1.12rem;}
.verdict .txt .sub{ color:var(--muted); font-size:.98rem; margin-top:.3rem;}
.verdict.up{ background:linear-gradient(135deg,#EAF7EE,#FFFFFF 70%); border-color:#B5E0C1;} .verdict.up .big,.verdict.up .lab{ color:var(--g700);}
.verdict.down{ background:linear-gradient(135deg,#FDEDE8,#FFFFFF 70%); border-color:#F2B8A5;} .verdict.down .big,.verdict.down .lab{ color:var(--red);}
.verdict.flat{ background:linear-gradient(135deg,#EEF2EF,#FFFFFF 70%);} .verdict.flat .big,.verdict.flat .lab{ color:#33433B;}

/* VISUALS: result panels and metric cards */
.ptitle{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.25rem; margin-bottom:.4rem;}
.tag{ font-size:.75rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; background:var(--mint); color:var(--g700); padding:.2rem .6rem; border-radius:999px;}
.stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:.7rem; margin:.7rem 0 .4rem;}
.stat{ background:var(--mint2); border:1px solid var(--line); border-radius:16px; padding:.8rem .95rem;}
.stat .k{ font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);}
.stat .v{ font-size:2rem; font-weight:800; letter-spacing:-0.03em; line-height:1.15; margin-top:.15rem; color:var(--g900);}
.stat .h{ font-size:.82rem; color:var(--muted); margin-top:.1rem;}

/* VISUALS: range-of-motion comparison bars */
.bar{ margin:.55rem 0 1rem;}
.bar .row{ display:flex; justify-content:space-between; font-weight:600; font-size:.98rem; margin-bottom:.3rem;}
.bar .track{ height:16px; background:#E6EEE8; border-radius:999px; overflow:hidden;}
.bar .fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,#16A34A,#0B6E3B);}
.bar .fill.old{ background:linear-gradient(90deg,#A8B8AE,#7C8E83);}

/* VISUALS: plain-language tip cards */
.tip{ display:flex; gap:.9rem; align-items:flex-start; background:#fff; border:1px solid var(--line); border-radius:16px; padding:.9rem 1.1rem; margin:.55rem 0; box-shadow:0 1px 2px rgba(15,42,32,.04);}
.tip p{ margin:0; font-size:1.02rem;}
.rule{ flex:0 0 auto; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.78rem; font-weight:700; padding:.2rem .55rem; border-radius:8px; background:var(--mint); color:var(--g700); margin-top:.1rem;}

/* VISUALS: informational cards and how-it-works grid */
.grid{
    display:grid;
    grid-template-columns:1fr;
    gap:.8rem;
    margin:.6rem 0;
}
.icard{ background:#fff; border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.3rem; box-shadow:var(--shadow);}
.icard .ic{ width:2.6rem; height:2.6rem; border-radius:14px; background:var(--mint); display:grid; place-items:center; font-size:1.3rem; margin-bottom:.6rem;}
.icard h3{ margin:.1rem 0 .4rem; font-size:1.15rem; font-weight:800;}
.icard ul{ margin:0; padding-left:1.1rem; color:#26382F;} .icard li{ margin:.25rem 0;}
.icard p{ margin:0; color:#26382F;}

/* VISUALS: shared Streamlit buttons, uploaders, expanders and tables */
.stButton > button{ min-height:3.0rem; border-radius:999px; padding:.55rem 1.15rem; font-size:1.02rem; font-weight:700; border:1.5px solid var(--line); transition:transform .16s ease, box-shadow .16s ease, background .16s ease;}
.stButton > button:hover{ transform:translateY(-1px); border-color:var(--g700); color:var(--g700); box-shadow:0 4px 12px rgba(11,110,59,.10);}
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

/* VISUALS: main navigation tab styling */
div[data-baseweb="tab-list"]{ gap:.5rem; border-bottom:none; background:var(--mint); padding:.35rem; border-radius:999px; width:fit-content;}
button[data-baseweb="tab"]{ border-radius:999px; padding:.55rem 1.3rem; height:auto; font-weight:700; color:var(--g900);}
button[data-baseweb="tab"][aria-selected="true"]{ background:#fff; color:var(--g700); box-shadow:0 2px 8px rgba(15,42,32,.12);}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"]{ display:none;}

/* VISUALS: radio-button styling (including tap-to-rate controls) */
div[role="radiogroup"]{ gap:.5rem;}
label[data-baseweb="radio"]{ background:#fff; border:1.5px solid var(--line); border-radius:999px; padding:.5rem 1.0rem; margin:0; transition:all .16s ease; cursor:pointer;}
label[data-baseweb="radio"] > div:first-child{ display:none;}
label[data-baseweb="radio"]:hover{ border-color:var(--g700);}
label[data-baseweb="radio"]:has(input:checked){ background:var(--g700); border-color:var(--g700); box-shadow:0 4px 12px rgba(11,110,59,.3);}
label[data-baseweb="radio"]:has(input:checked) *{ color:#fff !important; font-weight:700;}
label[data-baseweb="radio"]:has(input:focus-visible){ outline:3px solid var(--g700); outline-offset:2px;}

/* VISUALS AREA 2/2: DAILY WELLBEING DASHBOARD
   PROM-specific banner, sections, calendar and rating-button styles. */
.prom-banner{background:linear-gradient(125deg,#0B3D2E 0%,#0B6E3B 62%,#248653 120%);border:1px solid #0B6E3B;border-radius:20px;padding:1.5rem 1.7rem;margin:.4rem 0 1rem;box-shadow:0 8px 24px rgba(16,43,63,.14);}
.prom-banner .eyebrow{font-size:.76rem;letter-spacing:.13em;text-transform:uppercase;font-weight:800;color:#C6EBD1;margin-bottom:.35rem;}
.prom-banner h2{margin:.1rem 0 .45rem;font-size:1.85rem;color:#FFFFFF;}
.prom-banner p{margin:0;color:#E3F4E8;font-size:1.02rem;}
.prom-note{background:#F2F8F3;border:1px solid #D7E7DA;border-radius:16px;padding:.8rem 1rem;color:#234A33;margin:.6rem 0 1rem;}
.prom-section{background:#EDF6EF;border:1px solid #D4E6D8;border-left:4px solid #0B6E3B;border-radius:16px;padding:.85rem 1rem;margin:.9rem 0 .5rem;color:#17452D;font-weight:800;font-size:1.08rem;}
.prom-mini{color:#5A7161;font-size:.9rem;margin:.1rem 0 .5rem;}
.prom-calendar{background:#F7FBF7;border:1px solid #D9E8DC;border-radius:20px;padding:.7rem .6rem;margin:.6rem 0 1rem;}
.prom-save-hint{background:#EAF6ED;border:1px solid #C6E2CD;border-radius:16px;padding:.8rem 1rem;color:#155B35;margin:.5rem 0 1rem;}
/* ratings are clear, accessible tap targets rather than sliders */
div[role="radiogroup"]{display:flex;flex-wrap:wrap;gap:.4rem;}
label[data-baseweb="radio"]{background:#FFFFFF;border:1.5px solid #D2E3D7;border-radius:999px;padding:.48rem .85rem;margin:0;min-height:2.5rem;justify-content:center;box-shadow:0 1px 3px rgba(11,61,46,.05);}
label[data-baseweb="radio"]:hover{border-color:#0B6E3B;background:#F0F7F1;}
label[data-baseweb="radio"]:has(input:checked){background:#0B6E3B;border-color:#0B6E3B;box-shadow:0 3px 9px rgba(11,110,59,.18);}
label[data-baseweb="radio"]:has(input:checked) *{color:#fff !important;font-weight:800;}
.stButton > button[kind="secondary"]{background:#F5FAF6;border-color:#D5E5D9;color:#174B31;}
.stButton > button[kind="primary"]{background:linear-gradient(135deg,#0B6E3B,#16A34A);color:#fff;border-color:transparent;box-shadow:0 5px 14px rgba(11,110,59,.20);}
.stButton > button[kind="primary"]:hover{border-color:#0B3D2E;color:#fff;box-shadow:0 7px 18px rgba(11,110,59,.26);}
/* VISUAL POLISH: consistent rounded controls, calm greens and gentle interaction states. */
.stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="stDateInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input { border-radius:16px !important; }
.stDownloadButton > button { background:#F3F9F4; border:1.5px solid #D2E4D6; color:#174B31; font-weight:700; }
.stDownloadButton > button:hover { background:#E8F4EB; border-color:#0B6E3B; color:#0B3D2E; }
[data-testid="stFormSubmitButton"] > button { border-radius:999px !important; min-height:3.1rem; }
[data-testid="stDateInput"] input:focus, [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#0B6E3B !important; box-shadow:0 0 0 2px rgba(11,110,59,.12) !important; }
[data-testid="stAlert"] { border-radius:16px; }
[data-testid="stMetric"] { background:#F5FAF6; border:1px solid #DCE9DF; border-radius:18px; padding:1rem; }
@media (max-width:760px){ .hero{padding:1.5rem 1.3rem;} .hero h1{font-size:2rem;} .verdict .big{font-size:3rem;} .prom-banner{padding:1rem;} .prom-banner h2{font-size:1.45rem;} label[data-baseweb="radio"]{padding:.4rem .6rem;} }
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; } }

/* VISUAL REFINEMENT: SOFT GREEN PALETTE + ROUNDED CONTROLS
   Keep all interactive elements within SpyMotion's calm green design system.
   This final override block is intentionally centralised for easy future editing. */
:root{
  --ink:#15382B; --muted:#61766A; --line:#D7E6DA; --mint:#E8F4EB; --mint2:#F5F9F5;
  --g900:#123D2D; --g700:#176B43; --g500:#319B63; --card:#FFFFFF;
  --shadow:0 2px 6px rgba(18,61,45,.04), 0 10px 28px rgba(18,61,45,.06);
}
.stButton > button{
  border-radius:999px !important; min-height:2.9rem; padding:.55rem 1.15rem;
  border:1px solid #D2E3D7; background:#FFFFFF; color:#214A36;
  box-shadow:0 2px 5px rgba(18,61,45,.04);
}
.stButton > button:hover{
  border-color:#75B798 !important; background:#F1F8F3 !important; color:#0B5A36 !important;
  box-shadow:0 5px 14px rgba(18,61,45,.09); transform:translateY(-1px);
}
.stButton > button[kind="primary"]{
  background:linear-gradient(135deg,#176B43,#319B63) !important; color:#FFFFFF !important;
  border:1px solid #176B43 !important; border-radius:999px !important;
  box-shadow:0 5px 14px rgba(23,107,67,.18);
}
.stButton > button[kind="primary"]:hover{
  background:linear-gradient(135deg,#125735,#278653) !important; color:#FFFFFF !important;
  box-shadow:0 7px 18px rgba(23,107,67,.24);
}
/* Radio inputs become rounded, pill-shaped tap targets with a soft green selected state. */
div[role="radiogroup"]{gap:.45rem .5rem;}
label[data-baseweb="radio"]{
  background:#FFFFFF !important; border:1px solid #D2E3D7 !important;
  border-radius:999px !important; padding:.48rem .95rem !important; min-height:2.65rem;
  box-shadow:0 1px 3px rgba(18,61,45,.035); transition:background .15s ease,border-color .15s ease,box-shadow .15s ease;
}
label[data-baseweb="radio"]:hover{background:#F1F8F3 !important;border-color:#75B798 !important;}
label[data-baseweb="radio"]:has(input:checked){
  background:#176B43 !important; border-color:#176B43 !important;
  box-shadow:0 3px 9px rgba(23,107,67,.16);
}
label[data-baseweb="radio"]:has(input:checked) *{color:#FFFFFF !important;font-weight:750;}
label[data-baseweb="radio"]:has(input:focus-visible){outline:3px solid #75B798;outline-offset:2px;}
/* Calendar day buttons stay compact while keeping the same rounded green language. */
.prom-calendar .stButton > button{min-height:2.55rem;padding:.35rem .25rem;border-radius:14px !important;font-size:.94rem;}
.prom-banner{background:linear-gradient(125deg,#123D2D 0%,#176B43 65%,#319B63 120%);border-color:#D7E6DA;}
.prom-banner .eyebrow{color:#C7E8D3;}
.prom-note{background:#F3F8F4;border-color:#D7E6DA;color:#315441;}
.prom-section{background:#EFF7F0;border-color:#D7E6DA;border-left-color:#319B63;color:#214A36;border-radius:16px;}
.prom-calendar{background:#F7FAF7;border-color:#D7E6DA;border-radius:20px;}
.prom-save-hint{background:#EAF5ED;border-color:#C5E0CD;color:#1A5A39;}
[data-testid="stFileUploaderDropzone"]{border-color:#9BCBAA;background:#FCFEFC;}
[data-testid="stFileUploaderDropzone"]:hover{background:#F1F8F3;border-color:#319B63;}
[data-testid="stExpander"]{border-radius:20px;border-color:#D7E6DA;}
div[data-baseweb="tab-list"]{background:#EAF4EC;border:1px solid #D7E6DA;}
button[data-baseweb="tab"][aria-selected="true"]{color:#176B43;}
@media (max-width:760px){.stButton > button{min-height:2.7rem;padding:.45rem .85rem;}label[data-baseweb="radio"]{padding:.4rem .7rem !important;}}
/* Accessibility/mobile behaviour retained from the compact app version. */
@media (max-width:600px){
  html, body, [class*="css"] { font-size:18px; }
  .block-container { max-width:600px; padding-left:.8rem; padding-right:.8rem; }
  .stButton > button { width:100%; min-height:3.5rem; font-size:1.1rem; font-weight:600; }
  [data-testid="stFileUploader"] { padding:.5rem 0; }
  [data-testid="stMetricValue"] { font-size:2rem; }
  h1 { font-size:2rem !important; }
  h2 { font-size:1.4rem !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO = (
    '<svg width="34" height="34" viewBox="0 0 34 34" role="img" aria-label="SpyMotion logo">'
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


# ------------------------------------------------------------------ sidebar
# ======================================================================
# SIDEBAR AREA: PRIVACY CONTROLS AND ADVANCED SETTINGS
# Edit sidebar controls here; processing uses their values below.
# ====================================================================== sidebar
with st.sidebar:
    html(f'<div class="brand">{LOGO.replace("rgba(255,255,255,.16)", "#0B6E3B")}<span>SpyMotion</span></div>')
    st.markdown("### Display")

    view_mode = st.radio(
        "Choose how SpyMotion looks",
        ["📱 Phone", "🖥️ Website"],
        horizontal=True,
        label_visibility="collapsed",
        key="view_mode",
    )
    st.divider()
    st.caption("Your movement. Your data. Your device.")
    blur_face = st.toggle("Blur my face in the result video", value=True)
    save_history = st.toggle("Save my numbers on this device (never video)", value=False)
    show_medication = st.toggle("Show optional medication & routines tab", value=False,
                                help="Off by default to keep the app focused on movement. Entries stay on this device and are deleted by Delete everything.")
    with st.expander("Advanced settings"):
        conf_thresh = st.slider("Minimum tracking confidence per frame", 0.3, 0.9, CFG.conf_thresh, 0.05,
                                help="Frames below this are ignored, not guessed.")
        min_prom = st.slider("Smallest movement counted as a repetition (°)", 4, 20, int(CFG.min_prominence))
        complexity = 1  # the only pose model bundled in the package; 0 and 2 download from Google at first use
        st.caption("Pose model: MediaPipe Pose (standard), bundled with the app. It never downloads anything.")
    st.divider()
    if st.button("\U0001F5D1  Delete everything now", use_container_width=True):
        _rep = store.delete_everything_report()
        st.session_state.clear()
        if _rep["failed"]:
            st.error(f"Deleted {_rep['deleted']} file(s) but could not remove {len(_rep['failed'])}: "
                     + ", ".join(Path(f).name for f in _rep["failed"]))
        else:
            st.success(f"Deleted {_rep['deleted']} file(s) and cleared this session.")
    st.caption("Wipes saved numbers, the wellbeing diary, result videos and everything held in memory.")

cfg = SignalConfig(conf_thresh=float(conf_thresh), min_prominence=float(min_prom))
# ======================================================================
# DISPLAY MODE
# Phone mode keeps the interface narrow and app-like.
# Website mode uses the available desktop width.
# ======================================================================

if view_mode == "📱 Phone":
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 600px !important;
            margin: 0 auto !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .grid {
            grid-template-columns: 1fr !important;
        }

        .hero {
            padding: 1.4rem 1.2rem !important;
            border-radius: 20px !important;
        }

        .hero h1 {
            font-size: 2rem !important;
        }

        .hero p {
            font-size: 1rem !important;
        }

        .stats {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px !important;
            margin: 0 auto !important;
        }

        .grid {
            grid-template-columns:
                repeat(auto-fit, minmax(250px, 1fr)) !important;
        }

        .hero {
            padding: 2.2rem 2.4rem 1.8rem !important;
        }

        .hero h1 {
            font-size: 2.7rem !important;
        }

        .hero p {
            font-size: 1.15rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# ======================================================================
# HEADER / NAVIGATION AREA: BRAND BANNER AND TOP-LEVEL TABS
# Change the page intro and tab names here.
# ====================================================================== hero
html(
    f'<div class="hero"><div class="brand">{LOGO}<span>SpyMotion</span></div>'
    "<h1>See your recovery, privately.</h1>"
    "<p>Film an exercise. Get your range of motion for every repetition and a clear before-and-after, "
    "calculated entirely on this device.</p>"
    '<div class="pills"><span class="pill">\U0001F512 100% on-device</span><span class="pill">☁️ No uploads</span>'
    '<span class="pill">\U0001F464 No accounts</span><span class="pill">\U0001F3AC Original video never saved</span>'
    '<span class="pill">\U0001F6D1 Refuses to score when unsure</span></div></div>'
)
_tab_labels = ["Check my progress", "My history", "Daily wellbeing"]
if show_medication:
    _tab_labels.append("Medication & routines")
_tab_labels.append("Data & safety")
_tabs = st.tabs(_tab_labels)
tab_main, tab_hist, tab_prom = _tabs[0], _tabs[1], _tabs[2]
tab_medication = _tabs[3] if show_medication else None
tab_data = _tabs[-1]

# ------------------------------------------------------------------ helpers
# ======================================================================
# ANALYSIS HELPER AREA: VIDEO PROCESSING AND REUSABLE RESULT UI
# Functions below are called by the main exercise-analysis tab.
# ====================================================================== helpers
def run_one(upl, ex_key: str, side: str, label: str) -> dict:
    data = upl.getvalue()
    key = hashlib.sha256(data + repr((ex_key, side, cfg, blur_face, complexity)).encode()).hexdigest()[:20]
    cache = st.session_state.setdefault("cache", {})
    if key in cache:
        return cache[key]
    bar = st.progress(0.0, text=f"{label}: starting")
    try:
        # never raises for bad input; the original footage is always removed afterwards
        res, video = analyze_upload(
            data, Path(upl.name).suffix, key[:8], ex_key, side, cfg, blur_face, complexity,
            progress=lambda f, t: bar.progress(min(f, 1.0), text=f"{label}: {t}"),
        )
    finally:
        bar.empty()
    entry = {"res": res, "video": video, "label": label, "key": key}
    if not res.meta.get("failed"):  # do not cache a failure: a retry should really retry
        cache[key] = entry
    if save_history and res.status != "rejected":
        saved = st.session_state.setdefault("saved", set())
        if key not in saved:
            if store.append_entry({"exercise": ex_key, "side": res.meta["side"], "label": label, **res.summary()}):
                saved.add(key)
            else:
                st.warning("Your result is shown, but it could not be saved to your history (disk problem).")
    return entry


def banner(res) -> None:
    if res.status == "rejected":
        items = "".join(f"<li>{r}</li>" for r in res.reasons)
        html(f'<div class="notice bad" role="alert"><b class="t">We can’t score this video reliably</b><ul>{items}</ul>'
             "<div style='margin-top:.5rem'>Rather than guess, SpyMotion shows no score. "
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
        layers.append(alt.Chart(pk).mark_point(size=170, filled=True, color="#16A34A", opacity=1,
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


# ------------------------------------------------------------------ main tab
# ======================================================================
# EXERCISE ANALYSIS AREA: EXERCISE CHOICE, VIDEO UPLOAD, RESULTS AND TIPS
# Main workflow for analysing movement videos.
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
        if before and rb.status != "rejected" and ra.status != "rejected" and rb.meta["side"] != ra.meta["side"]:
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

# ------------------------------------------------------------------ history tab
# ======================================================================
# MOVEMENT HISTORY AREA: SAVED ANALYSIS TABLE, TREND CHART AND CSV EXPORT
# This section reads movement-analysis history from store.py.
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
        for _c in ("exercise", "side", "label", "n_reps", "saved_at"):  # tolerate rows from older versions / hand edits
            if _c not in df.columns:
                df[_c] = None
        df["mean_peak"] = pd.to_numeric(df.get("mean_peak"), errors="coerce")
        df["when"] = pd.to_datetime(df["saved_at"], errors="coerce")
        df = df.dropna(subset=["when", "mean_peak"]).reset_index(drop=True)
        st.altair_chart(themed(
            alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=110), strokeWidth=3).encode(
                x=alt.X("when:T", title="Date"), y=alt.Y("mean_peak:Q", title="Average peak (°)"),
                color=alt.Color("exercise:N", scale=alt.Scale(range=["#0B3D2E", "#0B6E3B", "#16A34A", "#75B798"])),
                tooltip=["exercise", "side", "mean_peak", "n_reps", "label"]).properties(height=300)),
            use_container_width=True)
        _diary = store.load_prom()[0]
        if _diary:
            _daily = df.assign(day=df["when"].dt.strftime("%Y-%m-%d")).groupby("day", as_index=False)["mean_peak"].mean()
            _feel = pd.DataFrame([{"day": d, "Pain": e.get("pain"), "Fatigue": e.get("fatigue")}
                                  for d, e in _diary.items() if isinstance(e, dict)])
            if not _feel.empty:
                _feel[["Pain", "Fatigue"]] = _feel[["Pain", "Fatigue"]].apply(pd.to_numeric, errors="coerce")
                _both = _daily.merge(_feel, on="day", how="inner")
                if len(_both) >= 2:
                    st.markdown("#### Movement and how you felt")
                    _top = alt.Chart(_both).mark_line(point=True, strokeWidth=3, color="#0B6E3B").encode(
                        x=alt.X("day:T", title=None), y=alt.Y("mean_peak:Q", title="Average peak (°)")).properties(height=150)
                    _long = _both.melt(id_vars="day", value_vars=["Pain", "Fatigue"], var_name="Symptom", value_name="Score")
                    _bottom = alt.Chart(_long).mark_line(point=True, strokeWidth=3).encode(
                        x=alt.X("day:T", title="Date"),
                        y=alt.Y("Score:Q", title="Score (0–10)", scale=alt.Scale(domain=[0, 10])),
                        color=alt.Color("Symptom:N", scale=alt.Scale(range=["#C2410C", "#6B7F74"]))).properties(height=150)
                    st.altair_chart(themed(alt.vconcat(_top, _bottom)), use_container_width=True)
                    st.caption("Days where you logged both. This shows patterns only; it cannot show what causes what.")
        st.dataframe(df.drop(columns=["when"]), hide_index=True, use_container_width=True)
        st.download_button("Download my data (CSV)", df.drop(columns=["when"]).to_csv(index=False),
                           "spymotion_history.csv", "text/csv")
        st.caption("This file lives only in the folder shown in the Data & safety tab. It is yours.")

# ------------------------------------------------------------------ data & safety tab
# ======================================================================
# DATA & SAFETY AREA: PRIVACY EXPLANATION, FILE AUDIT AND VALIDATION REPORT
# Edit user-facing safety/limitations text and data inventory here.
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
             "Movement numbers are saved only if you switch that on; diary entries only when you press Save. <b>Delete everything</b> wipes all of it, and the list below proves it."])
         + icard("\U0001F6E1️", "Security", [
             "No server and no network service: no remote attack surface and no data in transit.",
             "Files live in one folder; the list below shows exactly what is held right now.",
             "Honest limit: saved history and diary entries are not encrypted, so protect your device with a passcode."])
         + icard("⚖️", "Fairness", [
             "Pose models are mostly trained on able-bodied people, so they can be <b>less accurate for seated, "
             "wheelchair or assisted postures</b>, other body shapes, dark clothing and poor light.",
             "So SpyMotion checks its own reliability and <b>refuses to score</b> instead of guessing.",
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
    stress = Path(__file__).with_name("stress_report.md")
    if stress.exists():
        st.markdown(stress.read_text(encoding="utf-8"))
    html(f'<div class="safety" role="note"><b>Safety.</b> {SAFETY_NOTE}</div>')


# ======================================================================
# DAILY WELLBEING / PROM AREA
# Subsections below are deliberately separated: STORAGE, CALENDAR, FORM, INSIGHTS, EXPORT.
# ====================================================================== patient-reported outcomes (PROMs)
# Local JSON storage: entries stay in this app folder; this prototype has no cloud sync.
# PROM STORAGE: local JSON file path and default values for a new daily entry.
# The diary lives in store.py (DATA_DIR/wellbeing_entries.json): atomic writes, owner-only permissions,
# listed in the Data & safety audit and wiped by "Delete everything".

PROM_DEFAULTS = {
    "pain": 0,
    "fatigue": 0,
    "spasms": 0,
    "sleep_quality": 3,
    "mood": 3,
    "overall_wellbeing": 3,
    "medication": "Not applicable",
    "medication_missed_reason": "",
    "rehab_activity": "Not recorded",
    "notes": "",
    "pain_location": [],
    "pain_type": "Not specified",
    "skin_check": "Not recorded",
    "bowel_routine": "Not applicable",
    "bladder_routine": "Not applicable",
    "hydration": "Not recorded",
    "sleep_hours": 8.0,
    "activities": {},
}

# PROM STORAGE HELPERS: load, save and calculate averages.
def load_prom_entries() -> dict:
    entries, ok = store.load_prom()
    if not ok:
        st.warning("Your wellbeing file could not be read, so it was kept aside (not overwritten) in the data folder.")
    return entries

def save_prom_entries(entries: dict) -> None:
    if not store.save_prom(entries):
        st.error("Could not save your check-in (disk full or no permission). Nothing was changed.")
        st.stop()  # skip the 'saved successfully' message that follows

def prom_average(entries: dict, key: str, dates: list[str]):
    vals = []
    for day in dates:
        item = entries.get(day, {})
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            vals.append(float(value))
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

with tab_prom:
    # PROM VISUALS / INTRO: banner and short privacy note at the top of the wellbeing tab.
    html(
        '<div class="prom-banner"><div class="eyebrow">DAILY PATIENT-REPORTED OUTCOMES</div>'
        '<h2>Daily wellbeing check-in</h2>'
        '<p>Record symptoms, routines and daily function over time. '
        'Review your personal trends and share a summary with your care team.</p></div>'
    )
    html('<div class="prom-note"><b>Your data, your choice.</b> Entries are optional. Skip any question you do not want to answer.</div>')

    # CALENDAR AREA 1/2: load diary entries and initialise the selected date.
    prom_entries = load_prom_entries()
    today = date.today()
    if "prom_selected_date" not in st.session_state:
        st.session_state["prom_selected_date"] = today.isoformat()

    # CALENDAR AREA 2/2: previous/next month controls and month picker.
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("← Previous month", key="prom_prev_month", use_container_width=True):
            current = date.fromisoformat(st.session_state["prom_selected_date"])
            first = current.replace(day=1)
            previous_last = first - timedelta(days=1)
            st.session_state["prom_selected_date"] = previous_last.replace(day=min(current.day, previous_last.day)).isoformat()
            st.rerun()
    with nav2:
        selected_month = st.date_input(
            "Choose a month",
            value=date.fromisoformat(st.session_state["prom_selected_date"]).replace(day=1),
            key="prom_month_picker",
            label_visibility="collapsed",
        )
    with nav3:
        if st.button("Next month →", key="prom_next_month", use_container_width=True):
            current = date.fromisoformat(st.session_state["prom_selected_date"])
            first = current.replace(day=1)
            if first.month == 12:
                nxt = first.replace(year=first.year + 1, month=1)
            else:
                nxt = first.replace(month=first.month + 1)
            last_day = pycalendar.monthrange(nxt.year, nxt.month)[1]
            st.session_state["prom_selected_date"] = nxt.replace(day=min(current.day, last_day)).isoformat()
            st.rerun()

    # CALENDAR VISUALS: draw the month grid. Days with saved entries show a check mark;
    # clicking a day selects it for viewing/editing in the daily form below.
    month_date = selected_month.replace(day=1)
    cal = pycalendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(month_date.year, month_date.month)
    st.markdown(f"### {month_date.strftime('%B %Y')}")
    weekday_cols = st.columns(7)
    for col, label in zip(weekday_cols, ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        col.markdown(f"**{label}**")
    for week_idx, week in enumerate(weeks):
        cols = st.columns(7)
        for col, day_num in zip(cols, week):
            with col:
                if day_num == 0:
                    st.write(" ")
                else:
                    iso = date(month_date.year, month_date.month, day_num).isoformat()
                    mark = " ✓" if iso in prom_entries else ""
                    if st.button(
                        f"{day_num}{mark}",
                        key=f"prom_day_{iso}",
                        use_container_width=True,
                        type="primary" if iso == st.session_state["prom_selected_date"] else "secondary",
                    ):
                        st.session_state["prom_selected_date"] = iso
                        st.rerun()

    # SELECTED-DAY AREA: reconcile the selected day with the visible month, then load its entry.
    selected_iso = st.session_state["prom_selected_date"]
    # Keep selection aligned with the month being viewed if the month picker changes.
    if (date.fromisoformat(selected_iso).year, date.fromisoformat(selected_iso).month) != (month_date.year, month_date.month):
        selected_iso = month_date.isoformat()
        st.session_state["prom_selected_date"] = selected_iso
    selected_day = date.fromisoformat(selected_iso)
    existing = prom_entries.get(selected_iso, {})
    st.divider()
    st.markdown(f"### {selected_day.strftime('%A, %B')} {selected_day.day}, {selected_day.year}")
    st.caption("A tiny snapshot of your day. Choose the options that feel closest to your experience.")

    # DAILY CHECK-IN FORM AREA: edit the selected date. Add/remove fields inside this form.
    with st.form("prom_daily_form", clear_on_submit=False):
        # FORM SECTION A: symptoms and pain details.
        st.markdown('<div class="prom-section">Symptoms</div>', unsafe_allow_html=True)
        st.caption("Tap a number — 0 means none, and 10 means very severe.")
        c1, c2, c3 = st.columns(3)
        with c1:
            pain = st.radio("Pain", options=list(range(11)), index=max(0, min(10, int(existing.get("pain", 0)))), horizontal=True, key=f"prom_pain_{selected_iso}", label_visibility="visible")
        with c2:
            fatigue = st.radio("🪫 Fatigue", options=list(range(11)), index=max(0, min(10, int(existing.get("fatigue", 0)))), horizontal=True, key=f"prom_fatigue_{selected_iso}", label_visibility="visible")
        with c3:
            spasms = st.radio("〰️ Muscle spasms", options=list(range(11)), index=max(0, min(10, int(existing.get("spasms", 0)))), horizontal=True, key=f"prom_spasms_{selected_iso}", label_visibility="visible")
        pain_location = st.multiselect(
            "Pain location (optional)",
            ["Neck", "Shoulder/arm", "Back", "Chest/torso", "Hip", "Leg/foot", "Other"],
            default=existing.get("pain_location", []),
        )
        pain_type = st.selectbox(
            "Pain type (optional)",
            ["Not specified", "Burning", "Aching", "Sharp", "Electric/shooting", "Other"],
            index=["Not specified", "Burning", "Aching", "Sharp", "Electric/shooting", "Other"].index(
                existing.get("pain_type", "Not specified")
                if existing.get("pain_type", "Not specified") in ["Not specified", "Burning", "Aching", "Sharp", "Electric/shooting", "Other"]
                else "Not specified"
            ),
        )

        # FORM SECTION B: sleep duration/quality, mood and overall wellbeing.
        st.markdown('<div class="prom-section">Sleep & wellbeing</div>', unsafe_allow_html=True)
        quality_labels = {1: "1 · Very poor", 2: "2 · Poor", 3: "3 · Fair", 4: "4 · Good", 5: "5 · Very good"}
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep_quality = st.radio("Sleep quality", options=[1, 2, 3, 4, 5], index=max(0, min(4, int(existing.get("sleep_quality", 3))-1)), format_func=lambda x: quality_labels[x], horizontal=True, key=f"prom_sleep_{selected_iso}")
            sleep_hours = st.number_input("Hours slept (optional)", 0.0, 24.0, float(existing.get("sleep_hours", 8.0)), 0.5, key=f"prom_sleep_hours_{selected_iso}")
        with c2:
            mood = st.radio("Your mood", options=[1, 2, 3, 4, 5], index=max(0, min(4, int(existing.get("mood", 3))-1)), format_func=lambda x: quality_labels[x], horizontal=True, key=f"prom_mood_{selected_iso}")
        with c3:
            overall_wellbeing = st.radio("Overall wellbeing", options=[1, 2, 3, 4, 5], index=max(0, min(4, int(existing.get("overall_wellbeing", 3))-1)), format_func=lambda x: quality_labels[x], horizontal=True, key=f"prom_wellbeing_{selected_iso}")

        # FORM SECTION C: medication status, rehabilitation and daily health routines.
        st.markdown('<div class="prom-section">Medication & routines</div>', unsafe_allow_html=True)
        medication_options = ["Taken as prescribed", "Missed a dose", "Not applicable", "Prefer not to say"]
        medication_default = existing.get("medication", "Not applicable")
        if medication_default not in medication_options:
            medication_default = "Not applicable"
        medication = st.radio("Medication today", medication_options, index=medication_options.index(medication_default), horizontal=True)
        rehab_activity = st.selectbox(
            "Rehabilitation / planned activity",
            ["Not recorded", "Completed", "Partly completed", "Not completed", "Rest day"],
            index=["Not recorded", "Completed", "Partly completed", "Not completed", "Rest day"].index(
                existing.get("rehab_activity", "Not recorded")
                if existing.get("rehab_activity", "Not recorded") in ["Not recorded", "Completed", "Partly completed", "Not completed", "Rest day"]
                else "Not recorded"
            ),
        )
        c1, c2 = st.columns(2)
        with c1:
            bowel_routine = st.selectbox(
                "Bowel routine (optional)",
                ["Not applicable", "As planned", "Different from usual", "Not recorded", "Prefer not to say"],
                index=["Not applicable", "As planned", "Different from usual", "Not recorded", "Prefer not to say"].index(
                    existing.get("bowel_routine", "Not applicable")
                    if existing.get("bowel_routine", "Not applicable") in ["Not applicable", "As planned", "Different from usual", "Not recorded", "Prefer not to say"]
                    else "Not applicable"
                ),
            )
        with c2:
            bladder_routine = st.selectbox(
                "Bladder routine (optional)",
                ["Not applicable", "As planned", "Different from usual", "Not recorded", "Prefer not to say"],
                index=["Not applicable", "As planned", "Different from usual", "Not recorded", "Prefer not to say"].index(
                    existing.get("bladder_routine", "Not applicable")
                    if existing.get("bladder_routine", "Not applicable") in ["Not applicable", "As planned", "Different from usual", "Not recorded", "Prefer not to say"]
                    else "Not applicable"
                ),
            )
        c1, c2 = st.columns(2)
        with c1:
            skin_check = st.selectbox(
                "Skin check (optional)",
                ["Not recorded", "Completed", "Concern noticed", "Not applicable", "Prefer not to say"],
                index=["Not recorded", "Completed", "Concern noticed", "Not applicable", "Prefer not to say"].index(
                    existing.get("skin_check", "Not recorded")
                    if existing.get("skin_check", "Not recorded") in ["Not recorded", "Completed", "Concern noticed", "Not applicable", "Prefer not to say"]
                    else "Not recorded"
                ),
            )
        with c2:
            hydration = st.selectbox(
                "Hydration (optional)",
                ["Not recorded", "As planned", "Less than planned", "More than planned", "Prefer not to say"],
                index=["Not recorded", "As planned", "Less than planned", "More than planned", "Prefer not to say"].index(
                    existing.get("hydration", "Not recorded")
                    if existing.get("hydration", "Not recorded") in ["Not recorded", "As planned", "Less than planned", "More than planned", "Prefer not to say"]
                    else "Not recorded"
                ),
            )

        # FORM SECTION D: daily activities and level of assistance/function.
        st.markdown('<div class="prom-section">Daily activities & function</div>', unsafe_allow_html=True)
        st.caption("Rate how easy these activities felt today. Skip anything that is not relevant to you.")
        activity_names = [
            "Transfers", "Wheelchair mobility / getting around", "Dressing", "Bathing / personal care",
            "Preparing food", "Joining usual activities",
        ]
        activity_levels = ["Not relevant", "Unable today", "A lot of help", "Some help", "A little help", "Independent / usual"]
        old_activities = existing.get("activities", {})
        activities = {}
        act_cols = st.columns(2)
        for i, activity in enumerate(activity_names):
            prior = old_activities.get(activity, "Not relevant")
            if prior not in activity_levels:
                prior = "Not relevant"
            with act_cols[i % 2]:
                activities[activity] = st.selectbox(
                    activity, activity_levels, index=activity_levels.index(prior), key=f"prom_activity_{selected_iso}_{i}"
                )

        notes = st.text_area("Anything else you want to remember? (optional)", value=existing.get("notes", ""), max_chars=2000)
        submitted = st.form_submit_button("Save check-in", type="primary", use_container_width=True)

    # SAVE AREA: collect the form fields into one entry and write it to local JSON.
    if submitted:
        prom_entries[selected_iso] = {
            "pain": pain, "fatigue": fatigue, "spasms": spasms,
            "pain_location": pain_location, "pain_type": pain_type,
            "sleep_quality": sleep_quality, "sleep_hours": sleep_hours,
            "mood": mood, "overall_wellbeing": overall_wellbeing,
            "medication": medication, "rehab_activity": rehab_activity,
            "bowel_routine": bowel_routine, "bladder_routine": bladder_routine,
            "skin_check": skin_check, "hydration": hydration,
            "activities": activities, "notes": notes,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_prom_entries(prom_entries)
        st.success("Check-in saved successfully.")
        st.rerun()

    # DELETE-ONE-DAY AREA: remove only the selected day's saved entry.
    if selected_iso in prom_entries:
        if st.button("🗑️ Delete this day's check-in", key="prom_delete_day"):
            prom_entries.pop(selected_iso, None)
            save_prom_entries(prom_entries)
            st.success("Daily entry deleted.")
            st.rerun()

    # INSIGHTS / CHARTS AREA: summaries and visual trends for the last 7 or 30 days.
    # Add new averages, charts or pattern summaries in this section.
    st.divider()
    st.markdown("## Patterns & progress")
    st.caption("Summaries use recorded values only. They describe patterns, not causes or diagnoses.")
    period_choice = st.radio("Summary period", ["Last 7 days", "Last 30 days"], horizontal=True, key="prom_period")
    period_days = 7 if period_choice == "Last 7 days" else 30
    end_day = date.fromisoformat(selected_iso)
    period_dates = [(end_day - timedelta(days=i)).isoformat() for i in range(period_days - 1, -1, -1)]
    recorded_dates = [d for d in period_dates if d in prom_entries]
    if not recorded_dates:
        st.info("Save a few daily check-ins to see averages and trends here.")
    else:
        avg_cols = st.columns(4)
        for col, key, label, suffix in [
            (avg_cols[0], "pain", "Average pain", "/10"),
            (avg_cols[1], "fatigue", "Average fatigue", "/10"),
            (avg_cols[2], "spasms", "Average spasms", "/10"),
            (avg_cols[3], "overall_wellbeing", "Wellbeing", "/5"),
        ]:
            avg, n = prom_average(prom_entries, key, recorded_dates)
            col.metric(label, f"{avg:.1f}{suffix}" if avg is not None else "—", help=f"Based on {n} recorded day(s).")
        st.caption(f"{len(recorded_dates)} of {period_days} days have entries in this period.")

        trend_rows = []
        for d in recorded_dates:
            item = prom_entries[d]
            trend_rows.append({
                "Date": date.fromisoformat(d),
                "Pain": item.get("pain"),
                "Fatigue": item.get("fatigue"),
                "Spasms": item.get("spasms"),
            })
        trend_df = pd.DataFrame(trend_rows).sort_values("Date")
        long_df = trend_df.melt(id_vars="Date", var_name="Measure", value_name="Rating")
        st.altair_chart(
            themed(
                alt.Chart(long_df).mark_line(point=True).encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Rating:Q", title="Rating (0–10)", scale=alt.Scale(domain=[0, 10])),
                    color=alt.Color("Measure:N"),
                    tooltip=["Date:T", "Measure:N", "Rating:Q"],
                ).properties(height=260)
            ),
            use_container_width=True,
        )

        # Cautious, descriptive patterns; require at least 3 entries in each group.
        pain_sleep = [
            (prom_entries[d].get("pain"), prom_entries[d].get("sleep_quality"))
            for d in recorded_dates
            if isinstance(prom_entries[d].get("pain"), (int, float))
            and isinstance(prom_entries[d].get("sleep_quality"), (int, float))
        ]
        if len(pain_sleep) >= 4:
            poor_sleep_pain = [p for p, s in pain_sleep if s <= 2]
            better_sleep_pain = [p for p, s in pain_sleep if s >= 4]
            if len(poor_sleep_pain) >= 2 and len(better_sleep_pain) >= 2:
                a = sum(poor_sleep_pain) / len(poor_sleep_pain)
                b = sum(better_sleep_pain) / len(better_sleep_pain)
                if abs(a - b) >= 1:
                    st.markdown(
                        f"**Pattern to explore:** average pain was {a:.1f}/10 on entries with poor sleep "
                        f"and {b:.1f}/10 on entries with good sleep. This is an association in your log, "
                        "not evidence that sleep caused the difference."
                    )
        medication_logged = [prom_entries[d].get("medication") for d in recorded_dates]
        medication_known = [v for v in medication_logged if v in ["Taken as prescribed", "Missed a dose"]]
        if medication_known:
            adherence = sum(v == "Taken as prescribed" for v in medication_known) / len(medication_known) * 100
            st.markdown(
                f"**Medication log:** marked “taken as prescribed” on {adherence:.0f}% of days where a dose status was recorded "
                f"({len(medication_known)} day(s)). This is a log summary, not a judgement or medication advice."
            )
        function_scores = []
        for d in recorded_dates:
            for activity, value in prom_entries[d].get("activities", {}).items():
                if value in ["Unable today", "A lot of help", "Some help", "A little help", "Independent / usual"]:
                    function_scores.append((activity, value))
        if function_scores:
            st.markdown("**Everyday function logged:**")
            func_df = pd.DataFrame(function_scores, columns=["Activity", "Reported ease"])
            st.dataframe(func_df.groupby(["Activity", "Reported ease"]).size().rename("Entries").reset_index(),
                         hide_index=True, use_container_width=True)

    # EXPORT / DELETE AREA: CSV/JSON downloads and delete-all control.
    # Add additional export formats here if needed.
    st.divider()
    st.markdown("## 🎀 Your diary data")
    if prom_entries:
        export_df = pd.DataFrame([
            {"date": d, **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                           for k, v in item.items()}}
            for d, item in sorted(prom_entries.items())
        ])
        st.download_button(
            "📄 Download wellbeing data (CSV)",
            export_df.to_csv(index=False),
            file_name="spymotion_wellbeing.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "🧸 Download wellbeing data (JSON)",
            json.dumps(prom_entries, ensure_ascii=False, indent=2),
            file_name="spymotion_wellbeing.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("Delete all wellbeing entries", key="prom_delete_all"):
            store.delete_prom()
            st.session_state.pop("prom_selected_date", None)
            st.success("All wellbeing entries deleted from this device.")
            st.rerun()
    else:
        st.caption("No wellbeing entries have been saved yet.")

    st.caption(
        f"Storage file: {store.prom_path()}. This prototype saves locally and does not sync with a clinician or cloud. "
        "The file is not encrypted by this feature; use a protected device and do not use real patient data in an unapproved prototype."
    )
    html(
        '<div class="safety" role="note"><b>Health and safety.</b> This is a prototype wellbeing diary, not a diagnostic tool '
        'or clinical PROM validated for SCI. Do not use it to change medication or treatment. New, severe or concerning symptoms '
        'should be discussed with an appropriate healthcare professional; seek urgent help for emergencies.</div>'
    )

# ======================================================================
# MEDICATION & ROUTINES TAB
# Dedicated shortcut for daily medication and routine tracking.
# It saves into the same local PROM JSON entry for the selected date
# and preserves other saved wellbeing fields.
# ======================================================================

if show_medication:
    with tab_medication:

        st.markdown(
            '<div class="prom-banner">'
            '<div class="eyebrow">DAILY TRACKING</div>'
            '<h2>Medication & routines</h2>'
            '<p>A quick place to log medication, daily routines, sleep and hydration.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="prom-note">'
            '<b>Optional by design.</b> '
            'Skip anything that is not relevant to you.'
            '</div>',
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------------
        # DATE
        # Choose which day's medication/routine entry to edit.
        # --------------------------------------------------------------

        medication_entries = load_prom_entries()

        medication_date_default = date.fromisoformat(
            st.session_state.get(
                "prom_selected_date",
                date.today().isoformat(),
            )
        )

        medication_day = st.date_input(
            "Entry date",
            value=medication_date_default,
            key="prom_medication_tab_date",
        )

        medication_iso = medication_day.isoformat()

        medication_existing = medication_entries.get(
            medication_iso,
            {},
        )

        # --------------------------------------------------------------
        # MEDICATION
        # --------------------------------------------------------------

        st.markdown(
            '<div class="prom-section">Medication</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Record what happened today. "
            "This is a diary entry, not medication advice."
        )

        medication_options = [
            "Taken as prescribed",
            "Missed a dose",
            "Not applicable",
            "Prefer not to say",
        ]

        saved_medication = medication_existing.get(
            "medication",
            "Not applicable",
        )

        if saved_medication not in medication_options:
            saved_medication = "Not applicable"

        medication_status = st.radio(
            "Medication taken as prescribed?",
            medication_options,
            index=medication_options.index(saved_medication),
            format_func=lambda value: {
                "Taken as prescribed": "Yes — taken as prescribed",
                "Missed a dose": "No — missed a dose",
                "Not applicable": "Not applicable",
                "Prefer not to say": "Prefer not to say",
            }[value],
            horizontal=True,
            key=f"prom_medication_tab_status_{medication_iso}",
        )

        missed_reason = st.text_input(
            "Missed-dose reason (optional)",
            value=medication_existing.get(
                "medication_missed_reason",
                "",
            ),
            max_chars=300,
            placeholder="e.g. forgot, ran out, side effects, timing",
            key=f"prom_medication_tab_reason_{medication_iso}",
        )

        # --------------------------------------------------------------
        # DAILY ROUTINES
        # --------------------------------------------------------------

        st.markdown(
            '<div class="prom-section">Daily routines</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Only record routines that are relevant to you."
        )

        rehab_options = [
            "Not recorded",
            "Completed",
            "Partly completed",
            "Not completed",
            "Rest day",
        ]

        saved_rehab = medication_existing.get(
            "rehab_activity",
            "Not recorded",
        )

        if saved_rehab not in rehab_options:
            saved_rehab = "Not recorded"

        rehab_activity_tab = st.selectbox(
            "Rehabilitation / planned activity",
            rehab_options,
            index=rehab_options.index(saved_rehab),
            key=f"prom_medication_tab_rehab_{medication_iso}",
        )

        routine_options = [
            "Not applicable",
            "As planned",
            "Different from usual",
            "Not recorded",
            "Prefer not to say",
        ]

        routine_cols = st.columns(2)

        with routine_cols[0]:

            saved_bowel = medication_existing.get(
                "bowel_routine",
                "Not applicable",
            )

            if saved_bowel not in routine_options:
                saved_bowel = "Not applicable"

            bowel_routine_tab = st.selectbox(
                "Bowel routine (optional)",
                routine_options,
                index=routine_options.index(saved_bowel),
                key=f"prom_medication_tab_bowel_{medication_iso}",
            )

        with routine_cols[1]:

            saved_bladder = medication_existing.get(
                "bladder_routine",
                "Not applicable",
            )

            if saved_bladder not in routine_options:
                saved_bladder = "Not applicable"

            bladder_routine_tab = st.selectbox(
                "Bladder routine (optional)",
                routine_options,
                index=routine_options.index(saved_bladder),
                key=f"prom_medication_tab_bladder_{medication_iso}",
            )

        skin_options = [
            "Not recorded",
            "Completed",
            "Concern noticed",
            "Not applicable",
            "Prefer not to say",
        ]

        saved_skin = medication_existing.get(
            "skin_check",
            "Not recorded",
        )

        if saved_skin not in skin_options:
            saved_skin = "Not recorded"

        routine_cols2 = st.columns(2)

        with routine_cols2[0]:

            skin_check_tab = st.selectbox(
                "Skin check (optional)",
                skin_options,
                index=skin_options.index(saved_skin),
                key=f"prom_medication_tab_skin_{medication_iso}",
            )

        with routine_cols2[1]:

            hydration_options = [
                "Not recorded",
                "As planned",
                "Less than planned",
                "More than planned",
                "Prefer not to say",
            ]

            saved_hydration = medication_existing.get(
                "hydration",
                "Not recorded",
            )

            if saved_hydration not in hydration_options:
                saved_hydration = "Not recorded"

            hydration_tab = st.selectbox(
                "Hydration (optional)",
                hydration_options,
                index=hydration_options.index(saved_hydration),
                key=f"prom_medication_tab_hydration_{medication_iso}",
            )

        # --------------------------------------------------------------
        # SLEEP
        # --------------------------------------------------------------

        st.markdown(
            '<div class="prom-section">Sleep</div>',
            unsafe_allow_html=True,
        )

        sleep_hours_tab = st.number_input(
            "Hours slept (optional)",
            min_value=0.0,
            max_value=24.0,
            value=float(
                medication_existing.get(
                    "sleep_hours",
                    8.0,
                )
            ),
            step=0.5,
            key=f"prom_medication_tab_sleep_{medication_iso}",
        )

        # --------------------------------------------------------------
        # SAVE
        # Merge with the existing wellbeing entry instead of replacing it.
        # --------------------------------------------------------------

        if st.button(
            "Save medication & routines",
            type="primary",
            use_container_width=True,
            key="prom_medication_tab_save",
        ):

            merged_entry = dict(
                medication_entries.get(
                    medication_iso,
                    {},
                )
            )

            merged_entry.update({
                "medication": medication_status,
                "medication_missed_reason": missed_reason,
                "rehab_activity": rehab_activity_tab,
                "bowel_routine": bowel_routine_tab,
                "bladder_routine": bladder_routine_tab,
                "skin_check": skin_check_tab,
                "hydration": hydration_tab,
                "sleep_hours": sleep_hours_tab,
                "updated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            })

            medication_entries[medication_iso] = merged_entry

            save_prom_entries(
                medication_entries
            )

            st.session_state[
                "prom_selected_date"
            ] = medication_iso

            st.success(
                f"Medication and routines saved for "
                f"{medication_day.strftime('%d %B %Y')}."
            )

            st.rerun()

        st.caption(
            f"Entries are saved locally in {store.prom_path().name}. "
            "The file is not encrypted; avoid entering identifiable "
            "patient information in this prototype."
        )