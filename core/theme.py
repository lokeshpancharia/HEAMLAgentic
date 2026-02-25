"""Shared visual theme for HEAMLAgentic — inject with inject_theme()."""

import streamlit as st


def inject_theme():
    """Inject global CSS: animated HEA crystal + ML neural background."""
    st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ────────────────────────────────────────────────── */
:root {
    --bg-deep:      #050c1a;
    --bg-mid:       #081428;
    --bg-card:      rgba(10, 25, 55, 0.85);
    --accent-blue:  #1e90ff;
    --accent-teal:  #00d4aa;
    --accent-gold:  #f5a623;
    --accent-purple:#a855f7;
    --text-primary: #e8f4fd;
    --text-muted:   #7a9cc0;
    --border:       rgba(30, 144, 255, 0.18);
    --glow-blue:    rgba(30, 144, 255, 0.35);
    --glow-teal:    rgba(0, 212, 170, 0.25);
}

/* ── Animated SVG canvas background ─────────────────────────────── */
.stApp {
    background-color: var(--bg-deep);
    background-image:
        /* Neural network nodes layer */
        radial-gradient(circle at 15% 20%, rgba(30,144,255,0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 75%, rgba(0,212,170,0.07) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(168,85,247,0.05) 0%, transparent 60%),
        radial-gradient(circle at 80% 15%, rgba(245,166,35,0.06) 0%, transparent 30%),
        radial-gradient(circle at 20% 85%, rgba(30,144,255,0.06) 0%, transparent 35%);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Animated lattice grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(30,144,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,144,255,0.04) 1px, transparent 1px),
        linear-gradient(rgba(0,212,170,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.025) 1px, transparent 1px);
    background-size: 80px 80px, 80px 80px, 20px 20px, 20px 20px;
    animation: lattice-drift 30s linear infinite;
}

@keyframes lattice-drift {
    0%   { background-position: 0 0, 0 0, 0 0, 0 0; }
    100% { background-position: 80px 80px, 80px 80px, 20px 20px, 20px 20px; }
}

/* Floating atom orbs */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(circle 3px at 12% 30%, var(--accent-teal) 0%, transparent 100%),
        radial-gradient(circle 2px at 88% 20%, var(--accent-blue) 0%, transparent 100%),
        radial-gradient(circle 4px at 45% 80%, var(--accent-gold) 0%, transparent 100%),
        radial-gradient(circle 2px at 70% 60%, var(--accent-purple) 0%, transparent 100%),
        radial-gradient(circle 3px at 25% 70%, var(--accent-blue) 0%, transparent 100%),
        radial-gradient(circle 2px at 60% 15%, var(--accent-teal) 0%, transparent 100%),
        radial-gradient(circle 3px at 90% 85%, var(--accent-gold) 0%, transparent 100%);
    animation: float-atoms 20s ease-in-out infinite alternate;
    opacity: 0.7;
}

@keyframes float-atoms {
    0%   { transform: translate(0px, 0px) rotate(0deg); }
    33%  { transform: translate(15px, -20px) rotate(120deg); }
    66%  { transform: translate(-10px, 15px) rotate(240deg); }
    100% { transform: translate(8px, -8px) rotate(360deg); }
}

/* ── Main content wrapper ────────────────────────────────────────── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050f22 0%, #08182e 60%, #050c1a 100%);
    border-right: 1px solid var(--border);
    position: relative;
    z-index: 10;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal), var(--accent-gold));
    animation: shimmer 3s linear infinite;
    background-size: 200% 100%;
}

@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    transition: all 0.2s;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(30,144,255,0.12) !important;
    border-left: 3px solid var(--accent-blue);
}

/* ── Typography ──────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}

h1 {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-teal) 50%, var(--accent-gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem !important;
}

h2 {
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

h3 { color: var(--accent-teal) !important; }

p, li, label { color: var(--text-primary) !important; }
.stCaption, small { color: var(--text-muted) !important; }

/* ── Cards / containers ──────────────────────────────────────────── */
[data-testid="stVerticalBlock"] > div > div > [data-testid="stVerticalBlock"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ── Metrics ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(10,25,55,0.9) 0%, rgba(8,20,40,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3), 0 0 0 1px rgba(30,144,255,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(30,144,255,0.2);
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: var(--accent-teal) !important; font-weight: 700 !important; font-size: 1.6rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-blue) 0%, #1265cc 100%);
    border: none;
    border-radius: 8px;
    color: white !important;
    font-weight: 600;
    padding: 0.55rem 1.5rem;
    box-shadow: 0 4px 15px rgba(30,144,255,0.4);
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(30,144,255,0.55);
}
.stButton > button[kind="primary"]::after {
    content: '';
    position: absolute;
    top: -50%; left: -75%;
    width: 50%; height: 200%;
    background: rgba(255,255,255,0.15);
    transform: skewX(-20deg);
    transition: left 0.4s;
}
.stButton > button[kind="primary"]:hover::after { left: 150%; }

.stButton > button[kind="secondary"] {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-muted) !important;
    transition: all 0.2s;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-blue);
    color: var(--accent-blue) !important;
    background: rgba(30,144,255,0.08);
}

/* ── Inputs ──────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {
    background: rgba(8, 20, 45, 0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(30,144,255,0.15) !important;
    outline: none !important;
}

/* ── File uploader ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(8,20,45,0.6);
    border: 2px dashed rgba(30,144,255,0.3) !important;
    border-radius: 12px;
    transition: all 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
    background: rgba(30,144,255,0.06);
}

/* ── Tabs ────────────────────────────────────────────────────────── */
[data-testid="stTabs"] > div > div[role="tablist"] {
    border-bottom: 1px solid var(--border);
    gap: 0.25rem;
}
button[role="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--text-muted) !important;
    font-weight: 500;
    transition: all 0.2s;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
button[role="tab"]:hover {
    color: var(--accent-blue) !important;
    background: rgba(30,144,255,0.08) !important;
}
button[role="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    background: rgba(30,144,255,0.1) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Dataframes / tables ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] table thead {
    background: rgba(30,144,255,0.12) !important;
}
[data-testid="stDataFrame"] table thead th {
    color: var(--accent-blue) !important;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stDataFrame"] table tbody tr:nth-child(even) {
    background: rgba(30,144,255,0.04) !important;
}
[data-testid="stDataFrame"] table tbody tr:hover {
    background: rgba(0,212,170,0.06) !important;
}

/* ── Progress / spinners ─────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal)) !important;
    border-radius: 999px;
}

/* ── Alerts / info boxes ─────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    backdrop-filter: blur(6px);
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
    background: rgba(30,144,255,0.1) !important;
    border-color: var(--accent-blue) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
    background: rgba(0,212,170,0.1) !important;
    border-color: var(--accent-teal) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
    background: rgba(245,166,35,0.1) !important;
    border-color: var(--accent-gold) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
    background: rgba(239,68,68,0.1) !important;
    border-color: #ef4444 !important;
}

/* ── Dividers ────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── Code blocks ─────────────────────────────────────────────────── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(8,20,45,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--accent-teal) !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: rgba(8,20,45,0.6) !important;
    backdrop-filter: blur(6px);
}
[data-testid="stExpander"]:hover {
    border-color: var(--accent-blue) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(30,144,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Page hero banner ────────────────────────────────────────────── */
.hea-hero {
    background: linear-gradient(135deg,
        rgba(30,144,255,0.12) 0%,
        rgba(0,212,170,0.08) 50%,
        rgba(168,85,247,0.06) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hea-hero::before {
    content: '';
    position: absolute;
    top: -60%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(30,144,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: pulse-orb 4s ease-in-out infinite;
}
.hea-hero::after {
    content: '';
    position: absolute;
    bottom: -40%; left: 5%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: pulse-orb 6s ease-in-out infinite reverse;
}
@keyframes pulse-orb {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50%       { transform: scale(1.3); opacity: 1; }
}

.hea-hero h1 { margin: 0 !important; font-size: 2.4rem !important; }
.hea-hero p  { color: var(--text-muted) !important; margin: 0.5rem 0 0; font-size: 1rem; }

/* ── Status badge ────────────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-ready   { background: rgba(0,212,170,0.15); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.badge-pending { background: rgba(245,166,35,0.12); color: #f5a623; border: 1px solid rgba(245,166,35,0.3); }

/* ── Agent log box ───────────────────────────────────────────────── */
.agent-log {
    background: rgba(5,12,26,0.95);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent-teal);
    max-height: 300px;
    overflow-y: auto;
    line-height: 1.7;
}

/* ── Plotly chart backgrounds ────────────────────────────────────── */
.js-plotly-plot .plotly .bg {
    fill: rgba(8,20,45,0.01) !important;
}

/* ── Step pills in sidebar ───────────────────────────────────────── */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
    transition: background 0.15s;
}
.step-item:hover { background: rgba(30,144,255,0.08); }
</style>
""", unsafe_allow_html=True)


def page_hero(title: str, subtitle: str, icon: str = ""):
    """Render a styled hero banner at the top of each page."""
    st.markdown(f"""
<div class="hea-hero">
    <h1>{icon} {title}</h1>
    <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)
