"""Dashboard look: tesser.sh layout and palette (pure black, hairline bento cells, light
sentence-case headings, a centred 1200 px column, mono numbers, one pale-lime accent),
with buttermax-style confidence reserved for the hero render and headline numbers.

Colours come from .streamlit/config.toml (UI) and src.tiering.TIER_COLORS (data).
The ACCENT is for interface chrome only (index numbers, active pill, nav underline)
and never encodes a tier.

Layout helpers (all return HTML for st.markdown(..., unsafe_allow_html=True)):
    heading("01", "Operations queue", lede)  section heading: mono index, light title, muted lede
    section("01", "Operations queue")        the same without a lede (older name)
    cells([html, html, html])                static hairline bento grid (equal columns)
    cell("Title", "Body", inner_html)        content of one bento cell
    stat("394", "Critical", "CRITICAL")      mono headline number over a small label
    steps([("Title", "Body"), ...], active)  tesser numbered rows (01 …) on hairlines
    hero_stat("394", "Critical", "CRITICAL") giant number over an uppercase label
    label("Time to TCA")                     small uppercase label
    card(inner_html)                         one bordered cell
    pill("CRITICAL")                         tier as a coloured text pill
Streamlit blocks as bento cells: wrap st.columns in st.container(key="cells-<name>"); each
column becomes a hairline cell. Stacked cell rows join into one grid.
"""
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.tiering import RISK_FLOOR, TIERS, TIER_COLORS, format_prob

BG, SURFACE, LINE = "#000000", "#181817", "#262626"
TEXT, MUTED, ACCENT = "#F2F2F0", "#8E8E88", "#ECF0BD"
ACCENT_DIM = "#959783"
GRID = "rgba(242,242,240,0.06)"
FONT = "'Geist', 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
.stApp {{ background: {BG}; }}
html, body, .stApp, [data-testid="stMarkdownContainer"] {{ font-family: {FONT}; color: {TEXT};
  font-feature-settings: "tnum" 1; -webkit-font-smoothing: antialiased; }}
h1, h2, h3 {{ font-family: {FONT}; font-weight: 400; letter-spacing: -0.03em; }}

/* ---- page: centred 1200 px column, no Streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"], .stDeployButton {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; min-height: 0; }}
.block-container {{ max-width: 1248px !important; padding: 1.2rem 1.5rem 8rem !important; }}
[data-testid="stSidebar"] {{ background: {BG}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] label p {{ font: 500 0.7rem {MONO}; letter-spacing: 0.06em; text-transform: uppercase;
  color: {MUTED}; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED}; font-size: 0.85rem; line-height: 1.55; }}
[data-testid="stMarkdownContainer"] p {{ line-height: 1.6; }}

/* small uppercase labels (and the old eyebrow class) */
.bm-label, .hero-eyebrow {{ font: 500 0.7rem {MONO}; letter-spacing: 0.06em; text-transform: uppercase;
  color: {MUTED}; }}
.hero-sub {{ color: {MUTED}; max-width: 40rem; font-size: 1.05rem; margin-top: 0.4rem; }}

/* ---- section heading: tesser light title, mono index, generous space above ---- */
.bm-section {{ display: flex; align-items: baseline; gap: 1.1rem; margin: 6.5rem 0 0.9rem; }}
.bm-section .idx {{ font: 400 0.85rem {MONO}; color: {ACCENT}; }}
.bm-section .ttl {{ font: 300 clamp(2rem, 3.4vw, 2.75rem) {FONT}; letter-spacing: -0.035em; line-height: 1.1;
  color: {TEXT}; }}
[data-testid="stTabPanel"] > div > div > div:first-child .bm-section {{ margin-top: 3.2rem; }}
.bm-lede {{ font: 400 1.05rem/1.6 {FONT}; color: {MUTED}; max-width: 42rem; margin: 0 0 2.4rem; }}
.bm-lede b {{ color: {TEXT}; font-weight: 500; }}

/* ---- bento: hairline cells, shared borders, no radius, no fill ---- */
.tx-grid {{ display: grid; gap: 1px; background: {LINE}; border: 1px solid {LINE}; margin: 0 0 2.4rem; }}
.tx-grid > .tx-cell {{ background: {BG}; padding: 2.2rem 2.4rem; min-width: 0; }}
.tx-cell .t, .tx-cell-in .t {{ font: 400 1.3rem {FONT}; letter-spacing: -0.02em; color: {TEXT}; margin: 0 0 0.7rem; }}
.tx-cell p, .tx-cell-in p {{ color: {MUTED}; font-size: 0.95rem; line-height: 1.6; margin: 0; }}
.tx-cell p b, .tx-cell-in p b {{ color: {TEXT}; font-weight: 500; }}
div[class*="st-key-cells"] {{ border: 1px solid {LINE}; margin-top: -1px; gap: 0 !important; }}
div[class*="st-key-cells"] > div[data-testid="stHorizontalBlock"] {{ gap: 0 !important; }}
div[class*="st-key-cells"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
  padding: 2rem 2.2rem; border-right: 1px solid {LINE}; min-width: 0; }}
div[class*="st-key-cells"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
  border-right: none; }}
div[class*="st-key-cells"] + div[class*="st-key-cells"] {{ border-top: none; margin-top: 0; }}
div[class*="st-key-cells"] [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
  padding: 0; border: none; }}

/* mono headline numbers inside cells */
.tx-stat {{ display: flex; flex-direction: column; gap: 0.55rem; }}
.tx-stat .v {{ font: 400 clamp(2rem, 3.4vw, 3rem) {FONT}; letter-spacing: -0.045em; line-height: 1; color: {TEXT};
  white-space: nowrap; }}
.tx-stat .c {{ font: 500 0.7rem {MONO}; letter-spacing: 0.06em; text-transform: uppercase; color: {MUTED}; }}
.tx-stat .s {{ font-size: 0.85rem; color: {MUTED}; line-height: 1.5; }}

/* numbered rows (tesser '01 Install the CLI') */
.tx-steps {{ border-top: 1px solid {LINE}; margin: 0 0 2.4rem; }}
.tx-step {{ display: grid; grid-template-columns: 3.4rem 1fr; padding: 1.35rem 0 1.35rem 1.6rem;
  border-bottom: 1px solid {LINE}; border-left: 1px solid transparent; }}
.tx-step .n {{ font: 400 0.85rem {MONO}; color: {MUTED}; padding-top: 0.3rem; }}
.tx-step .t {{ font: 400 1.25rem {FONT}; letter-spacing: -0.02em; color: {MUTED}; }}
.tx-step .b {{ grid-column: 2; color: {MUTED}; font-size: 0.93rem; line-height: 1.6; margin-top: 0.6rem; }}
.tx-step.on {{ border-left-color: {ACCENT}; }}
.tx-step.on .t, .tx-step.on .n {{ color: {TEXT}; }}

/* giant numbers (hero) */
.bm-hero {{ display: flex; flex-direction: column; gap: 0.6rem; }}
.bm-hero .num {{ font: 500 clamp(3rem, 7vw, 6.6rem) {FONT}; letter-spacing: -0.06em; line-height: 0.88; }}
.stat-row .bm-hero {{ gap: 0.7rem; }}
.stat-row .bm-hero .num {{ font: 400 clamp(2.2rem, 4vw, 3.6rem) {FONT}; letter-spacing: -0.05em; white-space: nowrap;
  line-height: 1; }}

/* surfaces: one bordered cell; expanders and metrics on hairlines */
.bm-card {{ border: 1px solid {LINE}; background: {BG}; padding: 1.8rem 2rem; }}
[data-testid="stMetric"] {{ background: transparent; padding: 0; }}
[data-testid="stMetricLabel"] p {{ font: 500 0.7rem {MONO} !important; letter-spacing: 0.06em;
  text-transform: uppercase; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-family: {FONT}; font-weight: 400; letter-spacing: -0.045em; }}
[data-testid="stExpander"] details {{ border: 1px solid {LINE} !important; border-radius: 0 !important;
  background: {BG}; }}
[data-testid="stExpander"] summary p {{ font-size: 0.92rem; color: {MUTED}; }}
[data-testid="stExpander"] summary:hover p {{ color: {TEXT}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 0; }}
[data-testid="stAlert"] {{ border-radius: 0; border: 1px solid {LINE}; background: {SURFACE} !important; }}
.pill {{ display: inline-block; padding: 0.12rem 0.65rem; border-radius: 999px; font: 500 0.7rem {MONO};
  letter-spacing: 0.06em; border: 1px solid currentColor; }}

/* ---- top bar ---- */
.bm-top {{ display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0 1.1rem;
  border-bottom: 1px solid {LINE}; }}
.bm-top .mark {{ font: 500 1.05rem {FONT}; letter-spacing: -0.03em; }}
.bm-top .mark i {{ font-style: normal; color: {MUTED}; font-weight: 300; }}
.bm-top .meta {{ display: flex; gap: 1.8rem; font: 400 0.72rem {MONO}; color: {MUTED}; }}
.bm-top .meta b {{ color: {TEXT}; font-weight: 400; }}
.bm-dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {ACCENT};
  margin-right: 0.5rem; vertical-align: 1px; box-shadow: 0 0 12px {ACCENT}; animation: bm-pulse 2.4s ease-in-out infinite; }}
@keyframes bm-pulse {{ 50% {{ opacity: 0.35; }} }}

/* ---- hero: the one place for big confident type ---- */
.bm-display {{ font: 500 clamp(3rem, 6.4vw, 6.2rem) {FONT}; letter-spacing: -0.055em; line-height: 0.92;
  margin: 4.5rem 0 1.6rem; color: {TEXT}; }}
.bm-display em {{ font-style: normal; color: {MUTED}; font-weight: 300; }}
.bm-caption {{ font: 400 0.78rem/1.5 {FONT}; color: {MUTED}; margin: -0.6rem 0 0; max-width: 30rem; }}

/* KPI band: st.metric in bento cells (order matters to the tests) */
.st-key-kpis {{ margin-top: 3.2rem; }}
.st-key-kpis [data-testid="stHorizontalBlock"] {{ gap: 0 !important; flex-wrap: nowrap; border: 1px solid {LINE}; }}
.st-key-kpis [data-testid="stColumn"] {{ min-width: 0; padding: 1.6rem 1.6rem 1.5rem; border-right: 1px solid {LINE}; }}
.st-key-kpis [data-testid="stColumn"]:last-child {{ border-right: none; }}
.st-key-kpis [data-testid="stMetricValue"] {{ font-size: clamp(1.8rem, 3vw, 2.9rem); line-height: 1; white-space: nowrap; }}
.st-key-kpi-critical [data-testid="stMetricValue"], .st-key-kpi-critical [data-testid="stMetricValue"] * {{
  color: {TIER_COLORS["CRITICAL"]} !important; }}
.st-key-kpi-elevated [data-testid="stMetricValue"], .st-key-kpi-elevated [data-testid="stMetricValue"] * {{
  color: {TIER_COLORS["ELEVATED"]} !important; }}
.st-key-kpi-safe [data-testid="stMetricValue"], .st-key-kpi-safe [data-testid="stMetricValue"] * {{
  color: {TIER_COLORS["SAFE"]} !important; }}
.st-key-kpis-queue-depth [data-testid="stMetricValue"] {{ font-size: clamp(3.4rem, 8vw, 7rem); line-height: 0.9; }}

/* ticker of the worst events */
.bm-marquee {{ overflow: hidden; border-bottom: 1px solid {LINE}; margin: 0; padding: 1rem 0; white-space: nowrap;
  mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }}
.bm-marquee .track {{ display: inline-block; animation: bm-scroll 80s linear infinite; }}
.bm-marquee:hover .track {{ animation-play-state: paused; }}
.bm-marquee span {{ font: 400 0.85rem {FONT}; color: {MUTED}; margin-right: 3.4rem; }}
.bm-marquee span b {{ color: {TEXT}; font-weight: 500; }}
@keyframes bm-scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .bm-marquee .track, .bm-dot {{ animation: none; }} }}

/* filter chips: Streamlit 1.58 renders span[data-baseweb="tag"], 1.66 the inner span of
   stMultiSelectTagsContainer. Keep both selectors. */
span[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] > span > span {{
  background-color: {SURFACE} !important; border: 1px solid {LINE} !important; border-radius: 999px !important; }}
span[data-baseweb="tag"] *, [data-testid="stMultiSelectTagsContainer"] > span > span * {{
  color: {TEXT} !important; fill: {TEXT} !important; font-weight: 500; }}

/* ---- styled data tables (html_table): hairlines, mono numbers ---- */
.bm-table-wrap {{ overflow: auto; border: 1px solid {LINE}; scrollbar-width: thin;
  scrollbar-color: {LINE} transparent; }}
.bm-table {{ width: 100%; border-collapse: collapse; font: 400 0.9rem {FONT}; color: {TEXT}; }}
.bm-table thead th {{ position: sticky; top: 0; z-index: 2; background: {BG}; text-align: left;
  font: 500 0.66rem {MONO}; letter-spacing: 0.06em; text-transform: uppercase; color: {MUTED};
  padding: 0.95rem 0.8rem 0.8rem; border-bottom: 1px solid {LINE}; white-space: nowrap; }}
.bm-table td {{ padding: 0.85rem 0.8rem; border-bottom: 1px solid {LINE}; white-space: nowrap; vertical-align: middle; }}
.bm-table tbody tr:last-child td {{ border-bottom: none; }}
.bm-table th, .bm-table td {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
.bm-table tr {{ background: transparent !important; }}
.bm-table tbody tr {{ transition: background 0.12s ease; }}
.bm-table tbody tr:hover {{ background: {SURFACE} !important; }}
.bm-table td.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
.bm-table th.r {{ text-align: right; }}
.bm-table .dim {{ color: {MUTED}; }}
.bm-table .strong {{ font-weight: 500; }}
.bm-table .rank {{ color: #4A4A47; font: 400 0.78rem {MONO}; width: 2.6rem; }}
.bm-table .sub {{ display: block; font-size: 0.72rem; color: {MUTED}; margin-top: 0.15rem; }}
.bm-table td:has(.why) {{ white-space: normal; min-width: 13rem; }}
.bm-table .why {{ display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.35;
  white-space: normal; max-width: 16rem; overflow: hidden; color: {MUTED}; font-size: 0.84rem; cursor: help; }}
.bm-tier {{ display: inline-flex; align-items: center; gap: 0.45rem; font: 500 0.68rem {MONO};
  letter-spacing: 0.06em; padding: 0.26rem 0.6rem 0.26rem 0.5rem; border-radius: 999px; }}
.bm-tier::before {{ content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }}
.bm-bar {{ display: inline-flex; align-items: center; gap: 0.7rem; min-width: 7.5rem; }}
.bm-bar .track {{ flex: 1; height: 2px; background: {LINE}; overflow: hidden; }}
.bm-bar .fill {{ display: block; height: 100%; background: {TEXT}; }}
.bm-bar .val {{ width: 3.6rem; text-align: right; font-variant-numeric: tabular-nums; }}
.bm-chip {{ display: inline-block; font: 500 0.6rem {MONO}; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 0.22rem 0.5rem; border-radius: 999px; border: 1px solid {LINE}; color: {MUTED}; margin-right: 0.3rem; }}
.bm-chip.on {{ color: {TEXT}; border-color: #4A4A47; }}
.bm-table-foot {{ display: flex; justify-content: space-between; padding: 0.8rem 0 0.6rem; font: 400 0.7rem {MONO};
  color: {MUTED}; }}

/* ---- controls: outline pills, accent fill when active ---- */
[data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"] {{
  border-radius: 999px !important; padding: 0.3rem 0.95rem !important; min-height: 0 !important;
  font: 500 0.75rem {FONT} !important; }}
[data-testid="stBaseButton-pills"], [data-testid="stBaseButton-segmented_control"] {{
  background: transparent !important; border: 1px solid {LINE} !important; color: {MUTED} !important; }}
[data-testid="stBaseButton-pills"]:hover, [data-testid="stBaseButton-segmented_control"]:hover {{
  border-color: #4A4A47 !important; color: {TEXT} !important; }}
[data-testid="stBaseButton-pillsActive"], [data-testid="stBaseButton-segmented_controlActive"] {{
  background: {ACCENT} !important; border: 1px solid {ACCENT} !important; }}
[data-testid="stBaseButton-pillsActive"] *, [data-testid="stBaseButton-segmented_controlActive"] * {{
  color: {BG} !important; }}
[data-testid="stButtonGroup"] > div {{ flex-wrap: wrap; row-gap: 0.45rem; gap: 0.45rem; }}
.st-key-queue-filters label p, [data-testid="stWidgetLabel"] p {{ font: 500 0.7rem {MONO} !important;
  letter-spacing: 0.06em; text-transform: uppercase; color: {MUTED} !important; }}
.st-key-queue-filters [data-testid="stCheckbox"] label p {{ text-transform: none; letter-spacing: 0;
  font: 400 0.85rem {FONT} !important; color: {TEXT} !important; white-space: nowrap; overflow: visible !important; }}
input, textarea, [data-baseweb="select"] > div {{ border-radius: 0 !important; }}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{ background: {SURFACE} !important;
  border: 1px solid {LINE} !important; }}

/* buttons: outline pills, accent on hover */
.stButton button, [data-testid="stFormSubmitButton"] button {{ background: {TEXT} !important;
  border: 1px solid {TEXT} !important; border-radius: 999px !important; padding: 0.5rem 1.4rem !important;
  transition: background 0.15s ease, border-color 0.15s ease; }}
.stButton button *, [data-testid="stFormSubmitButton"] button * {{ color: {BG} !important;
  font: 500 0.8rem {FONT} !important; }}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: {ACCENT} !important;
  border-color: {ACCENT} !important; }}

/* sliders: thin white track */
[data-testid="stSlider"] [role="slider"] {{ background: {TEXT}; box-shadow: none; }}

/* ---- nav: tabs as a sticky hairline bar ---- */
[data-testid="stTabs"] [role="tablist"] {{ position: sticky; top: 0; z-index: 50; background: {BG};
  gap: 2.2rem; border-bottom: 1px solid {LINE}; padding-top: 0.6rem; margin-top: 0; }}
[data-testid="stTab"] {{ padding: 1rem 0 !important; background: transparent !important; }}
[data-testid="stTab"] p {{ font: 400 0.82rem {FONT} !important; color: {MUTED}; transition: color 0.15s ease; }}
[data-testid="stTab"]:hover p, [data-testid="stTab"][aria-selected="true"] p {{ color: {TEXT}; }}
[data-testid="stTab"][aria-selected="true"] {{ box-shadow: inset 0 -1px 0 {ACCENT}; }}
[data-testid="stTabs"] [role="tablist"] > :not([role="tab"]) {{ background: transparent !important; }}
[data-testid="stTabPanel"] {{ padding-top: 0; }}
</style>
"""

pio.templates["triage"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT, size=12.5),
    title=dict(font=dict(family=FONT, size=18, color=TEXT), x=0, xanchor="left"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=MUTED, size=12))),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=MUTED, size=12))),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text="", bgcolor="rgba(0,0,0,0)",
                font=dict(color=MUTED)),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(family=MONO, color=TEXT, size=12)),
    colorway=[TIER_COLORS[t] for t in TIERS], margin=dict(l=10, r=10, t=40, b=10),
))
pio.templates.default = "plotly_dark+triage"


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def pill(tier: str) -> str:
    """The tier as a coloured text pill: colour is never the only signal."""
    return f'<span class="pill" style="color:{TIER_COLORS[tier]}">{tier}</span>'


def label(text: str) -> str:
    return f'<div class="bm-label">{text}</div>'


def section(index: str, title: str) -> str:
    """Numbered section heading, e.g. section('01', 'Operations queue')."""
    return f'<div class="bm-section"><span class="idx">{index}</span><span class="ttl">{title}</span></div>'


def heading(index: str, title: str, lede: str = "") -> str:
    """Section heading plus an optional one-line lede (HTML, numbers in <b>)."""
    return section(index, title) + (f'<p class="bm-lede">{lede}</p>' if lede else "")


def stat(value: str, caption: str, tier: str | None = None, sub: str = "") -> str:
    """Mono headline number over a small label, for bento cells; coloured only for tier counts."""
    color = TIER_COLORS[tier] if tier else TEXT
    extra = f'<span class="s">{sub}</span>' if sub else ""
    return (f'<div class="tx-stat"><span class="v" style="color:{color}">{value}</span>'
            f'<span class="c">{escape(caption)}</span>{extra}</div>')


def cell(title: str = "", body: str = "", inner: str = "") -> str:
    """One bento cell's content: light title, muted body (HTML), then any inner HTML."""
    t = f'<div class="t">{escape(title)}</div>' if title else ""
    b = f"<p>{body}</p>" if body else ""
    return f'<div class="tx-cell-in">{t}{b}{inner}</div>'


def cells(items: list[str], cols: int | None = None) -> str:
    """Static hairline bento grid: one cell per item, `cols` per row (default: all on one row)."""
    n = cols or max(len(items), 1)
    body = "".join(f'<div class="tx-cell">{i}</div>' for i in items)
    return f'<div class="tx-grid" style="grid-template-columns:repeat({n}, minmax(0, 1fr))">{body}</div>'


def steps(items: list[tuple[str, str]], active: int | None = 0) -> str:
    """Numbered hairline rows ('01 Title'); the active row is lit and shows its body."""
    rows = []
    for i, (title, body) in enumerate(items):
        on = i == active
        b = f'<div class="b">{body}</div>' if body and (on or active is None) else ""
        rows.append(f'<div class="tx-step{" on" if on else ""}"><span class="n">{i + 1:02d}</span>'
                    f'<span class="t">{escape(title)}</span>{b}</div>')
    return f'<div class="tx-steps">{"".join(rows)}</div>'


def hero_stat(value: str, caption: str, tier: str | None = None, size: str = "hero") -> str:
    """Giant number over an uppercase label; coloured only when it counts a tier.

    size='hero' is up to 7rem (one or two per row); size='row' is up to 4.8rem and
    never wraps, for rows of 3-6 numbers.
    """
    color = TIER_COLORS[tier] if tier else TEXT
    html = (f'<div class="bm-hero"><span class="num" style="color:{color}">{value}</span>'
            f'{label(caption)}</div>')
    return f'<div class="stat-row">{html}</div>' if size == "row" else html


def card(inner_html: str) -> str:
    return f'<div class="bm-card">{inner_html}</div>'


# ---- styled data tables ----------------------------------------------------

def tier_pill(tier) -> str:
    """Tier word in its colour on a faint tint of the same colour; plain text if unknown."""
    if tier not in TIER_COLORS:
        return f'<span class="dim">{escape(str(tier or "—"))}</span>'
    c = TIER_COLORS[tier]
    return f'<span class="bm-tier" style="color:{c};background:{c}1F">{tier}</span>'


def prob_bar(p) -> str:
    """Probability as a thin bar plus its value; 'no forecast' when withheld (NaN)."""
    if p is None or p != p:
        return '<span class="dim">no forecast</span>'
    pct = max(0.0, min(1.0, float(p)))
    return (f'<span class="bm-bar"><span class="track"><span class="fill" style="width:{pct * 100:.1f}%">'
            f'</span></span><span class="val">{format_prob(p)}</span></span>')


def chip(text: str, on: bool = True) -> str:
    return f'<span class="bm-chip{" on" if on else ""}">{escape(text)}</span>'


def html_table(df, columns: list[tuple], max_height: int = 560, rank: bool = False, foot: str = "") -> str:
    """A styled table as HTML for st.markdown(..., unsafe_allow_html=True).

    columns: (key, header, align, render) per column, where align is 'l' or 'r' and
    render(value, row) -> HTML (or None to show the escaped value). Use key=None for
    columns computed from the whole row.
    """
    head = ("<th></th>" if rank else "") + "".join(
        f'<th class="{"r" if a == "r" else ""}">{escape(h)}</th>' for _, h, a, _ in columns)
    body = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        cells = f'<td class="rank">{i:02d}</td>' if rank else ""
        for key, _, align, render in columns:
            value = getattr(row, key) if key else None
            html = render(value, row) if render else escape("" if value is None else str(value))
            cells += f'<td class="{"r" if align == "r" else ""}">{html}</td>'
        body.append(f"<tr>{cells}</tr>")
    table = (f'<div class="bm-table-wrap" style="max-height:{max_height}px"><table class="bm-table">'
             f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return table + (f'<div class="bm-table-foot">{foot}</div>' if foot else "")


def top_bar(meta: list[tuple[str, str]]) -> str:
    """Wordmark plus uppercase status items, e.g. [('Split', 'test'), ('Events', '2,167')]."""
    items = "".join(f"<span>{k} <b>{v}</b></span>" for k, v in meta)
    return (f'<div class="bm-top"><span class="mark">Triage<i> / </i>TCA</span>'
            f'<span class="meta"><span><span class="bm-dot"></span>Live queue</span>{items}</span></div>')


def marquee(items: list[str]) -> str:
    """Seamless scrolling ticker: the items are repeated so the loop has no gap."""
    if not items:
        return ""
    track = "".join(f"<span>{i}</span>" for i in items) * 2
    return f'<div class="bm-marquee"><div class="track">{track}</div></div>'


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pc_axis(fig, axis: str = "y", lo: float = RISK_FLOOR, hi: float = 0, step: int = 5):
    """Ticks on a log10(Pc) axis written as powers of ten (10⁻⁶), not raw log10 numbers."""
    vals = list(range(int(lo), int(hi) + 1, step))
    text = ["1" if v == 0 else "10" + str(v).translate(_SUP) for v in vals]
    getattr(fig, f"update_{axis}axes")(tickvals=vals, ticktext=text, title_text="Collision probability (Pc)")
    return fig


MISS_TICKS = {10: "10 m", 100: "100 m", 1_000: "1 km", 5_000: "5 km", 10_000: "10 km", 100_000: "100 km"}


def miss_axis(fig, axis: str = "x", title: str = "Miss distance"):
    """Ticks on a log miss-distance axis (metres) written in m / km."""
    getattr(fig, f"update_{axis}axes")(type="log", tickvals=list(MISS_TICKS), ticktext=list(MISS_TICKS.values()),
                                       title_text=title)
    return fig
