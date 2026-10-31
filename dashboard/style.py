"""Dashboard look: near-black, one yellow interface accent, one tight grotesk with
giant display numbers and small uppercase labels; one Plotly template; readable
Pc / distance axes.

Colours come from .streamlit/config.toml (UI) and src.tiering.TIER_COLORS (data).
The yellow ACCENT is for interface chrome only and never encodes a tier.

Layout helpers (all return HTML for st.markdown(..., unsafe_allow_html=True)):
    section("01", "Operations queue")        numbered section header
    hero_stat("394", "Critical", "CRITICAL") giant number over an uppercase label
    label("Time to TCA")                     small uppercase label
    card(inner_html)                         20px rounded surface
    pill("CRITICAL")                         tier as a coloured text pill
"""
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.tiering import RISK_FLOOR, TIERS, TIER_COLORS, format_prob

BG, SURFACE, LINE = "#101010", "#1A1A1A", "#262626"
TEXT, MUTED, ACCENT = "#F2F2F2", "#8A8A8A", "#FFD600"
GRID = "rgba(242,242,242,0.07)"
FONT = "'Inter Tight', 'Helvetica Neue', Helvetica, Arial, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;800&display=swap');
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.5rem; max-width: 1480px; }}
html, body, .stApp, [data-testid="stMarkdownContainer"] {{
  font-family: {FONT}; font-feature-settings: "tnum" 1, "ss01" 1; }}
h1, h2, h3 {{ font-family: {FONT}; letter-spacing: -0.05em; line-height: 0.95; }}
h1 {{ font-size: clamp(3rem, 7vw, 6.5rem) !important; font-weight: 800 !important; }}

/* small uppercase labels and the old eyebrow class */
.bm-label, .hero-eyebrow {{ font: 600 0.72rem {FONT}; letter-spacing: 0.08em; text-transform: uppercase;
  color: {MUTED}; }}
.hero-eyebrow {{ margin-bottom: -0.2rem; }}
.hero-sub {{ color: {MUTED}; max-width: 44rem; font-size: 1.05rem; letter-spacing: -0.01em; margin-top: 0.4rem; }}

/* numbered sections: 01 / OPERATIONS QUEUE, over a hairline */
.bm-section {{ display: flex; align-items: baseline; gap: 1rem; border-top: 1px solid {LINE};
  padding-top: 0.9rem; margin: 2.2rem 0 1rem; }}
.bm-section .idx {{ font: 600 0.8rem {FONT}; color: {ACCENT}; letter-spacing: 0.04em; }}
.bm-section .ttl {{ font: 800 clamp(1.6rem, 3vw, 2.6rem) {FONT}; letter-spacing: -0.05em; line-height: 1; }}

/* giant numbers */
.bm-hero {{ display: flex; flex-direction: column; gap: 0.35rem; }}
.bm-hero .num {{ font: 800 clamp(3.2rem, 7.5vw, 7.2rem) {FONT}; letter-spacing: -0.07em; line-height: 0.82; }}

/* row-sized giant numbers (rows of 3-6); from the Fleet section's tested sizes */
.stat-row .bm-hero {{ gap: 0.75rem; }}
.stat-row .bm-hero .num {{ font-size: clamp(2.4rem, 4.6vw, 4.8rem); white-space: nowrap; line-height: 0.9; }}

/* surfaces */
.bm-card {{ background: {SURFACE}; border-radius: 20px; padding: 1.4rem 1.5rem; }}
[data-testid="stMetric"] {{ background: {SURFACE}; border-radius: 20px; padding: 1.1rem 1.3rem; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.72rem !important; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-family: {FONT}; font-weight: 800; letter-spacing: -0.05em; }}
[data-testid="stExpander"] details {{ border: none; background: {SURFACE}; border-radius: 20px; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

/* tabs: text only, yellow underline on the active one */

.pill {{ display: inline-block; padding: 0.12rem 0.65rem; border-radius: 999px; font: 600 0.72rem {FONT};
  letter-spacing: 0.08em; border: 1px solid currentColor; }}

/* ---- page shell: no Streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"], .stDeployButton {{ display: none !important; }}
header[data-testid="stHeader"] {{ height: 0; min-height: 0; }}
.block-container {{ padding: 1.4rem 2.6rem 6rem; }}
[data-testid="stSidebar"] {{ background: #0C0C0C; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] label p {{ font: 600 0.72rem {FONT}; letter-spacing: 0.08em; text-transform: uppercase;
  color: {MUTED}; }}

/* top bar: wordmark left, status right, hairline under */
.bm-top {{ display: flex; justify-content: space-between; align-items: center; padding-bottom: 1rem;
  border-bottom: 1px solid {LINE}; }}
.bm-top .mark {{ font: 800 1.15rem {FONT}; letter-spacing: -0.04em; }}
.bm-top .mark i {{ font-style: normal; color: {ACCENT}; }}
.bm-top .meta {{ display: flex; gap: 1.6rem; font: 600 0.68rem {FONT}; letter-spacing: 0.08em;
  text-transform: uppercase; color: {MUTED}; }}
.bm-top .meta b {{ color: {TEXT}; font-weight: 600; }}
.bm-dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {ACCENT};
  margin-right: 0.45rem; vertical-align: 1px; box-shadow: 0 0 10px {ACCENT}; }}

/* hero headline */
.bm-display {{ font: 800 clamp(2.8rem, 5.6vw, 6.6rem) {FONT}; letter-spacing: -0.065em; line-height: 0.88;
  margin: 2.4rem 0 1.4rem; color: {TEXT}; overflow-wrap: normal; word-break: keep-all; hyphens: none; }}
.bm-display em {{ font-style: normal; color: {MUTED}; }}
.bm-lede {{ font: 400 1.15rem/1.45 {FONT}; letter-spacing: -0.015em; color: {MUTED}; max-width: 34rem; }}
.bm-lede b {{ color: {TEXT}; font-weight: 600; }}

/* KPI row: big flat numbers on hairlines instead of boxed metrics */
.st-key-kpis [data-testid="stMetric"] {{ background: transparent; border-radius: 0; padding: 0.9rem 0 0;
  border-top: 1px solid {LINE}; }}
.st-key-kpis [data-testid="stMetricValue"] {{ font-size: clamp(2rem, 3.6vw, 4.2rem); line-height: 0.95;
  letter-spacing: -0.06em; white-space: nowrap; }}
.st-key-kpis [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; }}
.st-key-kpis [data-testid="stColumn"] {{ min-width: 0; }}
.st-key-kpis-queue-depth [data-testid="stMetric"] {{ background: transparent; padding: 0.9rem 0 0;
  border-top: 1px solid {LINE}; border-radius: 0; }}
.st-key-kpis-queue-depth [data-testid="stMetricValue"] {{ font-size: clamp(4rem, 9vw, 9rem); line-height: 0.85;
  letter-spacing: -0.07em; }}
.st-key-kpi-critical [data-testid="stMetricValue"] {{ color: {TIER_COLORS["CRITICAL"]}; }}
.st-key-kpi-elevated [data-testid="stMetricValue"] {{ color: {TIER_COLORS["ELEVATED"]}; }}
.st-key-kpi-safe [data-testid="stMetricValue"] {{ color: {TIER_COLORS["SAFE"]}; }}

/* scrolling ticker of the worst events */
.bm-marquee {{ overflow: hidden; border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE};
  margin: 2.6rem -2.6rem 0; padding: 0.95rem 0; white-space: nowrap; }}
.bm-marquee .track {{ display: inline-block; animation: bm-scroll 70s linear infinite; }}
.bm-marquee:hover .track {{ animation-play-state: paused; }}
.bm-marquee span {{ font: 600 0.85rem {FONT}; letter-spacing: 0.04em; text-transform: uppercase;
  color: {MUTED}; margin-right: 3.2rem; }}
.bm-marquee span b {{ color: {TEXT}; font-weight: 800; letter-spacing: -0.01em; }}
@keyframes bm-scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .bm-marquee .track {{ animation: none; }} }}

/* filter chips stay neutral: the yellow accent never marks data. Streamlit 1.58 renders them as
   span[data-baseweb="tag"] (white on #F2F2F2 by default, i.e. invisible), not inside stMultiSelectTagsContainer */
span[data-baseweb="tag"] {{ background-color: {LINE} !important; border: 1px solid #3A3A3A !important;
  border-radius: 999px !important; }}
span[data-baseweb="tag"] * {{ color: {TEXT} !important; fill: {TEXT} !important;
  font-weight: 600; letter-spacing: 0.04em; }}
.st-key-kpis [data-testid="stHorizontalBlock"] {{ gap: 1.6rem; }}

/* ---- styled data tables (html_table) ---- */
.bm-table-wrap {{ overflow: auto; border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE};
  scrollbar-width: thin; scrollbar-color: {LINE} transparent; }}
.bm-table {{ width: 100%; border-collapse: collapse; font: 400 0.92rem {FONT};
  font-feature-settings: "tnum" 1; color: {TEXT}; }}
.bm-table thead th {{ position: sticky; top: 0; z-index: 2; background: {BG}; text-align: left;
  font: 600 0.66rem {FONT}; letter-spacing: 0.09em; text-transform: uppercase; color: {MUTED};
  padding: 0.85rem 0.9rem 0.7rem; border-bottom: 1px solid {LINE}; white-space: nowrap; }}
.bm-table td {{ padding: 0.78rem 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.045);
  white-space: nowrap; vertical-align: middle; }}
.bm-table th, .bm-table td {{ border-left: none !important; border-right: none !important;
  border-top: none !important; }}
.bm-table tr {{ background: transparent !important; }}
.bm-table tbody tr {{ transition: background 0.12s ease; }}
.bm-table tbody tr:hover {{ background: {SURFACE} !important; }}
.bm-table .r {{ text-align: right; }}
.bm-table th.r {{ text-align: right; }}
.bm-table .dim {{ color: {MUTED}; }}
.bm-table .strong {{ font-weight: 800; letter-spacing: -0.02em; }}
.bm-table .rank {{ color: #4A4A4A; font-weight: 600; width: 2.4rem; }}
.bm-tier {{ display: inline-flex; align-items: center; gap: 0.45rem; font: 700 0.7rem {FONT};
  letter-spacing: 0.08em; padding: 0.28rem 0.65rem 0.28rem 0.55rem; border-radius: 999px; }}
.bm-tier::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
.bm-bar {{ display: inline-flex; align-items: center; gap: 0.7rem; min-width: 9.5rem; }}
.bm-bar .track {{ flex: 1; height: 4px; border-radius: 4px; background: {LINE}; overflow: hidden; }}
.bm-bar .fill {{ display: block; height: 100%; border-radius: 4px; background: {TEXT}; }}
.bm-bar .val {{ width: 3.6rem; text-align: right; font-weight: 600; }}
.bm-chip {{ display: inline-block; font: 700 0.6rem {FONT}; letter-spacing: 0.09em; text-transform: uppercase;
  padding: 0.22rem 0.5rem; border-radius: 6px; border: 1px solid {LINE}; color: {MUTED}; margin-right: 0.3rem; }}
.bm-chip.on {{ color: {TEXT}; border-color: #5A5A5A; }}
.bm-table-foot {{ display: flex; justify-content: space-between; padding-top: 0.7rem; font: 600 0.66rem {FONT};
  letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED}; }}

.st-key-queue-filters [data-testid="stButtonGroup"] > div {{ flex-wrap: wrap; row-gap: 0.4rem; }}
.st-key-queue-filters [data-testid="stCheckbox"] {{ margin-top: 0.35rem; }}
.st-key-queue-filters [data-testid="stCheckbox"] label p {{ text-transform: none; letter-spacing: 0;
  font: 500 0.82rem {FONT} !important; color: {TEXT} !important; }}
.st-key-queue-filters [data-testid="stCheckbox"] label {{ white-space: nowrap; }}
.st-key-queue-filters [data-testid="stCheckbox"] label p {{ overflow: visible !important; }}
.bm-table td, .bm-table thead th {{ padding-left: 0.7rem; padding-right: 0.7rem; }}
.bm-table .sub {{ display: block; font-size: 0.72rem; color: {MUTED}; margin-top: 0.15rem; }}
.bm-bar {{ min-width: 7.5rem !important; }}
.bm-table td:has(.why) {{ white-space: normal; min-width: 13rem; }}
.bm-table .why {{ display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.3;
  white-space: normal; max-width: 16rem; overflow: hidden;
  vertical-align: middle; color: {MUTED}; font-size: 0.84rem; cursor: help; }}

/* queue filter row: small uppercase labels, pills as filter chips */
.st-key-queue-filters {{ padding: 0.4rem 0 1.1rem; }}
.st-key-queue-filters label p {{ font: 600 0.66rem {FONT} !important; letter-spacing: 0.09em;
  text-transform: uppercase; color: {MUTED} !important; }}
.st-key-queue-filters [data-testid="stBaseButton-pills"],
.st-key-queue-filters [data-testid="stBaseButton-pillsActive"] {{ border-radius: 999px !important;
  padding: 0.25rem 0.8rem !important; min-height: 0 !important; font: 600 0.72rem {FONT}; }}
.st-key-queue-filters [data-testid="stBaseButton-pills"] {{ background: transparent !important;
  border: 1px solid {LINE} !important; color: {MUTED} !important; }}
.st-key-queue-filters [data-testid="stBaseButton-pillsActive"] {{ background: {TEXT} !important;
  border: 1px solid {TEXT} !important; }}
.st-key-queue-filters [data-testid="stBaseButton-pillsActive"] * {{ color: {BG} !important; }}
.st-key-queue-filters input {{ font-feature-settings: "tnum" 1; }}

/* buttons: solid white pills with dark text (the white primary made text vanish) */
.stButton button, [data-testid="stFormSubmitButton"] button {{ background: {TEXT} !important;
  border: none !important; border-radius: 999px !important; padding: 0.5rem 1.4rem !important; }}
.stButton button *, [data-testid="stFormSubmitButton"] button * {{ color: {BG} !important;
  font: 600 0.8rem {FONT} !important; letter-spacing: 0.06em; text-transform: uppercase; }}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background: {ACCENT} !important; }}

/* tabs become a sticky top nav (Streamlit 1.66 markup: stTabs > ... > role=tablist > stTab) */
[data-testid="stTabs"] [role="tablist"] {{ position: sticky; top: 0; z-index: 50; background: {BG};
  gap: 2.4rem; border-bottom: 1px solid {LINE}; padding-top: 0.5rem; margin-top: 1.8rem; }}
[data-testid="stTab"] {{ padding: 0.95rem 0 !important; background: transparent !important; }}
[data-testid="stTab"] p {{ font: 600 0.74rem {FONT} !important; letter-spacing: 0.1em !important;
  text-transform: uppercase; color: {MUTED}; transition: color 0.15s ease; }}
[data-testid="stTab"]:hover p, [data-testid="stTab"][aria-selected="true"] p {{ color: {TEXT}; }}
[data-testid="stTab"][aria-selected="true"] {{ box-shadow: inset 0 -2px 0 {ACCENT}; }}
[data-testid="stTabs"] [role="tablist"] > :not([role="tab"]) {{ background: transparent !important; }}
[data-testid="stTabPanel"] {{ padding-top: 0.6rem; }}
</style>
"""

pio.templates["triage"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT, size=13),
    title=dict(font=dict(family=FONT, size=18, color=TEXT), x=0, xanchor="left"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=MUTED, size=12))),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=MUTED, size=12))),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text="", bgcolor="rgba(0,0,0,0)",
                font=dict(color=MUTED)),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(family=FONT, color=TEXT)),
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
    """Numbered section header, e.g. section('01', 'Operations queue')."""
    return f'<div class="bm-section"><span class="idx">{index}</span><span class="ttl">{title}</span></div>'


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
    return (f'<div class="bm-top"><span class="mark">Triage<i>—</i>TCA</span>'
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
