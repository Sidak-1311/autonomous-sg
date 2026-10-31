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
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.tiering import RISK_FLOOR, TIERS, TIER_COLORS

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

/* surfaces */
.bm-card {{ background: {SURFACE}; border-radius: 20px; padding: 1.4rem 1.5rem; }}
[data-testid="stMetric"] {{ background: {SURFACE}; border-radius: 20px; padding: 1.1rem 1.3rem; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.72rem !important; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-family: {FONT}; font-weight: 800; letter-spacing: -0.05em; }}
[data-testid="stExpander"] details {{ border: none; background: {SURFACE}; border-radius: 20px; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

/* tabs: text only, yellow underline on the active one */
.stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font: 600 0.8rem {FONT}; letter-spacing: 0.08em; text-transform: uppercase;
  color: {MUTED}; padding: 0.9rem 0; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {TEXT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; height: 2px; }}

.pill {{ display: inline-block; padding: 0.12rem 0.65rem; border-radius: 999px; font: 600 0.72rem {FONT};
  letter-spacing: 0.08em; border: 1px solid currentColor; }}
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


def hero_stat(value: str, caption: str, tier: str | None = None) -> str:
    """Giant number over an uppercase label; coloured only when it counts a tier."""
    color = TIER_COLORS[tier] if tier else TEXT
    return (f'<div class="bm-hero"><span class="num" style="color:{color}">{value}</span>'
            f'{label(caption)}</div>')


def card(inner_html: str) -> str:
    return f'<div class="bm-card">{inner_html}</div>'


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
