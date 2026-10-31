"""Interstellar look for the dashboard: a static starfield, one Plotly template, and readable Pc / distance axes.

Colours come from .streamlit/config.toml (UI) and src.tiering.TIER_COLORS (data). Nothing animates.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.tiering import RISK_FLOOR, TIERS, TIER_COLORS

MUTED, GRID, SURFACE = "#8B95A8", "rgba(139,149,168,0.10)", "#0C1322"

# A tiled starfield (three layers at different tile sizes, so it doesn't look gridded) under a faint
# blue planetary-horizon glow, after the 'Space missions made simple' reference.
CSS = """
<style>
.stApp {
  background:
    radial-gradient(ellipse 90% 45% at 50% -12%, rgba(74,134,197,0.20), transparent 70%),
    radial-gradient(1px 1px at 23px 41px, rgba(230,234,242,0.55), transparent 50%),
    radial-gradient(1px 1px at 151px 97px, rgba(230,234,242,0.35), transparent 50%),
    radial-gradient(1.5px 1.5px at 89px 173px, rgba(201,209,230,0.45), transparent 50%),
    radial-gradient(1px 1px at 211px 19px, rgba(230,234,242,0.30), transparent 50%),
    #05070D;
  background-size: 100% 100%, 240px 240px, 310px 310px, 420px 420px, 270px 270px, auto;
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.hero-eyebrow { font: 600 0.72rem 'JetBrains Mono', monospace; letter-spacing: 0.22em; color: #8B95A8;
                text-transform: uppercase; margin-bottom: -0.4rem; }
.hero-sub { color: #8B95A8; max-width: 46rem; margin-top: -0.6rem; }
[data-testid="stMetric"] { background: rgba(12,19,34,0.72); backdrop-filter: blur(6px); }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.04em; }
.pill { display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font: 600 0.75rem 'JetBrains Mono',
        monospace; letter-spacing: 0.08em; border: 1px solid currentColor; }
</style>
"""

pio.templates["interstellar"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,19,34,0.55)",
    font=dict(family="Inter, sans-serif", color="#C9D1E6", size=13),
    title=dict(font=dict(family="Space Grotesk, sans-serif", size=16, color="#E6EAF2"), x=0, xanchor="left"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#1C2638", tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#1C2638", tickfont=dict(color=MUTED)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text="", bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor="#1C2638", font=dict(family="JetBrains Mono, monospace")),
    colorway=[TIER_COLORS[t] for t in TIERS], margin=dict(l=10, r=10, t=50, b=10),
))
pio.templates.default = "plotly_dark+interstellar"


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def pill(tier: str) -> str:
    """The tier as a coloured text pill: colour is never the only signal."""
    return f'<span class="pill" style="color:{TIER_COLORS[tier]}">{tier}</span>'


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
