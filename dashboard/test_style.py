"""Verification loop for the dashboard's look: run `pytest dashboard -q`."""
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import style  # noqa: E402
from src.tiering import CRITICAL, TIER_COLORS  # noqa: E402


def test_pc_axis_writes_powers_of_ten():
    ax = style.pc_axis(go.Figure()).layout.yaxis
    assert list(ax.tickvals) == [-30, -25, -20, -15, -10, -5, 0]
    assert list(ax.ticktext) == ["10⁻³⁰", "10⁻²⁵", "10⁻²⁰", "10⁻¹⁵", "10⁻¹⁰", "10⁻⁵", "1"]
    assert ax.title.text == "Collision probability (Pc)"


def test_pc_axis_on_x_with_a_custom_range():
    ax = style.pc_axis(go.Figure(), "x", lo=-8, step=4).layout.xaxis
    assert list(ax.ticktext) == ["10⁻⁸", "10⁻⁴", "1"]


def test_miss_axis_is_log_with_metre_and_km_labels():
    ax = style.miss_axis(go.Figure(), "y").layout.yaxis
    assert ax.type == "log" and "1 km" in ax.ticktext and "5 km" in ax.ticktext and "100 m" in ax.ticktext


def test_template_is_default_and_uses_the_tier_palette():
    assert pio.templates.default == "plotly_dark+interstellar"
    assert pio.templates["interstellar"].layout.colorway[-1] == TIER_COLORS[CRITICAL]


def test_pill_always_carries_the_tier_as_text():
    html = style.pill(CRITICAL)
    assert ">CRITICAL<" in html and TIER_COLORS[CRITICAL] in html
