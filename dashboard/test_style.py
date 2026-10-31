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
    assert pio.templates.default == "plotly_dark+triage"
    assert pio.templates["triage"].layout.colorway[-1] == TIER_COLORS[CRITICAL]


def test_interface_accent_never_doubles_as_a_tier_colour():
    assert style.ACCENT.lower() not in {c.lower() for c in TIER_COLORS.values()}


def test_layout_helpers():
    assert 'class="idx">01<' in style.section("01", "Operations queue")
    assert ">Operations queue<" in style.section("01", "Operations queue")
    hero = style.hero_stat("394", "Critical", CRITICAL)
    assert ">394<" in hero and TIER_COLORS[CRITICAL] in hero and ">Critical<" in hero
    assert style.TEXT in style.hero_stat("2,167", "Events")   # uncoloured when not a tier count
    assert style.card("<b>x</b>").startswith('<div class="bm-card">')


def test_pill_always_carries_the_tier_as_text():
    html = style.pill(CRITICAL)
    assert ">CRITICAL<" in html and TIER_COLORS[CRITICAL] in html


def test_default_data_dir_is_the_repo_data_folder(monkeypatch):
    # Regression: sections/common.py sits one level deeper than app.py
    from dashboard.sections import common
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert common.data_dir() == Path(__file__).resolve().parent.parent / "data"


def test_shell_helpers():
    bar = style.top_bar([("Split", "test"), ("Events", "2,167")])
    assert "Triage" in bar and "<b>2,167</b>" in bar
    assert style.marquee([]) == ""
    ticker = style.marquee(["EVT 1", "EVT 2"])
    assert ticker.count("EVT 1") == 2  # repeated for a seamless loop


def test_hero_stat_row_size_wraps_in_stat_row():
    row = style.hero_stat("1,466", "Safe", "SAFE", size="row")
    assert row.startswith('<div class="stat-row">') and ">1,466<" in row
    assert "stat-row" not in style.hero_stat("394", "Critical")
    assert ".stat-row .bm-hero .num" in style.CSS and "nowrap" in style.CSS
