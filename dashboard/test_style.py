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
    assert "Apsis" in bar and "<b>2,167</b>" in bar
    assert style.marquee([]) == ""
    ticker = style.marquee(["EVT 1", "EVT 2"])
    assert ticker.count("EVT 1") == 2  # repeated for a seamless loop


def test_hero_stat_row_size_wraps_in_stat_row():
    row = style.hero_stat("1,466", "Safe", "SAFE", size="row")
    assert row.startswith('<div class="stat-row">') and ">1,466<" in row
    assert "stat-row" not in style.hero_stat("394", "Critical")
    assert ".stat-row .bm-hero .num" in style.CSS and "nowrap" in style.CSS


def test_html_table_renders_rows_escapes_and_ranks():
    import pandas as pd
    df = pd.DataFrame({"event_id": [7, 8], "tier_final": ["CRITICAL", "SAFE"], "p": [0.5, float("nan")],
                       "note": ["<b>x</b>", None]})
    html = style.html_table(df, [
        ("event_id", "Event", "l", None),
        ("tier_final", "Tier", "l", lambda v, r: style.tier_pill(v)),
        ("p", "P", "r", lambda v, r: style.prob_bar(v)),
        ("note", "Note", "l", None),
    ], rank=True, foot="2 rows")
    assert html.count("<tr>") == 3                      # header + 2 rows
    assert ">01<" in html and ">02<" in html            # rank column
    assert "&lt;b&gt;x&lt;/b&gt;" in html               # raw values are escaped
    assert TIER_COLORS[CRITICAL] in html and "no forecast" in html and "50.0%" in html
    assert 'class="bm-table-foot"' in html


def test_table_bits_handle_edge_values():
    assert "width:100.0%" in style.prob_bar(1.7)        # clipped to the track
    assert "width:0.0%" in style.prob_bar(-0.2)
    assert "dim" in style.tier_pill(None) and "UNKNOWN" not in style.tier_pill(None)
    assert 'class="bm-chip"' in style.chip("Storm", on=False) and "on" in style.chip("Review")


def test_bento_helpers():
    assert ">Fleet overview<" in style.heading("03", "Fleet overview")
    assert 'class="bm-lede"' in style.heading("03", "Fleet", "<b>96%</b> need a look")
    assert "bm-lede" not in style.heading("03", "Fleet")                  # no empty lede paragraph
    grid = style.cells(["a", "b", "c"])
    assert grid.count('class="tx-cell"') == 3 and "repeat(3," in grid
    assert "repeat(2," in style.cells(["a", "b", "c"], cols=2)
    assert "repeat(1," in style.cells([])                                   # empty: still valid CSS
    assert style.cell() == '<div class="tx-cell-in"></div>'
    assert "&lt;x&gt;" in style.cell("<x>")                                 # titles are escaped


def test_stat_colours_only_tier_counts():
    assert TIER_COLORS[CRITICAL] in style.stat("396", "Critical", CRITICAL)
    plain = style.stat("2,167", "Events", sub="test split")
    assert style.TEXT in plain and 'class="s"' in plain


def test_steps_light_one_row_and_number_from_01():
    html = style.steps([("Pc", "body one"), ("Miss", "body two")], active=1)
    assert ">01<" in html and ">02<" in html and html.count("tx-step on") == 1
    assert "body two" in html and "body one" not in html                  # only the active body shows
    both = style.steps([("Pc", "x"), ("Miss", "y")], active=None)
    assert "x" in both and "y" in both and "tx-step on" not in both
    assert style.steps([]) == '<div class="tx-steps"></div>'


def test_accent_is_not_yellow_and_not_a_tier():
    assert style.ACCENT == "#ECF0BD" and style.BG == "#000000"


def test_status_pill_is_neutral_text():
    html = style.status_pill("watch")
    assert ">WATCH<" in html and "bm-status" in html
    assert not any(c in html for c in TIER_COLORS.values())


def test_sci_typesets_powers_times_and_keeps_quantities_whole():
    assert style.sci("7.9×10⁻³⁰") == '7.9<span class="times">×</span>10<sup class="sx">−30</sup>'
    assert style.sci("<10⁻³⁰").endswith('10<sup class="sx">−30</sup>')     # floor form
    assert '<span class="nw">2 d 04 h</span>' in style.sci("T−2 d 04 h")
    assert '<span class="nw">88.8%–92.6%</span>' in style.sci("95% CI 88.8%–92.6%")
    assert style.sci(style.sci("10⁻⁴")) == style.sci("10⁻⁴")               # idempotent
    assert style.sci("plain 1,000 m and <b>tags</b>") == "plain 1,000 m and <b>tags</b>"
    assert style.sci(None) == "None" and style.sci("") == ""


def test_helpers_apply_sci():
    assert 'class="sx"' in style.stat("7.9×10⁻³⁰", "95% bound")
    assert 'class="sx"' in style.cell("Pc", "Pc ≥ 1.0×10⁻⁴")
    assert 'class="sx"' in style.hero_stat("<10⁻³⁰", "Pc now")
    import pandas as pd
    html = style.html_table(pd.DataFrame({"pc": ["1.3×10⁻³"]}), [("pc", "Pc", "r", None)])
    assert 'class="sx"' in html


def test_keep_case_protects_math_and_units_in_uppercase_labels():
    assert style.keep_case("Pc now") == '<span class="kc">Pc</span> now'
    assert '<span class="kc">P(Critical at TCA)</span>' in style.keep_case("P(Critical at TCA)")
    assert '<span class="kc">cm</span>' in style.keep_case("fragments ≥ 10 cm")
    assert '<span class="kc">yr</span>' in style.keep_case("chain risk / yr")
    assert style.keep_case("Miss now") == "Miss now"                       # ordinary words untouched
    assert style.keep_case("Time to TCA") == "Time to TCA"
    assert 'class="kc"' in style.label("Pc at TCA")
    import pandas as pd
    assert '<span class="kc">Pc</span>' in style.html_table(pd.DataFrame({"a": [1]}), [("a", "Pc now", "r", None)])


def test_plain_round_trips_typeset_text():
    for text in ["7.9×10⁻³⁰", "<10⁻³⁰ at TCA", "Pc ≥ 1.0×10⁻⁴, 2 d 04 h", "95% CI 88.8%–92.6%"]:
        assert style.plain(style.sci(text)) == text
    assert style.plain(style.label("fragments ≥ 10 cm")) == "fragments ≥ 10 cm"


def test_sci_and_keep_case_never_touch_tag_attributes():
    html = '<span class="why" title="q90 Pc 9.6×10⁻¹⁶ and 2 d 04 h">miss 97 m &lt; 1,000 m</span>'
    out = style.sci(html)
    assert 'title="q90 Pc 9.6×10⁻¹⁶ and 2 d 04 h"' in out          # attribute untouched
    assert style.keep_case('<b title="Pc now">Pc</b>') == '<b title="Pc now"><span class="kc">Pc</span></b>'
    assert "--n:9" in style.stat("7.9×10⁻³⁰", "95% bound")             # long values get a smaller size
