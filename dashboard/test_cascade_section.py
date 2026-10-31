"""Verification loop for the debris-cascade display (Event detail card + section 06): run `pytest dashboard -q`."""
import pandas as pd

from dashboard.sections.cascade import pct
from dashboard.test_app import by_label, isolated, page_text, render, write_pipeline  # noqa: F401  (autouse fixture)

# events 0-4 of the shared fixture: 0 and 3 CRITICAL, 2 alone in its shell, 4 a partial breakup
CASCADE = pd.DataFrame({
    "event_id": [0, 1, 2, 3, 4], "frag_10cm": [1400.0, 800.0, 12000.0, 600.0, 5.1],
    "catastrophic": [True, True, True, True, False], "energy_j_per_g": [5000.0, 900.0, 7000.0, 300.0, 0.5],
    "shell_alt_km": [700.0, 710.0, 420.0, 690.0, 800.0], "shell_missions": [2, 3, 0, 1, 1],
    "chain_p1y_max": [0.0103, 0.0002, 0.0, 0.00042, 2e-6], "mass_primary_kg": 2000.0, "mass_secondary_kg": 10.0,
    "chain_hits_1y": [0.012, 0.0003, 0.0, 0.00042, 2e-6]})


def with_cascade(tmp_path):
    root = write_pipeline(tmp_path)
    path = root / "processed" / "events_scored.parquet"
    ev = pd.read_parquet(path).merge(CASCADE, on="event_id")
    ev["debris_expected"] = 10 ** ev.risk_forecast * ev.frag_10cm
    ev.to_parquet(path)
    return root


def test_pct_keeps_small_probabilities_readable():
    assert pct(0.0103) == "1.0%" and pct(0.00042) == "0.042%" and pct(2e-6) == "0.00020%" and pct(float("nan")) == "n/a"
    assert pct(0) == "0%" and pct(0.15) == "15%" and pct(0.0042) == "0.42%"   # one precision down a column


def test_detail_card_for_a_catastrophic_event_with_neighbours(tmp_path, monkeypatch):
    text = page_text(render(with_cascade(tmp_path), monkeypatch))   # event 0 is selected first
    assert "If this collision happened" in text and ">1,400</span>" in text and "fragments ≥ 10 cm" in text
    assert "catastrophic: both bodies break up (5,000 J/g)" in text
    assert "2 other missions share its ±50 km shell at 700 km" in text and "<b>1.0%</b> chance per year" in text


def test_detail_card_alone_in_its_shell_and_a_partial_breakup(tmp_path, monkeypatch):
    at = render(with_cascade(tmp_path), monkeypatch)
    by_label(at.selectbox, "Event (worst first)").set_value(2).run()
    assert "No other mission in this dataset shares its ±50 km shell at 420 km" in page_text(at)
    by_label(at.selectbox, "Event (worst first)").set_value(4).run()
    text = page_text(at)
    assert "non-catastrophic: only the smaller body" in text and "1 other mission share" in text


def test_section_ranks_by_expected_fragments(tmp_path, monkeypatch):
    at = render(with_cascade(tmp_path), monkeypatch)
    text = page_text(at)
    assert "<b>80%</b> would be catastrophic" in text and "(event 0)" in text   # 4 of 5; event 0 has the worst chain
    table = next(d.value for d in at.dataframe if "debris_expected" in d.value)
    expected = (CASCADE.set_index("event_id").frag_10cm
                * 10 ** pd.Series({0: -4.0, 1: -6.0, 2: -30.0, 3: -30.0, 4: -3.0}))
    assert list(table.event_id) == list(expected.sort_values(ascending=False).index)


def test_without_cascade_columns_nothing_breaks(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    text = page_text(at)
    assert "If this collision happened" not in text and "No cascade columns" in text
