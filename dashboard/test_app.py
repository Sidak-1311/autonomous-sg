"""Verification loop for the dashboard: run `pytest dashboard -q` after every change to dashboard/app.py.

Each test renders the whole app headless (every tab runs on each rerun) against a small synthetic data folder
(DATA_DIR), so it needs neither the real CSVs nor the pipeline. Edge cases: values exactly on the tier
thresholds, the -30 risk floor, single-CDM events, events with no CDM at the decision point, unknown object
types, missing model columns (NaN p_critical / drivers / reason), empty data and filters that match nothing.
"""
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import brief  # noqa: E402  (the module object the app imports, so patches reach it)

APP = str(Path(__file__).resolve().parent / "app.py")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No cached loads or briefs leak between tests, and no real key or network is ever used."""
    st.cache_data.clear()
    monkeypatch.setattr(brief, "api_key", lambda *a: None)

    def no_network(*a, **k):
        raise AssertionError("a test reached the Gemini API")
    monkeypatch.setattr(brief, "call_gemini", no_network)
COLS = ["event_id", "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
        "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type"]


def cdm(event, t, risk, miss, obj="DEBRIS", sigma=50.0):
    return dict(event_id=event, time_to_tca=t, risk=risk, max_risk_estimate=risk, miss_distance=miss,
                relative_speed=10_000.0, relative_position_r=miss / 3, relative_position_t=miss / 2,
                relative_position_n=miss / 2, c_object_type=obj, c_sigma_r=sigma, t_sigma_r=5.0)


# Edge cases on purpose: exact thresholds, the floor, one CDM, odd object types, missing and huge sigma.
RAW = pd.DataFrame([
    cdm(0, 6.0, -8.0, 20_000), cdm(0, 4.0, -7.0, 9_000), cdm(0, 2.1, -4.0, 6_000),  # risk exactly -4: CRITICAL
    cdm(1, 3.0, -6.0, 5_001, sigma=np.nan),                                         # risk exactly -6, one CDM, no sigma
    cdm(2, 2.5, -30.0, 1_000),                                                      # floor; miss exactly 1 km
    cdm(3, 2.0, -30.0, 5_000, "TBA"),                                               # miss exactly 5 km, t == 2
    cdm(4, 5.0, -12.0, 40_000, "UNKNOWN", sigma=1e6),                               # sigma >> d/sqrt(2): Pc diluted
    cdm(4, 1.5, -3.0, 200, "UNKNOWN"),                                              # only one CDM at t >= 2
    cdm(5, 1.0, -2.0, 50),                                                          # no CDM at the decision point
])
EXPECTED_STUB_TIERS = {0: "CRITICAL", 1: "ELEVATED", 2: "ELEVATED", 3: "ELEVATED", 4: "SAFE"}


def render(data_dir: Path, monkeypatch, **widgets) -> AppTest:
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    for label, value in widgets.items():
        next(w for w in at.multiselect if w.label == label).set_value(value)
    if widgets:
        at.run()
        assert not at.exception, at.exception
    return at


def queue(at: AppTest) -> pd.DataFrame:
    return at.dataframe[0].value


def page_text(at: AppTest) -> str:
    """Everything the page says: markdown cards, alerts and captions (layouts move text between them)."""
    return "\n".join(e.value for kind in (at.markdown, at.info, at.warning, at.error, at.success, at.caption)
                     for e in kind)


def stub_dir(tmp_path: Path, raw: pd.DataFrame = RAW) -> Path:
    raw.to_csv(tmp_path / "test_data.csv", index=False)
    return tmp_path


def test_stub_tiers_follow_the_rules_at_the_thresholds(tmp_path, monkeypatch):
    at = render(stub_dir(tmp_path), monkeypatch)
    assert "stub" in at.sidebar.caption[0].value
    assert at.metric[0].value == "5"  # event 5 has no CDM at time_to_tca >= 2, so it can't be scored
    q = queue(at).set_index("event_id")
    assert q.tier_final.to_dict() == EXPECTED_STUB_TIERS  # SAFE event 4 is queued: it needs review
    # Events 1-4 have one CDM up to the decision point (1 also lacks sigma); 0 has three.
    assert q.needs_review.to_dict() == {0: False, 1: True, 2: True, 3: True, 4: True}
    assert at.metric[4].value == "4"
    assert list(q.index[:1]) == [0]  # worst first
    assert q.pc_diluted.to_dict() == {0: False, 1: False, 2: False, 3: False, 4: True}  # NaN sigma (1) isn't flagged
    assert any(c.value.startswith("1 of 5 events are flagged") for c in at.caption)  # event 4
    # test data: no outcomes, so no verdict, and one hint for both outcome checks instead of two boxes
    assert page_text(at).count("need known outcomes: choose the train split") == 1
    assert any("No data/processed/metrics.json" in i.value for i in at.info)
    assert "Suggested maneuver" in page_text(at)  # event 0 is CRITICAL


def by_label(elements, label):
    return next(e for e in elements if e.label == label)


def write_pipeline(tmp_path: Path, drop: tuple = (), metrics: dict | None = None,
                   review: tuple = (False, True, False, False, True), train_in_sample: bool = True) -> Path:
    """Contract-shaped pipeline output for events 0-4, train outcomes included; drop removes columns."""
    raw = RAW[RAW.event_id != 5]
    cdms = raw.assign(split="test", tier="SAFE")
    cdms = pd.concat([cdms, cdms.assign(split="train")], ignore_index=True)
    latest = raw.loc[raw.groupby("event_id").time_to_tca.idxmin()]
    ev = pd.DataFrame({
        "event_id": latest.event_id, "c_object_type": latest.c_object_type, "n_cdms": 1,
        "time_to_tca": latest.time_to_tca, "risk_latest": latest.risk, "miss_latest_m": latest.miss_distance,
        "risk_forecast": latest.risk, "risk_forecast_q90": np.nan, "miss_forecast_m": latest.miss_distance,
        "p_critical": [0.9, np.nan, 0.1, np.nan, 0.0], "tier_now": "SAFE",
        "tier_final": ["CRITICAL", "ELEVATED", "SAFE", "CRITICAL", "SAFE"],
        "reason": [None, "x", None, "y", "z"], "drivers": [None, "", np.nan, "a; b; c", None],
        "forecast_source": "model", "risk_final": np.nan, "miss_final_m": np.nan,
        "pc_diluted": [False, False, True, False, True], "risk_potential": latest.risk + [0, 0, 3, 0, 2],
        # CRITICAL events only: event 0 needs a 250 m outward shift, event 3 is Critical by forecast alone
        "maneuver_dr_m": [250.0, np.nan, np.nan, 0.0, np.nan], "maneuver_dv_mps": [0.03, np.nan, np.nan, 0.0, np.nan],
        "needs_review": list(review), "review_reason": ["only one CDM" if r else "" for r in review],
        "risk_lo": -30.0, "risk_hi": latest.risk + 1, "p_critical_storm": [0.95, np.nan, 0.6, np.nan, 0.0],
        "storm_flips": [False, False, True, False, False],  # event 2: SAFE now, Critical in a storm
    })
    train = ev.assign(split="train", risk_final=-5.0, miss_final_m=800.0, forecast_in_sample=train_in_sample,
                      tier_actual=["CRITICAL", "SAFE", "SAFE", "CRITICAL", "SAFE"])
    events = (pd.concat([ev.assign(split="test", tier_actual=None, forecast_in_sample=False), train])
              .drop(columns=list(drop)))
    (tmp_path / "processed").mkdir(parents=True)
    events.to_parquet(tmp_path / "processed" / "events_scored.parquet")
    cdms.to_parquet(tmp_path / "processed" / "cdms.parquet")
    if metrics is not None:
        (tmp_path / "processed" / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path


def test_pipeline_output_with_missing_model_columns(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    assert "pipeline output (model)" in at.sidebar.caption[0].value
    # CRITICAL by p_critical (NaN last), then ELEVATED, then SAFE event 4 because it needs review
    assert list(queue(at).event_id) == [0, 3, 1, 4]
    assert "need known outcomes: choose the train split" in page_text(at)  # test split: outcomes unknown
    at.sidebar.selectbox[0].set_value("train").run()  # real outcomes drive the 24 h replay and the fatigue curve
    assert not at.exception, at.exception
    # Both actual Criticals (events 0 and 3) are the queue's first two rows.
    assert any("top 2 of 5" in c.value and "with the baseline ranking" in c.value for c in at.caption)
    assert any(w.value.startswith("In-sample") for w in at.warning)  # forecast_source == "model" on train
    # Flagged events 2 and 4 both ended SAFE: the dashboard must report the flag's failure, not praise it.
    text = page_text(at)
    assert "Measured failure: flagged events ended Critical no more often than the rest (0.0% vs 66.7%)" in text
    assert "Not a failure here" not in text


def test_brief_explains_the_selected_event(tmp_path, monkeypatch):
    monkeypatch.setattr(brief, "api_key", lambda *a: "k")
    monkeypatch.setattr(brief, "call_gemini", lambda prompt, key, model, timeout=20: f"FAKE BRIEF via {model}")
    at = render(write_pipeline(tmp_path), monkeypatch)
    by_label(at.button, "Explain this event").click().run()
    assert not at.exception, at.exception
    assert any(f"FAKE BRIEF via {brief.MODELS[0]}" in m.value for m in at.markdown)


def test_brief_refuses_without_calling_the_api(tmp_path, monkeypatch):
    monkeypatch.setattr(brief, "api_key", lambda *a: "k")  # call_gemini still raises if reached
    at = render(write_pipeline(tmp_path), monkeypatch)
    by_label(at.selectbox, "Event (worst first)").set_value(1).run()  # needs_review
    assert not at.exception, at.exception
    assert any(i.value.startswith("No brief: too little evidence") for i in at.info)
    assert not any(b.label == "Explain this event" for b in at.button)


def test_brief_without_a_key_says_so(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    assert any(i.value.startswith("Brief off") for i in at.info)


def test_brief_failure_is_shown_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(brief, "api_key", lambda *a: "k")

    def down(prompt, key, model, timeout=20):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(brief, "call_gemini", down)
    at = render(write_pipeline(tmp_path), monkeypatch)
    by_label(at.button, "Explain this event").click().run()
    assert any(e.value.startswith("Brief unavailable") and "offline" in e.value for e in at.error)


def sign(at: AppTest, analyst: str, decision: str, note: str = "") -> AppTest:
    by_label(at.text_input, "Analyst").input(analyst)
    by_label(at.radio, "Decision").set_value(decision)
    by_label(at.text_input, "Note").input(note)
    by_label(at.button, "Sign decision").click().run()
    assert not at.exception, at.exception
    return at


def test_signed_decision_is_logged_with_what_the_system_showed(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    assert at.metric[5].value == "4"  # queued: 0, 3, 1 (non-SAFE) and 4 (needs review)
    sign(at, "  Ada  ", "Maneuver", "burn planned")  # event 0, the worst, is selected by default
    log = pd.read_csv(tmp_path / "decisions.csv")
    assert log[["split", "event_id", "analyst", "decision", "note", "tier_final", "forecast_source"]].iloc[0].tolist() \
        == ["test", 0, "Ada", "Maneuver", "burn planned", "CRITICAL", "model"]
    assert queue(at).set_index("event_id").decision[0] == "Maneuver"
    assert at.metric[5].value == "3"
    at.sidebar.selectbox[0].set_value("train").run()  # decisions belong to one split
    assert queue(at).set_index("event_id").decision[0] == ""


def test_unsigned_decision_is_refused(tmp_path, monkeypatch):
    at = sign(render(write_pipeline(tmp_path), monkeypatch), "   ", "Monitor")
    assert any("every decision is signed" in e.value for e in at.error)
    assert not (tmp_path / "decisions.csv").exists()


def test_dismissing_a_critical_event_needs_a_reason(tmp_path, monkeypatch):
    at = sign(render(write_pipeline(tmp_path), monkeypatch), "Ada", "Dismiss")
    assert any("needs a note" in e.value for e in at.error)
    assert not (tmp_path / "decisions.csv").exists()
    sign(at, "Ada", "Dismiss", "debris catalogue error, confirmed with owner")
    assert pd.read_csv(tmp_path / "decisions.csv").decision.tolist() == ["Dismiss"]


def test_dismissing_a_safe_event_needs_no_reason(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    by_label(at.selectbox, "Event (worst first)").set_value(2).run()  # SAFE, enough evidence
    sign(at, "Ada", "Dismiss")
    assert pd.read_csv(tmp_path / "decisions.csv").event_id.tolist() == [2]


def test_queue_shows_pc_and_distances_not_raw_logs(tmp_path, monkeypatch):
    q = queue(render(write_pipeline(tmp_path), monkeypatch)).set_index("event_id")
    assert q.risk_forecast[0] == "1.0×10⁻⁴"  # exactly on the Critical threshold
    assert q.miss_latest_m[0] == "6,000 m" and q.miss_latest_m[3] == "5,000 m"  # never rounded onto 5 km
    assert q.risk_forecast[4] == "1.0×10⁻³"
    q_all = queue(render(write_pipeline(tmp_path / "all", review=(True,) * 5), monkeypatch)).set_index("event_id")
    assert q_all.risk_forecast[2] == "<10⁻³⁰"  # the floor reads as effectively zero, not -30
    assert q.time_to_tca[0] == "2 d 02 h"  # 2.1 days (narrow no-break spaces)


def event_charts(at: AppTest) -> list[dict]:
    specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    return [s for s in specs if s["layout"].get("xaxis", {}).get("title", {}).get("text") == "days to TCA"
            and "scene" not in s["layout"]]   # the two history charts (their titles are page labels now)


@pytest.mark.parametrize("event", [0, 2])  # 2 is Pc-diluted: the axis setup once only ran for diluted events
def test_event_history_counts_down_to_tca(tmp_path, monkeypatch, event):
    at = render(write_pipeline(tmp_path), monkeypatch)
    by_label(at.selectbox, "Event (worst first)").set_value(event).run()
    charts = event_charts(at)
    assert len(charts) == 2
    for spec in charts:
        assert spec["layout"]["xaxis"]["autorange"] == "reversed"
        assert spec["layout"]["xaxis"]["title"]["text"] == "days to TCA"
    pc = next(s for s in charts if s["layout"]["yaxis"]["title"]["text"].startswith("Collision"))["layout"]["yaxis"]
    assert all(t == "1" or t.startswith("10⁻") for t in pc["ticktext"]) and len(pc["ticktext"]) >= 4


def test_floor_events_are_counted_not_plotted(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)  # events 2 and 3 sit at the -30 floor
    assert any(c.value.startswith("2 events (40%) sit at the floor") for c in at.caption)


def test_queue_depth_headline_quotes_its_interval(tmp_path, monkeypatch):
    depth = {"model": 1198, "baseline": 1514, "reduction": 0.209, "ci95": {"reduction": [0.062, 0.373]}}
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN, "queue_depth": depth}), monkeypatch)
    at.sidebar.selectbox[0].set_value("train").run()
    text = page_text(at)  # the headline lives in Evaluation as a big-number card
    assert "Fewer reviews than today's ranking, at 95% recall" in text and ">21%<" in text
    assert "95% CI 6%–37%" in text


def test_no_queue_depth_no_headline(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN}), monkeypatch)
    at.sidebar.selectbox[0].set_value("train").run()
    assert "Fewer reviews" not in page_text(at)


def test_reason_lines_split_clauses_without_markup():
    from dashboard.sections.detail import reason_lines
    out = reason_lines("critical now (miss 202 m < 1,000 m); forecast Critical (P(crit) 99%); Pc falling")
    assert out == "Critical now (miss 202 m < 1,000 m)<br>Forecast Critical (P(crit) 99%)<br>Pc falling"
    assert reason_lines("") == "" and "<b>" not in out


def test_event_header_strip(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    text = page_text(at)  # event 0: CRITICAL, Pc exactly on the threshold, miss 6,000 m, 2.1 days out
    for shown in ("CRITICAL", "1.0×10⁻⁴", "6,000 m", "2 d 02 h", "90.0%", "Time to TCA",
                  "95% bound at TCA", "one-sided 95% upper bound on Pc"):
        assert shown in text, shown
    by_label(at.selectbox, "Event (worst first)").set_value(1).run()  # needs review: no forecast numbers
    assert page_text(at).count(">withheld</span>") == 2   # the two strip numbers, not other mentions


def test_evaluation_headline_compares_model_with_baseline(tmp_path, monkeypatch):
    depth = {"model": 1198, "baseline": 1514, "reduction": 0.209, "ci95": {"reduction": [0.062, 0.373]}}
    model = {**METRIC_RUN, "recall_critical": 0.95, "ci95": {"recall_critical": [0.93, 0.97]}}
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN, "model": model, "queue_depth": depth}),
                monkeypatch)
    text = page_text(at)
    assert ">21%<" in text and "95% CI 6%–37%" in text
    assert ">95.0%<" in text and "baseline 80.0% · 95% CI 93.0%–97.0%" in text
    assert "risk forecaster did not beat" in text  # same MSE as the baseline: the fallback is a failure to show


def test_no_regressor_failure_when_the_model_wins_or_is_absent(tmp_path, monkeypatch):
    better = {**METRIC_RUN, "mse_hr": 5.0}
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN, "model": better}), monkeypatch)
    assert "risk forecaster did not beat" not in page_text(at)
    at = render(write_pipeline(tmp_path / "b", metrics={"baseline": METRIC_RUN}), monkeypatch)
    assert "risk forecaster did not beat" not in page_text(at)


def test_out_of_fold_train_rows_are_not_called_in_sample(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, train_in_sample=False), monkeypatch)
    at.sidebar.selectbox[0].set_value("train").run()
    assert any("top 2 of 5" in c.value for c in at.caption)
    assert not any(w.value.startswith("In-sample") for w in at.warning)


def test_storm_stress_lists_the_flipped_events(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    assert "1 of 5 events change tier" in page_text(at)
    table = next(d.value for d in at.dataframe if "p_critical_storm" in d.value)
    assert list(table.event_id) == [2]


def test_pipeline_output_from_before_the_storm_and_conformal_columns(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, drop=("risk_lo", "risk_hi", "p_critical_storm", "storm_flips",
                                               "forecast_in_sample")), monkeypatch)
    assert "0 of 5 events change tier" in page_text(at)
    at.sidebar.selectbox[0].set_value("train").run()
    assert any(w.value.startswith("In-sample") for w in at.warning)  # old output: model rows were in-sample


def test_pipeline_output_from_before_the_dilution_flag(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, drop=("pc_diluted", "risk_potential")), monkeypatch)
    assert not queue(at).pc_diluted.any()
    at.sidebar.selectbox[0].set_value("train").run()
    assert not at.exception, at.exception
    text = page_text(at)  # nothing flagged, so no verdict either way
    assert "Measured failure: flagged" not in text and "Not a failure here" not in text


def test_maneuver_suggestion(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    shift = [m.value for m in at.markdown if "Suggested maneuver" in m.value]
    assert len(shift) == 1 and "250 m (outward)" in shift[0] and "3.0 cm/s" in shift[0]  # event 0, worst first
    by_label(at.selectbox, "Event (worst first)").set_value(3).run()
    assert "None from geometry" in page_text(at)  # CRITICAL, but dr == 0
    by_label(at.selectbox, "Event (worst first)").set_value(1).run()  # ELEVATED: no suggestion at all
    text = page_text(at)
    assert "Suggested maneuver" not in text
    assert "Insufficient evidence" in text and "only one CDM" in text  # and it needs review


def test_refusal_withholds_the_maneuver(tmp_path, monkeypatch):
    # Event 0 is CRITICAL with a 250 m plan, but on thin evidence: no plan may be shown.
    at = render(write_pipeline(tmp_path, review=(True, False, False, False, False)), monkeypatch)
    text = page_text(at)
    assert "Insufficient evidence" in text and "Suggested maneuver" not in text


def test_pipeline_output_from_before_the_refusal(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, drop=("needs_review", "review_reason")), monkeypatch)
    assert at.metric[4].value == "0" and not at.error
    assert list(queue(at).event_id) == [0, 3, 1]  # SAFE events stay out of the queue


def test_pipeline_output_from_before_the_maneuver_columns(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, drop=("maneuver_dr_m", "maneuver_dv_mps")), monkeypatch)
    assert "Suggested maneuver" not in page_text(at)


METRIC_RUN = {"recall_critical": 0.8, "precision_critical": 0.8, "f2": 0.4, "mse_hr": 8.0, "loss": 20.0,
              "n_critical": 551, "n_events": 8293, "ci95": {"recall_critical": [0.78, 0.85]}}


def metrics_panel(at: AppTest) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if d.value.index.name == "metric")


def test_metrics_baseline_only(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN}), monkeypatch)
    panel = metrics_panel(at)
    assert list(panel.columns) == ["baseline", "baseline 95% CI"]  # no model columns, no verdict
    assert panel.loc["Critical recall", "baseline 95% CI"] == "[0.78, 0.85]"
    assert panel.loc["F2 (high risk)", "baseline 95% CI"] == "—"  # CI missing (n_boot=0)
    assert any("No trained model yet" in i.value for i in at.info)


def test_metrics_model_vs_baseline_respects_direction(tmp_path, monkeypatch):
    # The model is better on recall (higher) and loss (lower), worse on precision and MSE, equal on F2.
    model = {**METRIC_RUN, "recall_critical": 0.95, "precision_critical": 0.7, "mse_hr": 9.0, "loss": 15.0}
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN, "model": model}), monkeypatch)
    verdict = metrics_panel(at)["model better?"].to_dict()
    assert verdict == {"Critical recall": "yes", "Critical precision": "no", "F2 (high risk)": "same",
                       "MSE on high-risk events": "no", "loss = MSE / F2": "yes"}
    assert not any("No trained model yet" in i.value for i in at.info)


def test_filters_that_match_nothing(tmp_path, monkeypatch):
    at = render(stub_dir(tmp_path), monkeypatch, **{"Tier (final)": []})
    assert at.metric[0].value == "0"
    assert queue(at).empty


def test_empty_data(tmp_path, monkeypatch):
    at = render(stub_dir(tmp_path, RAW.iloc[:0]), monkeypatch)
    assert at.metric[0].value == "0"


def test_real_test_data_if_present(monkeypatch):
    root = Path(__file__).resolve().parent.parent / "data"
    if not (root / "test_data.csv").exists():
        pytest.skip("data/test_data.csv missing")
    at = render(root, monkeypatch)
    assert int(at.metric[0].value.replace(",", "")) > 2000
    if (root / "processed" / "metrics.json").exists():
        assert 0 < metrics_panel(at).loc["Critical recall", "baseline"] <= 1
