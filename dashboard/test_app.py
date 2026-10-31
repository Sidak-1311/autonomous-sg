"""Verification loop for the dashboard: run `pytest dashboard -q` after every change to dashboard/app.py.

Each test renders the whole app headless (every tab runs on each rerun) against a small synthetic data folder
(DATA_DIR), so it needs neither the real CSVs nor the pipeline. Edge cases: values exactly on the tier
thresholds, the -30 risk floor, single-CDM events, events with no CDM at the decision point, unknown object
types, missing model columns (NaN p_critical / drivers / reason), empty data and filters that match nothing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent / "app.py")
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


def stub_dir(tmp_path: Path, raw: pd.DataFrame = RAW) -> Path:
    raw.to_csv(tmp_path / "test_data.csv", index=False)
    return tmp_path


def test_stub_tiers_follow_the_rules_at_the_thresholds(tmp_path, monkeypatch):
    at = render(stub_dir(tmp_path), monkeypatch)
    assert "stub" in at.sidebar.caption[0].value
    assert at.metric[0].value == "5"  # event 5 has no CDM at time_to_tca >= 2, so it can't be scored
    q = queue(at).set_index("event_id")
    assert q.tier_final.to_dict() == {k: v for k, v in EXPECTED_STUB_TIERS.items() if v != "SAFE"}
    assert list(q.index[:1]) == [0]  # worst first
    assert not q.pc_diluted.any()  # NaN sigma (event 1) is not flagged
    assert any(c.value.startswith("1 of 5 events are flagged") for c in at.caption)  # event 4
    assert any("can't be checked" in i.value for i in at.info)  # test data: no outcomes, no verdict
    assert any("No data/processed/metrics.json" in i.value for i in at.info)
    assert any(w.value.startswith("Suggested maneuver") for w in [*at.warning, *at.info])  # event 0 is CRITICAL


def by_label(elements, label):
    return next(e for e in elements if e.label == label)


def write_pipeline(tmp_path: Path, drop: tuple = (), metrics: dict | None = None) -> Path:
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
    })
    train = ev.assign(split="train", risk_final=-5.0, miss_final_m=800.0,
                      tier_actual=["CRITICAL", "SAFE", "SAFE", "CRITICAL", "SAFE"])
    events = pd.concat([ev.assign(split="test", tier_actual=None), train]).drop(columns=list(drop))
    (tmp_path / "processed").mkdir()
    events.to_parquet(tmp_path / "processed" / "events_scored.parquet")
    cdms.to_parquet(tmp_path / "processed" / "cdms.parquet")
    if metrics is not None:
        (tmp_path / "processed" / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path


def test_pipeline_output_with_missing_model_columns(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    assert "pipeline output (model)" in at.sidebar.caption[0].value
    assert list(queue(at).event_id) == [0, 3, 1]  # CRITICAL by p_critical (NaN last), then ELEVATED
    assert not any("Alert fatigue" in h.value for h in at.subheader)  # test split: outcomes unknown
    at.sidebar.selectbox[0].set_value("train").run()  # real outcomes drive the 24 h replay and the fatigue curve
    assert not at.exception, at.exception
    # Both actual Criticals (events 0 and 3) are the queue's first two rows.
    assert any("top 2 of 5" in c.value for c in at.caption)
    # Flagged events 2 and 4 both ended SAFE: the dashboard must report the flag's failure, not praise it.
    assert any("Measured failure" in w.value and "0.0% vs 66.7%" in w.value for w in at.warning)
    assert not at.success


def test_pipeline_output_from_before_the_dilution_flag(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, drop=("pc_diluted", "risk_potential")), monkeypatch)
    assert not queue(at).pc_diluted.any()
    at.sidebar.selectbox[0].set_value("train").run()
    assert not at.exception, at.exception
    assert not any("Measured failure" in w.value for w in at.warning) and not at.success  # nothing flagged


def test_maneuver_suggestion(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    shift = [w.value for w in at.warning if w.value.startswith("Suggested maneuver")]
    assert len(shift) == 1 and "250 m (outward)" in shift[0] and "3.0 cm/s" in shift[0]  # event 0, worst first
    by_label(at.selectbox, "Event (worst first)").set_value(3).run()
    assert any("none from geometry" in i.value for i in at.info)  # CRITICAL, but dr == 0
    by_label(at.selectbox, "Event (worst first)").set_value(1).run()  # ELEVATED: no suggestion at all
    assert not any("Suggested maneuver" in m.value for m in [*at.warning, *at.info])


def test_pipeline_output_from_before_the_maneuver_columns(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, drop=("maneuver_dr_m", "maneuver_dv_mps")), monkeypatch)
    assert not any("Suggested maneuver" in m.value for m in [*at.warning, *at.info])


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
    assert verdict == {"Critical recall": "yes", "Critical precision": "no", "F2 (high risk)": "no",
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
