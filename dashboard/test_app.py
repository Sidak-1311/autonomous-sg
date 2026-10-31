"""Verification loop for the dashboard: run `pytest dashboard -q` after every change to dashboard/app.py.

Each test renders the whole app headless (every tab runs on each rerun) against a small synthetic data folder
(DATA_DIR), so it needs neither the real CSVs nor the pipeline. Edge cases: values exactly on the tier
thresholds, the -30 risk floor, single-CDM events, events with no CDM at the decision point, unknown object
types, missing model columns (NaN p_critical / drivers / reason), empty data and filters that match nothing.
"""
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


def write_pipeline(tmp_path: Path, drop: tuple = ()) -> Path:
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
    })
    train = ev.assign(split="train", risk_final=-5.0, miss_final_m=800.0,
                      tier_actual=["CRITICAL", "SAFE", "SAFE", "CRITICAL", "SAFE"])
    events = pd.concat([ev.assign(split="test", tier_actual=None), train]).drop(columns=list(drop))
    (tmp_path / "processed").mkdir()
    events.to_parquet(tmp_path / "processed" / "events_scored.parquet")
    cdms.to_parquet(tmp_path / "processed" / "cdms.parquet")
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
    assert not at.warning and not at.success  # nothing flagged, so no verdict either way


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
