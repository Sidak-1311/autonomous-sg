import numpy as np
import pandas as pd

from src.features import FEATURES, build_event_table
from src.ingest import validate
from src.physics import log10_pc_approx


def _cdm(event_id, ttca, risk, miss, split="train"):
    return {
        "split": split, "event_id": event_id, "time_to_tca": ttca, "risk": risk,
        "max_risk_estimate": risk + 1, "miss_distance": miss, "relative_speed": 10_000.0,
        "relative_position_r": 50.0, "relative_position_t": miss, "relative_position_n": 10.0,
        "relative_velocity_r": 1.0, "relative_velocity_t": 2.0, "relative_velocity_n": 3.0,
        "c_object_type": "DEBRIS", "c_sigma_r": 100.0, "t_sigma_r": 20.0,
        "F10": 80.0, "AP": 10.0,
    }


def _table(rows):
    df = pd.DataFrame(rows)
    split = df.pop("split")
    df = validate(df)
    df.insert(0, "split", split.loc[df.index])
    return build_event_table(df)


def test_uses_only_cdms_before_cutoff_and_targets_last_cdm():
    ev = _table([
        _cdm(1, 5.0, -8.0, 3_000),
        _cdm(1, 3.0, -6.0, 2_000),
        _cdm(1, 0.5, -3.5, 800),     # after the 2-day cutoff: target only
    ])
    row = ev.iloc[0]
    assert row.n_cdms == 2
    assert row.risk_latest == -6.0 and row.miss_latest_m == 2_000
    assert row.risk_slope > 0  # risk rose as TCA approached
    assert row.has_target and row.risk_final == -3.5 and row.miss_final_m == 800
    assert row.tier_now == "ELEVATED" and row.tier_actual == "CRITICAL"
    assert not ev[FEATURES].isna().any().any()


def test_test_split_and_far_final_have_no_target():
    ev = _table([
        _cdm(1, 4.0, -10.0, 9_000, split="test"),
        _cdm(1, 2.5, -9.0, 8_000, split="test"),
        _cdm(2, 4.0, -10.0, 9_000),
        _cdm(2, 2.5, -9.0, 8_000),   # last train CDM still 2.5 days out
    ]).set_index(["split", "event_id"])
    for key in [("test", 1), ("train", 2)]:
        assert not ev.loc[key, "has_target"]
        assert np.isnan(ev.loc[key, "risk_final"])
        assert ev.loc[key, "tier_actual"] is None


def test_pc_approx_monotone_in_miss_distance():
    pcs = log10_pc_approx(np.array([0, 100, 500, 5_000]), 100.0, 20.0)
    assert np.all(np.diff(pcs) <= 0)
    assert pcs[-1] == -30.0
