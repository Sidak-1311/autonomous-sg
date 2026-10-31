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
    return build_event_table(_cdm_frame(rows))


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


def test_test_split_has_no_target():
    ev = _table([
        _cdm(1, 4.0, -10.0, 9_000, split="test"),
        _cdm(1, 2.5, -9.0, 8_000, split="test"),
    ]).set_index(["split", "event_id"])
    row = ev.loc[("test", 1)]
    assert not row["has_target"]
    assert np.isnan(row["risk_final"]) and np.isnan(row["final_ttca"])
    assert row["tier_actual"] is None


def test_far_final_is_still_the_target():
    # Organisers' labels: the final state is the last CDM, even when it is
    # still >= 2 days out (then it equals the decision-point state)
    ev = _table([
        _cdm(2, 4.0, -10.0, 9_000),
        _cdm(2, 2.5, -9.0, 4_000),
    ]).set_index(["split", "event_id"])
    row = ev.loc[("train", 2)]
    assert row["has_target"] and row["final_ttca"] == 2.5
    assert row["risk_final"] == -9.0 and row["miss_final_m"] == 4_000
    assert row["tier_actual"] == row["tier_now"] == "ELEVATED"


def test_pc_approx_monotone_in_miss_distance():
    pcs = log10_pc_approx(np.array([0, 100, 500, 5_000]), 100.0, 20.0)
    assert np.all(np.diff(pcs) <= 0)
    assert pcs[-1] == -30.0


# --- edge cases -------------------------------------------------------------

def _cdm_frame(rows):
    df = pd.DataFrame(rows)
    split = df.pop("split")
    df = validate(df)
    df.insert(0, "split", split.loc[df.index])
    return df


def test_empty_input_gives_empty_table_with_all_columns():
    empty = build_event_table(_cdm_frame([_cdm(1, 3.0, -8.0, 3_000)]).iloc[0:0])
    assert len(empty) == 0
    assert set(FEATURES) <= set(empty.columns)


def test_all_cdms_after_cutoff_gives_empty_table():
    assert len(_table([_cdm(1, 1.0, -5.0, 900)])) == 0


def test_single_cdm_event_has_zero_trends():
    row = _table([_cdm(1, 3.0, -8.0, 3_000)]).iloc[0]
    assert row.n_cdms == 1
    assert row.risk_slope == 0 and row.risk_delta_last == 0 and row.risk_std == 0
    assert row.sigma_shrink == 0


def test_event_with_no_cdm_before_cutoff_is_dropped():
    ev = _table([_cdm(1, 1.5, -5.0, 900), _cdm(2, 3.0, -8.0, 3_000)])
    assert list(ev.event_id) == [2]


def test_floor_risk_flagged():
    row = _table([_cdm(1, 3.0, -30.0, 20_000)]).iloc[0]
    assert row.risk_is_floor == 1 and row.tier_now == "SAFE"


def test_missing_values_filled_and_flagged():
    a = _cdm(1, 3.0, -8.0, 3_000)
    b = _cdm(1, 2.5, -7.0, 2_500)
    a["c_sigma_r"] = np.nan
    a["F10"] = b["F10"] = np.nan
    a["AP"] = b["AP"] = np.nan
    other = _cdm(2, 3.0, -8.0, 3_000)
    ev = _table([a, b, other]).set_index("event_id")
    assert not ev[FEATURES].isna().any().any()
    assert ev.loc[1, "sw_missing"] == 1


def test_unknown_and_tba_object_types_map_to_unknown():
    a, b, c = _cdm(1, 3.0, -8.0, 3_000), _cdm(2, 3.0, -8.0, 3_000), _cdm(3, 3.0, -8.0, 3_000)
    a["c_object_type"], b["c_object_type"], c["c_object_type"] = "TBA", "ALIEN", None
    ev = _table([a, b, c])
    assert (ev.c_object_type == "UNKNOWN").all() and (ev.obj_UNKNOWN == 1).all()


def test_invalid_rows_dropped():
    good = _cdm(1, 3.0, -8.0, 3_000)
    bad_miss, bad_risk, bad_sigma = (_cdm(1, 2.8, -8.0, -5), _cdm(1, 2.6, 0.5, 100),
                                     _cdm(1, 2.4, -8.0, 100))
    bad_sigma["t_sigma_r"] = -1.0
    row = _table([good, bad_miss, bad_risk, bad_sigma]).iloc[0]
    assert row.n_cdms == 1


def test_tier_thresholds_exact_on_events():
    ev = _table([
        _cdm(1, 3.0, -4.0, 20_000), _cdm(2, 3.0, -6.0, 20_000),
        _cdm(3, 3.0, -30.0, 1_000), _cdm(4, 3.0, -30.0, 5_000), _cdm(5, 3.0, -30.0, 5_000.1),
    ]).set_index("event_id")
    assert list(ev.tier_now) == ["CRITICAL", "ELEVATED", "ELEVATED", "ELEVATED", "SAFE"]
