import numpy as np
import pandas as pd

from src.error_analysis import analyse, final_outcomes, lockbox_outcomes, miss_watch_tradeoff
from src.tiering import CRITICAL, ELEVATED, SAFE

THR = 0.03


def _cdms(rows):
    """rows: (split, event_id, time_to_tca, risk, miss_m)"""
    df = pd.DataFrame(rows, columns=["split", "event_id", "time_to_tca", "risk", "miss_distance"])
    return df


def _ev(rows):
    """rows: (event_id, tier_final, tier_now, risk_latest, miss_latest_m, p_critical, needs_review)"""
    df = pd.DataFrame(rows, columns=["event_id", "tier_final", "tier_now", "risk_latest",
                                     "miss_latest_m", "p_critical", "needs_review"])
    df["time_to_tca"], df["n_cdms"], df["c_object_type"] = 2.5, 3, "DEBRIS"
    df["miss_forecast_m"] = df["miss_latest_m"]
    return df


def test_final_outcome_is_smallest_ttca_and_ignores_test_split():
    c = _cdms([("train", 1, 3.0, -3.0, 500.0), ("train", 1, 0.2, -20.0, 20_000.0),
               ("test", 1, 0.1, -3.0, 100.0), ("train", 2, 2.5, -30.0, 900.0)])
    f = final_outcomes(c).set_index("event_id")
    assert f.loc[1, "tier_true"] == SAFE and f.loc[1, "final_ttca"] == 0.2
    assert f.loc[2, "tier_true"] == CRITICAL       # single CDM, floor risk, miss < 1 km
    assert len(f) == 2


def test_threshold_boundaries_in_final_tier():
    c = _cdms([("train", 1, 0.5, -4.0, 9_000.0), ("train", 2, 0.5, -6.0, 9_000.0),
               ("train", 3, 0.5, -20.0, 1_000.0), ("train", 4, 0.5, -20.0, 5_000.0),
               ("train", 5, 0.5, -20.0, 5_001.0)])
    t = final_outcomes(c).set_index("event_id")["tier_true"]
    assert list(t) == [CRITICAL, ELEVATED, ELEVATED, ELEVATED, SAFE]


def test_causes_and_confusion_matrix():
    c = _cdms([
        ("train", 1, 0.3, -3.0, 2_000.0),     # Critical, caught
        ("train", 2, 0.3, -5.0, 400.0),       # Critical by miss only, missed
        ("train", 3, 0.3, -3.0, 9_000.0),     # Critical, decision at floor, missed
        ("train", 4, 0.3, -3.5, 9_000.0),     # Critical, withheld for review
        ("train", 5, 0.3, -20.0, 9_000.0),    # Safe, false alarm: Pc fell
        ("train", 6, 0.3, -20.0, 9_000.0),    # Safe, correct
    ])
    ev = _ev([
        (1, CRITICAL, CRITICAL, -3.2, 2_100.0, 0.9, False),
        (2, ELEVATED, ELEVATED, -5.5, 3_000.0, 0.01, False),
        (3, SAFE, SAFE, -30.0, 9_000.0, 0.001, False),
        (4, SAFE, SAFE, -10.0, 9_000.0, np.nan, True),
        (5, CRITICAL, CRITICAL, -3.0, 9_000.0, 0.8, False),
        (6, SAFE, SAFE, -25.0, 9_000.0, 0.0, False),
    ])
    r = analyse(ev, final_outcomes(c), THR)
    assert r["n_events"] == 6 and r["n_critical"] == 4
    assert r["confusion_matrix"]["counts"] == [[1, 0, 1], [0, 0, 0], [2, 1, 1]]
    assert r["recall_critical"] == 0.25
    assert r["silent_missed_critical"] == 2
    assert r["recall_critical_incl_review"] == 0.5
    m = r["missed_critical"]["by_cause"]
    assert m["withheld_for_review"]["n"] == 1
    assert m["miss_driven"]["n"] == 1
    assert m["no_evidence_after_cutoff"]["n"] == 1
    assert sum(v["n"] for v in m.values()) == 3
    fa = r["false_alarm"]["by_cause"]
    assert fa["critical_now"]["n"] == 1 and fa["pc_fell"]["flag_n"] == 1
    assert {e["event_id"] for e in r["missed_examples"]} == {2, 3, 4}


def test_near_threshold_flag_boundaries():
    c = _cdms([("train", i, 0.3, -3.0, 9_000.0) for i in (1, 2, 3)])
    ev = _ev([(1, SAFE, SAFE, -4.5, 9_000.0, THR / 2, False),        # exactly at half: near
              (2, SAFE, SAFE, -4.5, 9_000.0, THR / 2 - 1e-6, False),  # just below: not near
              (3, ELEVATED, ELEVATED, -4.5, 9_000.0, THR - 1e-9, False)])
    r = analyse(ev, final_outcomes(c), THR)
    m = r["missed_critical"]["by_cause"]
    assert m["near_threshold"]["n"] == 2 and m["other"]["n"] == 1


def test_empty_and_unknown_object_type():
    c = _cdms([("train", 1, 0.3, -20.0, 9_000.0)])
    ev = _ev([(1, SAFE, SAFE, -20.0, 9_000.0, 0.0, False)])
    r = analyse(ev, final_outcomes(c), THR)
    assert r["n_critical"] == 0 and np.isnan(r["recall_critical"])
    assert r["missed_critical"]["n"] == 0 and r["missed_examples"] == []
    ev2 = _ev([(1, CRITICAL, CRITICAL, -3.0, 500.0, 0.9, False)])
    ev2["c_object_type"] = None
    r2 = analyse(ev2, final_outcomes(c), THR)
    assert r2["false_alarm"]["by_object_type"] == {"UNKNOWN": 1}
    r3 = analyse(ev.iloc[:0], final_outcomes(c), THR)
    clash = analyse(ev.assign(tier_true=CRITICAL), final_outcomes(c), THR)   # finals win over ev columns
    assert clash["n_critical"] == 0
    assert r3["n_events"] == 0


def test_miss_watch_tradeoff_counts_only_unissued_and_falls_back_to_latest():
    d = pd.DataFrame({
        "tier_true": [CRITICAL, CRITICAL, SAFE, SAFE, CRITICAL],
        "tier_final": [SAFE, CRITICAL, SAFE, SAFE, ELEVATED],
        "miss_forecast_m": [1_800.0, 100.0, 1_999.0, 2_000.0, np.nan],
        "miss_latest_m": [9_000.0, 100.0, 9_000.0, 9_000.0, 1_200.0],
    })
    t = {r["miss_below_m"]: r for r in miss_watch_tradeoff(d, [1_500.0, 2_000.0])}
    assert t[1_500.0]["recovered_critical"] == 1 and t[1_500.0]["extra_alerts"] == 0   # NaN -> latest 1,200
    assert t[2_000.0]["recovered_critical"] == 2 and t[2_000.0]["extra_alerts"] == 1   # 2,000 not < 2,000
    assert t[2_000.0]["alerts_per_recovered"] == 0.5
    assert miss_watch_tradeoff(d.iloc[:0], [1_000.0])[0]["alerts_per_recovered"] is None


def test_lockbox_outcomes_reads_true_risk_or_risk():
    raw = pd.DataFrame({"event_id": [7, 8], "time_to_tca": [0.3, 0.1],
                        "true_risk": [-3.9, -30.0], "miss_distance": [9_000.0, 999.0]})
    f = lockbox_outcomes(raw).set_index("event_id")
    assert f.loc[7, "tier_true"] == CRITICAL and f.loc[8, "tier_true"] == CRITICAL
    g = lockbox_outcomes(raw.rename(columns={"true_risk": "risk"}))
    assert list(g["tier_true"]) == list(f["tier_true"])
