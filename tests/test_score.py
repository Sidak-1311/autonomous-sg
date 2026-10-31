import numpy as np
import pandas as pd
import pytest

from src import score as score_mod
from src.ingest import validate
from tests.test_features import _cdm


@pytest.fixture(autouse=True)
def baseline_only(monkeypatch, tmp_path):
    # Score with the latest-risk baseline even if a trained model exists
    monkeypatch.setattr(score_mod, "MODEL_PATH", tmp_path / "missing.joblib")


def _score(rows):
    df = pd.DataFrame(rows)
    split = df.pop("split")
    df = validate(df)
    df.insert(0, "split", split.loc[df.index])
    return score_mod.score(df).set_index("event_id")


def test_columns_match_contract_and_empty_input():
    ev = _score([_cdm(1, 3.0, -8.0, 3_000)])
    assert sorted(ev.reset_index().columns) == sorted(score_mod.EVENT_COLS)
    empty = _score([_cdm(1, 1.0, -8.0, 3_000)])   # no CDM before the cutoff
    assert len(empty) == 0


def test_single_cdm_needs_review_and_is_refused():
    ev = _score([_cdm(1, 3.0, -30.0, 20_000)])
    row = ev.loc[1]
    assert row.needs_review and "only one CDM" in row.review_reason
    assert row.reason.startswith("INSUFFICIENT EVIDENCE")
    assert row.tier_final == "SAFE"   # rules about the present still apply
    assert row.forecast_source == "withheld" and np.isnan(row.p_critical)
    assert row.risk_forecast == row.risk_latest and row.drivers == ""


def test_stale_and_missing_covariance_need_review():
    a1, a2 = _cdm(1, 6.5, -8.0, 9_000), _cdm(1, 5.5, -8.0, 9_000)
    b1, b2 = _cdm(2, 3.0, -8.0, 9_000), _cdm(2, 2.5, -8.0, 9_000)
    b2["t_sigma_r"] = np.nan
    ev = _score([a1, a2, b1, b2])
    assert ev.loc[1, "needs_review"] and "stale" in ev.loc[1, "review_reason"]
    assert ev.loc[2, "needs_review"] and "covariance missing" in ev.loc[2, "review_reason"]


def test_well_observed_event_does_not_need_review():
    ev = _score([_cdm(1, 4.0, -8.0, 9_000), _cdm(1, 3.0, -7.5, 8_000), _cdm(1, 2.2, -7.0, 7_000)])
    assert not ev.loc[1, "needs_review"] and ev.loc[1, "review_reason"] == ""
    assert not ev.loc[1, "reason"].startswith("INSUFFICIENT")


def test_maneuver_only_for_critical_and_thresholds_exact():
    ev = _score([
        _cdm(1, 3.0, -4.0, 20_000), _cdm(1, 2.5, -4.0, 20_000),     # Pc exactly 1e-4
        _cdm(2, 3.0, -30.0, 1_000), _cdm(2, 2.5, -30.0, 1_000),     # exactly 1 km
        _cdm(3, 3.0, -30.0, 999), _cdm(3, 2.5, -30.0, 999),
    ])
    assert ev.loc[1, "tier_final"] == "CRITICAL" and ev.loc[1, "maneuver_dv_mps"] >= 0
    assert ev.loc[2, "tier_final"] == "ELEVATED" and np.isnan(ev.loc[2, "maneuver_dv_mps"])
    assert ev.loc[3, "tier_final"] == "CRITICAL" and ev.loc[3, "maneuver_dr_m"] != 0


def test_dilution_note_is_not_a_warning():
    rows = [_cdm(1, 3.0, -12.0, 20_000), _cdm(1, 2.5, -12.0, 20_000)]
    for r in rows:
        r["c_sigma_r"], r["t_sigma_r"] = 300_000.0, 300_000.0
    row = _score(rows).loc[1]
    assert row.pc_diluted and "not a risk signal" in row.reason
    assert "could reach" not in row.reason and row.tier_final == "SAFE"


def test_queue_sorted_worst_first():
    ev = _score([_cdm(1, 3.0, -30.0, 20_000), _cdm(2, 3.0, -3.0, 500), _cdm(3, 3.0, -5.0, 9_000)])
    assert list(ev.tier_final) == ["CRITICAL", "ELEVATED", "SAFE"]


def test_reason_uses_scientific_notation_and_metres():
    ev = _score([_cdm(1, 4.0, -5.0, 1_500), _cdm(1, 3.0, -4.5, 1_200), _cdm(1, 2.2, -4.2, 999)])
    reason = ev.loc[1, "reason"]
    assert "Critical now (miss 999 m < 1,000 m)" in reason
    assert "1e-" not in reason and "orders of magnitude per day" in reason
