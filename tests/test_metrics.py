import numpy as np
import pytest

from src.metrics import evaluate
from src.tiering import CRITICAL, ELEVATED, SAFE

# Truth: one critical by Pc, one critical by miss, one elevated, one safe
RISK_FINAL = [-3.0, -10.0, -5.0, -20.0]
MISS_FINAL = [5_000.0, 500.0, 3_000.0, 20_000.0]


def test_perfect_prediction():
    m = evaluate(RISK_FINAL, MISS_FINAL, RISK_FINAL,
                 [CRITICAL, CRITICAL, ELEVATED, SAFE], n_boot=0)
    assert m["recall_critical"] == 1.0 and m["precision_critical"] == 1.0
    assert m["f2"] == 1.0 and m["mse_hr"] == 0.0 and m["loss"] == 0.0
    assert m["n_critical"] == 2 and m["n_events"] == 4


def test_missed_critical_and_false_alarm():
    risk_pred = [-3.5, -10.0, -7.0, -5.0]   # 2 high-risk true, preds: hit, miss, FP
    m = evaluate(RISK_FINAL, MISS_FINAL, risk_pred,
                 [CRITICAL, SAFE, ELEVATED, CRITICAL], n_boot=0)
    assert m["recall_critical"] == 0.5 and m["precision_critical"] == 0.5
    # high-risk: true {0, 2}, pred {0, 3} -> p = r = 0.5 -> f2 = 0.5
    assert m["f2"] == pytest.approx(0.5)
    assert m["mse_hr"] == pytest.approx((0.5 ** 2 + 2.0 ** 2) / 2)
    assert m["loss"] == pytest.approx(m["mse_hr"] / 0.5)


def test_predictions_clipped_and_zero_f2_gives_inf_loss():
    m = evaluate([-3.0], [5_000.0], [-50.0], [SAFE], n_boot=0)
    assert m["mse_hr"] == pytest.approx(27.0 ** 2)   # -50 clipped to -30
    assert m["f2"] == 0.0 and np.isinf(m["loss"])


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    risk_final = rng.uniform(-12, -2, 300)
    miss = rng.uniform(200, 20_000, 300)
    risk_pred = risk_final + rng.normal(0, 1, 300)
    tier_pred = np.where(risk_pred >= -4, CRITICAL, SAFE)
    m = evaluate(risk_final, miss, risk_pred, tier_pred, n_boot=200)
    for k in ["recall_critical", "precision_critical", "f2", "mse_hr", "loss"]:
        lo, hi = m["ci95"][k]
        assert lo <= m[k] <= hi


def test_empty_input_does_not_crash():
    m = evaluate([], [], [], [], n_boot=100)
    assert m["n_events"] == 0 and m["n_critical"] == 0 and m["ci95"] == {}


def test_values_exactly_on_thresholds():
    # Pc exactly 1e-4 and miss just under 1 km are Critical; exactly 1 km and 5 km are not
    risk_final = [-4.0, -10.0, -10.0, -10.0, -6.0]
    miss_final = [50_000.0, 999.9, 1_000.0, 5_000.0, 50_000.0]
    m = evaluate(risk_final, miss_final, risk_final,
                 [CRITICAL, CRITICAL, ELEVATED, ELEVATED, ELEVATED], n_boot=0)
    assert m["n_critical"] == 2 and m["recall_critical"] == 1.0 and m["precision_critical"] == 1.0
    # risk exactly -6 counts as high risk; exactly -4 too
    assert m["f2"] == 1.0 and m["mse_hr"] == 0.0


def test_floor_risk_and_nan_predictions():
    m = evaluate([-30.0, -30.0], [20_000.0, 20_000.0], [-30.0, -30.0], [SAFE, SAFE], n_boot=0)
    assert m["n_critical"] == 0 and np.isnan(m["recall_critical"]) and np.isnan(m["mse_hr"])
