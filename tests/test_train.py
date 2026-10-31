import numpy as np
import pandas as pd
import pytest

from src import train
from src.features import build_event_table
from src.ingest import validate
from src.tiering import CRITICAL, RISK_FLOOR

OBJ = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA", None]


def _events(n=80, seed=0, critical=True):
    """Synthetic CDM histories: varied lengths (incl. single-CDM), floor risks, all object types."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n):
        n_cdm = 1 + e % 4
        risk0 = RISK_FLOOR if e % 7 == 0 else rng.uniform(-12, -3)
        miss0 = rng.uniform(300, 30_000)
        ttcas = list(np.linspace(6, 2.5, n_cdm)) + [0.3]   # last CDM after cutoff = target
        for k, t in enumerate(ttcas):
            last = k == len(ttcas) - 1
            risk = risk0 if not last else (risk0 + rng.normal(0, 1) if risk0 > RISK_FLOOR else RISK_FLOOR)
            if not critical:
                risk = min(risk, -7.0)
            rows.append({
                "event_id": e, "time_to_tca": t, "risk": float(np.clip(risk, RISK_FLOOR, 0)),
                "max_risk_estimate": float(np.clip(risk + 1, RISK_FLOOR, 0)),
                "miss_distance": miss0 if (critical or not last) else max(miss0, 6_000.0),
                "relative_speed": 10_000.0,
                "relative_position_r": 40.0, "relative_position_t": miss0, "relative_position_n": 5.0,
                "relative_velocity_r": 1.0, "relative_velocity_t": 2.0, "relative_velocity_n": 3.0,
                "c_object_type": OBJ[e % len(OBJ)],
                "c_sigma_r": np.nan if e % 5 == 0 else 80.0 / (k + 1),
                "t_sigma_r": 20.0, "F10": np.nan if e % 9 == 0 else 90.0, "AP": 12.0,
            })
    df = validate(pd.DataFrame(rows))
    df.insert(0, "split", "train")
    return build_event_table(df)


@pytest.fixture(autouse=True)
def fast_models(monkeypatch):
    monkeypatch.setattr(train, "MAX_ITER", 15)


def test_threshold_for_recall():
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    y = np.array([1, 1, 1, 1, 0], bool)
    t = train._threshold_for_recall(p, y, target=0.75)
    assert t == 0.2 and (p[y] >= t).mean() >= 0.75
    assert train._threshold_for_recall(p, np.zeros(5, bool)) == 0.5   # no positives


def test_train_and_predict_contract():
    ev = _events()
    bundle, cv = train.train(ev, n_folds=3)
    assert {"models", "medians", "crit_threshold", "use_risk_model", "use_miss_model"} <= set(bundle)
    assert cv["model"]["recall_critical"] >= cv["baseline"]["recall_critical"]   # worse() never lowers a tier

    test = ev.sample(25, random_state=1)   # non-default index must be kept
    out = train.predict(bundle, test)
    assert list(out.index) == list(test.index)
    assert list(out.columns) == ["risk_forecast", "risk_forecast_q90", "miss_forecast_m",
                                 "p_critical", "drivers"]
    assert out[["risk_forecast", "risk_forecast_q90", "miss_forecast_m", "p_critical"]].notna().all().all()
    assert (out["risk_forecast_q90"] >= out["risk_forecast"]).all()
    assert out["risk_forecast"].between(RISK_FLOOR, 0).all()
    assert out["p_critical"].between(0, 1).all() and (out["miss_forecast_m"] > 0).all()
    assert out["drivers"].map(lambda s: isinstance(s, str) and s.count(";") <= 2).all()


def test_fallback_uses_latest_values():
    ev = _events()
    bundle, _ = train.train(ev, n_folds=3)
    bundle.update(use_risk_model=False, use_miss_model=False)
    out = train.predict(bundle, ev)
    assert np.allclose(out["risk_forecast"], ev["risk_latest"])
    assert np.allclose(out["risk_forecast_q90"], ev["risk_latest"])
    assert np.allclose(out["miss_forecast_m"], ev["miss_latest_m"])


def test_no_critical_events_gives_constant_classifier():
    ev = _events(critical=False)
    assert not (ev["tier_actual"] == CRITICAL).any()
    bundle, _ = train.train(ev, n_folds=3)
    out = train.predict(bundle, ev)
    assert (out["p_critical"] == 0).all()


def test_predict_empty_input():
    ev = _events()
    bundle, _ = train.train(ev, n_folds=3)
    out = train.predict(bundle, ev.iloc[:0])
    assert out.empty and "p_critical" in out.columns


def test_floor_never_forecast_for_high_risk_events():
    ev = _events()
    bundle, _ = train.train(ev, n_folds=3)
    out = train.predict(bundle, ev)
    high = ev["risk_latest"] >= -6
    assert (out.loc[high, "risk_forecast"] > RISK_FLOOR).all()
