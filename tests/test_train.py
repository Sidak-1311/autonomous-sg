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
                                 "p_safe", "p_elevated", "p_critical", "drivers"]
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


def test_conformal_interval_coverage_and_clipping():
    ev = _events(n=200)
    calib = train.calibrate_interval(ev, alpha=0.2, n_folds=3)
    assert 0.6 <= calib["coverage"] <= 1.0   # small sample, so a loose check on 80%
    iv = train.risk_interval(ev, calib)
    assert list(iv.index) == list(ev.index)
    assert (iv["risk_lo"] <= iv["risk_hi"]).all()
    assert iv["risk_lo"].between(RISK_FLOOR, 0).all() and iv["risk_hi"].between(RISK_FLOOR, 0).all()


def test_conformal_small_groups_use_pooled_bounds():
    ev = _events(n=40)
    calib = train.calibrate_interval(ev, n_folds=3)
    assert calib["groups"] == {}   # every tier_now group under INTERVAL_MIN_N
    iv = train.risk_interval(ev, calib)
    lo = np.clip(ev["risk_latest"] + calib["pooled"]["lo"], RISK_FLOOR, 0)
    assert np.allclose(iv["risk_lo"], lo)
    assert train.risk_interval(ev.iloc[:0], calib).empty


def test_storm_stress_raises_activity_only():
    ev = _events()
    bundle, _ = train.train(ev, n_folds=3)
    levels = train.storm_levels(ev)
    assert levels["F10"] == ev["F10"].quantile(0.99)
    out = train.storm_stress(bundle, ev, levels)
    assert list(out.columns) == ["p_critical_storm", "storm_flips"]
    assert out["p_critical_storm"].between(0, 1).all()
    # With storm levels below every event's own values nothing changes
    calm = {c: -1.0 for c in levels}
    same = train.storm_stress(bundle, ev, calm)
    assert np.allclose(same["p_critical_storm"], train.predict(bundle, ev)["p_critical"])
    assert not same["storm_flips"].any()
    assert train.storm_stress(bundle, ev.iloc[:0], levels).empty


def test_oof_forecast_replaces_only_fitted_train_rows():
    ev = _events()
    bundle, cv = train.train(ev, n_folds=3)
    oof = cv["oof_table"]
    assert len(oof) == ev["has_target"].sum() and oof["event_id"].is_unique
    assert oof[train.OOF_COLS].notna().all().all()

    both = pd.concat([ev, ev.assign(split="test")], ignore_index=True)   # same ids in test
    fc = train.predict(bundle, both)
    out = train.oof_forecast(fc, both, oof)
    is_train = both["split"] == "train"
    want = oof.set_index("event_id").loc[both.loc[is_train, "event_id"], "p_critical"].to_numpy()
    assert np.allclose(out.loc[is_train, "p_critical"], want)
    assert out.loc[~is_train].equals(fc.loc[~is_train])            # test rows untouched
    assert (out["drivers"] == fc["drivers"]).all()
    assert train.oof_forecast(fc.iloc[:0], both.iloc[:0], oof).empty


def _miscalibrated(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.beta(0.3, 3, n)
    y = rng.random(n) < true_p
    return true_p ** 0.3, y   # overconfident, but rank-preserving


def test_isotonic_map_is_strictly_increasing_and_bounded():
    p, y = _miscalibrated()
    m = train._isotonic_map(p, y)
    assert np.all(np.diff(m["x"]) > 0) and np.all(np.diff(m["y"]) > 0)
    q = train.calibrate_p(np.r_[-1.0, 0.0, p, 1.0, 2.0], m)
    assert q.min() >= 0 and q.max() <= 1 and np.all(np.diff(np.sort(q)) >= 0)
    assert np.array_equal(train.calibrate_p(p, None), p)   # no map = identity


def test_fit_calibration_improves_brier_and_keeps_ranking_and_recall():
    p, y = _miscalibrated()
    cal = train.fit_calibration(p, y, np.arange(len(p)), n_folds=3)
    assert cal["brier_calibrated"] < cal["brier_raw"]
    q = cal["p_calibrated"]
    inside = (p > cal["map"]["x"][0]) & (p < cal["map"]["x"][-1])   # tails are clipped flat
    assert np.array_equal(np.argsort(p[inside]), np.argsort(q[inside]))
    assert (q[y] >= cal["crit_threshold"]).mean() >= 0.95
    assert sum(r["n"] for r in cal["reliability_calibrated"]) == len(p)


def test_apply_calibration_is_rerunnable():
    p, y = _miscalibrated(n=300)
    oof = pd.DataFrame({"event_id": np.arange(300), "p_critical": p})
    cal = train.fit_calibration(p, y, oof["event_id"].to_numpy(), n_folds=3)
    bundle, out = train.apply_calibration({"crit_threshold": 0.1}, oof, cal)
    assert np.array_equal(out["p_critical_raw"], p) and bundle["crit_threshold_raw"] == cal["crit_threshold_raw"]
    _, again = train.apply_calibration(bundle, out, cal)
    assert np.array_equal(again["p_critical_raw"], p)   # raw is never overwritten


def test_predict_and_storm_use_calibration():
    ev = _events()
    bundle, _ = train.train(ev, n_folds=3)
    raw = train.predict(bundle, ev)["p_critical"].to_numpy()
    cal = {"x": [0.0, 1.0], "y": [0.0, 0.5]}   # halves every score
    out = train.predict({**bundle, "calibration": cal}, ev)
    assert np.allclose(out["p_critical"], raw / 2)
    storm = train.storm_stress({**bundle, "calibration": cal}, ev, {c: -1.0 for c in train.STORM_COLS})
    assert np.allclose(storm["p_critical_storm"], raw / 2)


def test_calibration_never_claims_certainty():
    p = np.linspace(0, 1, 400)
    y = p > 0.5                        # perfectly separable: plain isotonic maps the ends to exactly 0 and 1
    m = train._isotonic_map(p, y)
    q = train.calibrate_p(np.r_[0.0, p, 1.0], m)
    assert 0 < q.min() and q.max() < 1
    assert np.all(np.diff(m["y"]) >= 0)
    one = train._isotonic_map(np.array([0.2, 0.4]), np.array([True, True]))   # a single block
    assert 0 < one["y"][0] < 1


def test_holdout_mask_splits_by_event_and_is_deterministic():
    ev = _events(n=50)
    m = train.holdout_mask(ev)
    assert m.sum() == 10 and m.equals(train.holdout_mask(ev))   # 20%, same every run
    assert not set(ev.loc[m, "event_id"]) & set(ev.loc[~m, "event_id"])
    assert not train.holdout_mask(ev.iloc[:1]).any()             # one event: nothing to hold out
    assert not train.holdout_mask(ev.iloc[:0]).any()


def test_evaluate_holdout_scores_only_unseen_events():
    ev = _events(n=100)
    test, m = train.evaluate_holdout(ev, n_folds=3)
    held = ev[ev["has_target"]]
    assert set(test["event_id"]) == set(held.loc[train.holdout_mask(held.reset_index(drop=True)).to_numpy(),
                                                  "event_id"])
    assert list(test.columns) == train.HOLDOUT_COLS
    assert m["split"]["n_dev"] + m["split"]["n_holdout"] == len(held)
    assert test["p_critical"].between(0, 1).all()
    assert m["model"]["recall_critical"] >= m["baseline"]["recall_critical"]   # worse() never lowers a tier


def test_tier_probs_sum_to_one_and_never_negative():
    pc = np.array([0.0, 0.2, 0.9, 0.5, 1.0])
    pe = np.array([0.0, 0.7, 0.95, 0.3, 1.0])   # 4th: ordinal scores cross
    p = train.tier_probs(pc, pe)
    total = p["p_safe"] + p["p_elevated"] + p["p_critical"]
    assert np.allclose(total, 1) and all((v >= 0).all() for v in p.values())
    assert p["p_elevated"][3] == 0 and p["p_critical"][3] == 0.5
    assert np.allclose(p["p_elevated"][:3], [0.0, 0.5, 0.05])


def test_predict_tier_probabilities_after_calibration():
    ev = _events(n=100)
    bundle, cv = train.train(ev, n_folds=3)
    fitted = ev[ev["has_target"]]
    bundle, oof = train.apply_calibration(bundle, cv["oof_table"], *train.calibrate_both(cv["oof_table"], fitted, 3))
    assert {"calibration", "calibration_elev"} <= set(bundle)
    assert np.allclose(oof[["p_safe", "p_elevated", "p_critical"]].sum(axis=1), 1)
    out = train.predict(bundle, ev.head(30))
    probs = out[["p_safe", "p_elevated", "p_critical"]]
    assert np.allclose(probs.sum(axis=1), 1) and (probs >= 0).all().all()
    # old bundles without the Elevated classifier still predict P(crit)
    old = {**bundle, "models": {k: v for k, v in bundle["models"].items() if k != "clf_elev"}}
    old.pop("calibration_elev")
    assert train.predict(old, ev.head(5))["p_critical"].notna().all()


def test_models_never_early_stop():
    # The final fit (all events) must match the CV folds' capacity, or the
    # calibration fit on OOF scores doesn't apply to it
    assert train._clf().early_stopping is False
    assert train._reg(loss="absolute_error").early_stopping is False


def test_oof_forecast_swaps_all_three_tier_probabilities():
    # Regression: only p_critical was swapped, so train rows' three probabilities didn't sum to 1
    import pandas as pd
    from src import train as tr
    ev = pd.DataFrame({"split": ["train", "test"], "event_id": [1, 1]})
    fc = pd.DataFrame({"risk_forecast": [-5.0, -5.0], "risk_forecast_q90": [-4.0, -4.0],
                       "miss_forecast_m": [900.0, 900.0], "p_safe": [0.1, 0.1], "p_elevated": [0.1, 0.1],
                       "p_critical": [0.8, 0.8]})
    oof = pd.DataFrame({"event_id": [1], "risk_forecast": [-6.0], "risk_forecast_q90": [-5.0],
                        "miss_forecast_m": [2_000.0], "p_safe": [0.7], "p_elevated": [0.2], "p_critical": [0.1]})
    out = tr.oof_forecast(fc, ev, oof)
    assert list(out.loc[0, ["p_safe", "p_elevated", "p_critical"]]) == [0.7, 0.2, 0.1]
    assert list(out.loc[1, ["p_safe", "p_elevated", "p_critical"]]) == [0.1, 0.1, 0.8]   # test row untouched
