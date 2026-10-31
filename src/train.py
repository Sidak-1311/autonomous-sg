"""Train the forecaster of each event's state at TCA from its CDMs at the 2-day cutoff.

    python -m src.train --sample 500     # quick check on a few events
    python -m src.train                  # full train set (ask the team first)

Models (scikit-learn HistGradientBoosting, 5-fold GroupKFold on event_id):
- classifier for P(tier at TCA == CRITICAL); its threshold is set on the
  out-of-fold predictions to catch 95% of Critical events
- hurdle classifier for "final Pc drops to the -30 floor", then regressors for
  the median and 90th percentile of delta = risk_final - risk_latest on the rest
- regressor for the change in log10 miss distance
Each regressor must beat the latest-value baseline out of fold, or predict()
falls back to it. Saves models/forecaster.joblib, models/cv_metrics.json and
models/oof_predictions.parquet (out-of-fold forecasts for the train events).

Also writes models/conformal.json (no training needed, --conformal-only):
split-conformal bounds on final risk around risk_latest per tier_now group,
with out-of-fold coverage, and the storm levels for storm_stress().
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src.features import FEATURES, build_event_table
from src.metrics import evaluate
from src.tiering import CRITICAL, RISK_ELEVATED, RISK_FLOOR, tier_array, worse

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET_RECALL = 0.95
N_FOLDS = 5
MAX_ITER = 300
SEED = 0
# Forecast the -30 floor only when the hurdle classifier is this sure
FLOOR_PROB = 0.8
# Risk regressors are fit and used only where the latest Pc is at least 1e-8
RISK_NEAR = -8.0
# Conformal interval on final risk: 90% coverage, per tier_now group
INTERVAL_ALPHA = 0.1
INTERVAL_MIN_N = 30
# Storm stress test: space weather at this quantile of the data
STORM_QUANTILE = 0.99
STORM_COLS = ["F10", "AP", "F10_mean", "AP_mean"]

# Readable names for drivers: (text when above the train median, text when below)
DRIVER_TEXT = {
    "risk_latest": ("Pc high", "Pc low"),
    "max_risk_estimate": ("worst-case Pc high", "worst-case Pc low"),
    "miss_latest_m": ("miss distance large", "miss distance small"),
    "log_miss": ("miss distance large", "miss distance small"),
    "relative_speed": ("fast encounter", "slow encounter"),
    "time_to_tca": ("TCA far off", "TCA close"),
    "risk_is_floor": ("Pc at floor", "Pc above floor"),
    "risk_slope": ("risk rising", "risk falling"),
    "risk_delta_last": ("last CDM raised risk", "last CDM lowered risk"),
    "risk_min": ("risk never low", "risk was low"),
    "risk_max": ("risk peaked high", "risk never high"),
    "risk_std": ("risk jumping around", "risk steady"),
    "n_cdms": ("many CDMs", "few CDMs"),
    "miss_slope": ("miss distance growing", "miss distance shrinking"),
    "miss_delta_last": ("last CDM widened miss", "last CDM narrowed miss"),
    "relative_position_r": ("radial offset large", "radial offset small"),
    "relative_position_t": ("along-track offset large", "along-track offset small"),
    "relative_position_n": ("cross-track offset large", "cross-track offset small"),
    "radial_share": ("mostly radial miss", "little radial miss"),
    "c_sigma_r": ("debris position uncertain", "debris position precise"),
    "t_sigma_r": ("our position uncertain", "our position precise"),
    "sigma_comb": ("combined uncertainty high", "combined uncertainty low"),
    "sigma_shrink": ("sigma shrinking fast", "sigma shrinking slowly"),
    "gap_worst": ("big gap to worst case", "small gap to worst case"),
    "F10": ("solar flux high", "solar flux low"),
    "AP": ("geomagnetic activity high", "geomagnetic activity low"),
    "F10_mean": ("solar flux high", "solar flux low"),
    "AP_mean": ("geomagnetic activity high", "geomagnetic activity low"),
    "log10_pc_approx": ("physics Pc high", "physics Pc low"),
    "pc_disagreement": ("physics Pc above reported", "physics Pc below reported"),
    "c_sigma_r_missing": ("debris sigma missing", "debris sigma known"),
    "sw_missing": ("space weather missing", "space weather known"),
    "obj_DEBRIS": ("debris object", "not debris"),
    "obj_PAYLOAD": ("payload object", "not a payload"),
    "obj_ROCKET_BODY": ("rocket body", "not a rocket body"),
    "obj_UNKNOWN": ("uncatalogued object", "catalogued object"),
}


def _clf():
    return HistGradientBoostingClassifier(max_iter=MAX_ITER, learning_rate=0.05,
                                          class_weight="balanced", random_state=SEED)


def _reg(**kw):
    return HistGradientBoostingRegressor(max_iter=MAX_ITER, learning_rate=0.05,
                                         random_state=SEED, **kw)


class _Constant:
    """Stand-in classifier when a fold has only one class."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def _targets(ev: pd.DataFrame) -> dict:
    return {
        "crit": (ev["tier_actual"] == CRITICAL).to_numpy(),
        "delta": (ev["risk_final"] - ev["risk_latest"]).to_numpy(),
        "dmiss": (np.log10(ev["miss_final_m"].clip(lower=1.0)) - ev["log_miss"]).to_numpy(),
    }


def _classifier(y):
    return _clf().fit if 0 < y.sum() < len(y) else (lambda X, y: _Constant(float(y.mean())))


def _fit(X: pd.DataFrame, y: dict) -> dict:
    # Hurdle: many events end at the -30 floor (Pc effectively zero). One
    # classifier says whether the event drops there; the delta regressors only
    # learn from events that don't, so the floor jumps can't drag them down.
    # Low-risk events mostly drift further down; learning delta from them
    # biases the few high-risk ones, so the regressors see only the near regime.
    floor = y["delta"] + X["risk_latest"].to_numpy() <= RISK_FLOOR
    live = ~floor & (X["risk_latest"].to_numpy() >= RISK_NEAR)
    if live.sum() < 20:
        live = ~floor if (~floor).any() else np.ones(len(X), bool)
    return {
        "clf": _classifier(y["crit"])(X, y["crit"]),
        "clf_floor": _classifier(floor)(X, floor),
        "reg_delta": _reg(loss="absolute_error").fit(X[live], y["delta"][live]),
        "reg_q90": _reg(loss="quantile", quantile=0.9).fit(X[live], y["delta"][live]),
        "reg_miss": _reg(loss="absolute_error").fit(X, y["dmiss"]),
    }


def _raw_predict(models: dict, X: pd.DataFrame, risk_latest, log_miss) -> dict:
    risk_latest = np.asarray(risk_latest, float)
    near = risk_latest >= RISK_NEAR
    risk = np.where(near, np.clip(risk_latest + models["reg_delta"].predict(X), RISK_FLOOR, 0.0),
                    risk_latest)
    p_floor = models["clf_floor"].predict_proba(X)[:, 1]
    # A wrong floor call on a high-risk event costs ~24^2 in squared error, so
    # only send events to the floor that are already below the high-risk line
    risk = np.where((p_floor >= FLOOR_PROB) & (risk_latest < RISK_ELEVATED), RISK_FLOOR, risk)
    # q90 ignores the floor classifier: the cautious bound assumes the event stays live
    q90 = np.where(near, np.clip(risk_latest + models["reg_q90"].predict(X), RISK_FLOOR, 0.0),
                   risk_latest)
    log_miss_fc = np.asarray(log_miss, float) + models["reg_miss"].predict(X)
    return {
        "risk_forecast": risk,
        "risk_forecast_q90": np.maximum(q90, risk),
        "miss_forecast_m": 10 ** log_miss_fc,
        "p_critical": models["clf"].predict_proba(X)[:, 1],
    }


def _tier_final(tier_now, q90, miss_fc, p_crit, threshold):
    """Same rule as src.score: forecast tier, raised to Critical by P(crit)."""
    tf = tier_array(q90, miss_fc)
    tf[np.asarray(p_crit) >= threshold] = CRITICAL
    return np.array([worse(a, b) for a, b in zip(tier_now, tf)], dtype=object)


def _threshold_for_recall(p, y, target=TARGET_RECALL) -> float:
    pos = np.sort(np.asarray(p)[np.asarray(y, bool)])
    if len(pos) == 0:
        return 0.5
    # Highest threshold that still keeps >= target of positives at or above it
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def cross_validate(ev: pd.DataFrame, n_folds: int = N_FOLDS) -> dict:
    """Out-of-fold predictions plus model-vs-baseline decisions and metrics."""
    X, y = ev[FEATURES], _targets(ev)
    oof = {k: np.zeros(len(ev)) for k in
           ["risk_forecast", "risk_forecast_q90", "miss_forecast_m", "p_critical"]}
    folds = GroupKFold(n_splits=min(n_folds, ev["event_id"].nunique()))
    for k, (tr, va) in enumerate(folds.split(X, groups=ev["event_id"])):
        models = _fit(X.iloc[tr], {n: v[tr] for n, v in y.items()})
        pred = _raw_predict(models, X.iloc[va], ev["risk_latest"].iloc[va], ev["log_miss"].iloc[va])
        for n in oof:
            oof[n][va] = pred[n]
        log.info("fold %d/%d done", k + 1, folds.n_splits)

    risk_final, miss_final = ev["risk_final"].to_numpy(), ev["miss_final_m"].to_numpy()
    base = evaluate(risk_final, miss_final, ev["risk_latest"], ev["tier_now"], n_boot=0)
    raw = evaluate(risk_final, miss_final, oof["risk_forecast"], ev["tier_now"], n_boot=0)
    use_risk = bool(raw["loss"] < base["loss"])
    log_err = lambda m: np.mean(np.abs(np.log10(np.clip(m, 1.0, None)) - np.log10(np.clip(miss_final, 1.0, None))))
    use_miss = bool(log_err(oof["miss_forecast_m"]) < log_err(ev["miss_latest_m"]))
    log.info("risk model loss %.3f vs baseline %.3f -> %s", raw["loss"], base["loss"],
             "model" if use_risk else "baseline")
    log.info("miss model log10 MAE %.3f vs baseline %.3f -> %s", log_err(oof["miss_forecast_m"]),
             log_err(ev["miss_latest_m"]), "model" if use_miss else "baseline")

    oof = _apply_fallback(oof, use_risk, use_miss, ev["risk_latest"], ev["miss_latest_m"])
    threshold = _threshold_for_recall(oof["p_critical"], y["crit"])
    tier_final = _tier_final(ev["tier_now"], oof["risk_forecast_q90"], oof["miss_forecast_m"],
                             oof["p_critical"], threshold)
    model_metrics = evaluate(risk_final, miss_final, oof["risk_forecast"], tier_final)
    base_metrics = evaluate(risk_final, miss_final, ev["risk_latest"], ev["tier_now"])
    return {"oof": oof, "threshold": threshold, "use_risk_model": use_risk,
            "use_miss_model": use_miss, "model": model_metrics, "baseline": base_metrics}


def _apply_fallback(pred: dict, use_risk, use_miss, risk_latest, miss_latest_m) -> dict:
    pred = dict(pred)
    if not use_risk:
        pred["risk_forecast"] = np.asarray(risk_latest, float)
        pred["risk_forecast_q90"] = np.asarray(risk_latest, float)
    if not use_miss:
        pred["miss_forecast_m"] = np.asarray(miss_latest_m, float)
    return pred


def train(ev: pd.DataFrame, n_folds: int = N_FOLDS) -> tuple[dict, dict]:
    """Cross-validate, then fit the final models on every event with a target."""
    ev = ev[ev["has_target"]].reset_index(drop=True)
    cv = cross_validate(ev, n_folds)
    bundle = {
        "features": list(FEATURES),
        "models": _fit(ev[FEATURES], _targets(ev)),
        "medians": ev[FEATURES].median(),
        "crit_threshold": cv["threshold"],
        "use_risk_model": cv["use_risk_model"],
        "use_miss_model": cv["use_miss_model"],
    }
    cv["oof_table"] = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), **cv["oof"]})
    return bundle, cv


OOF_COLS = ["risk_forecast", "risk_forecast_q90", "miss_forecast_m", "p_critical"]


def oof_forecast(fc: pd.DataFrame, ev: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Swap in out-of-fold forecasts for train events the final model was fit on,
    so nothing scored on train rows is in-sample. fc and ev share an index."""
    fc = fc.copy()
    lookup = oof.set_index("event_id")[OOF_COLS]
    rows = (ev["split"] == "train") & ev["event_id"].isin(lookup.index)
    fc.loc[rows, OOF_COLS] = lookup.loc[ev.loc[rows, "event_id"]].to_numpy()
    return fc


def _drivers(models: dict, X: pd.DataFrame, medians: pd.Series, p: np.ndarray, top: int = 3):
    """Top features by how much P(crit) drops when each is reset to the train median."""
    effects = np.zeros((len(X), len(X.columns)))
    for j, f in enumerate(X.columns):
        Xr = X.copy()
        Xr[f] = medians[f]
        effects[:, j] = p - models["clf"].predict_proba(Xr)[:, 1]
    order = np.argsort(-np.abs(effects), axis=1)[:, :top]
    vals, med = X.to_numpy(), medians.to_numpy()
    out = []
    for i, cols in enumerate(order):
        parts = []
        for j in cols:
            if abs(effects[i, j]) < 1e-6:
                continue
            hi, lo = DRIVER_TEXT.get(X.columns[j], (f"{X.columns[j]} high", f"{X.columns[j]} low"))
            parts.append(f"{hi if vals[i, j] > med[j] else lo} {'+' if effects[i, j] > 0 else '−'}")
        out.append("; ".join(parts))
    return out


def predict(bundle: dict, ev_df: pd.DataFrame) -> pd.DataFrame:
    """Forecast for each event row: risk, q90 risk, miss, P(crit) and top drivers."""
    X = ev_df[bundle["features"]]
    if len(X) == 0:
        return pd.DataFrame(columns=["risk_forecast", "risk_forecast_q90", "miss_forecast_m",
                                     "p_critical", "drivers"], index=ev_df.index)
    pred = _raw_predict(bundle["models"], X, ev_df["risk_latest"], ev_df["log_miss"])
    pred = _apply_fallback(pred, bundle["use_risk_model"], bundle["use_miss_model"],
                           ev_df["risk_latest"], ev_df["miss_latest_m"])
    pred["drivers"] = _drivers(bundle["models"], X, bundle["medians"], pred["p_critical"])
    return pd.DataFrame(pred, index=ev_df.index)


def _residual_bounds(resid: np.ndarray, alpha: float) -> tuple[float, float]:
    """Split-conformal equal-tailed bounds with the finite-sample correction."""
    n = len(resid)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha / 2)) / n)
    return float(np.quantile(resid, 1 - level)), float(np.quantile(resid, level))


def calibrate_interval(ev: pd.DataFrame, alpha: float = INTERVAL_ALPHA,
                       n_folds: int = N_FOLDS) -> dict:
    """Per-tier_now residual bounds for final risk around risk_latest, plus coverage
    measured out of fold (calibrate on k-1 folds, check on the held-out one)."""
    ev = ev[ev["has_target"]].reset_index(drop=True)
    resid = (ev["risk_final"] - ev["risk_latest"]).to_numpy()
    groups = ev["tier_now"].to_numpy()
    calib = {"alpha": alpha, "center": "risk_latest",
             **calibrate_interval_bounds(ev, resid, alpha)}
    for g, d in calib["groups"].items():
        d["n"] = int((groups == g).sum())
    calib["pooled"]["n"] = int(len(resid))

    covered = np.zeros(len(ev), bool)
    folds = GroupKFold(n_splits=min(n_folds, ev["event_id"].nunique()))
    for tr, va in folds.split(ev, groups=ev["event_id"]):
        fold = calibrate_interval_bounds(ev.iloc[tr], resid[tr], alpha)
        iv = risk_interval(ev.iloc[va], fold)
        covered[va] = ((ev["risk_final"].iloc[va] >= iv["risk_lo"] - 1e-9)
                       & (ev["risk_final"].iloc[va] <= iv["risk_hi"] + 1e-9)).to_numpy()
    calib["coverage"] = float(covered.mean())
    for g, d in calib["groups"].items():
        d["coverage"] = float(covered[groups == g].mean())
    return calib


def calibrate_interval_bounds(ev: pd.DataFrame, resid: np.ndarray, alpha: float) -> dict:
    """Bounds per tier_now group (groups under INTERVAL_MIN_N use the pooled ones)."""
    groups = ev["tier_now"].to_numpy()
    out = {"groups": {}, "pooled": dict(zip(["lo", "hi"], _residual_bounds(resid, alpha)))}
    for g in np.unique(groups):
        r = resid[groups == g]
        if len(r) >= INTERVAL_MIN_N:
            out["groups"][str(g)] = dict(zip(["lo", "hi"], _residual_bounds(r, alpha)))
    return out


def risk_interval(ev: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """Interval [risk_lo, risk_hi] (log10 Pc) the final risk falls in with ~1-alpha coverage."""
    latest = ev["risk_latest"].to_numpy(float)
    bounds = [calib["groups"].get(str(t), calib["pooled"]) for t in ev["tier_now"]]
    lo = np.array([b["lo"] for b in bounds], float).reshape(-1)
    hi = np.array([b["hi"] for b in bounds], float).reshape(-1)
    return pd.DataFrame({"risk_lo": np.clip(latest + lo, RISK_FLOOR, 0.0),
                         "risk_hi": np.clip(latest + hi, RISK_FLOOR, 0.0)}, index=ev.index)


def storm_levels(ev: pd.DataFrame, q: float = STORM_QUANTILE) -> dict:
    """Space-weather values at the data's own upper tail."""
    return {c: float(ev[c].quantile(q)) for c in STORM_COLS}


def storm_stress(bundle: dict, ev: pd.DataFrame, levels: dict | None = None) -> pd.DataFrame:
    """P(crit) with solar flux / geomagnetic activity at storm levels, and which
    events that change pushes over the alert threshold."""
    levels = levels or bundle.get("storm_levels") or storm_levels(ev)
    X = ev[bundle["features"]]
    if len(X) == 0:
        return pd.DataFrame(columns=["p_critical_storm", "storm_flips"], index=ev.index)
    Xs = X.copy()
    for c, v in levels.items():
        if c in Xs:
            Xs[c] = np.maximum(Xs[c], v)   # a storm never lowers activity
    clf, thr = bundle["models"]["clf"], bundle["crit_threshold"]
    p = clf.predict_proba(X)[:, 1]
    p_storm = clf.predict_proba(Xs)[:, 1]
    return pd.DataFrame({"p_critical_storm": p_storm,
                         "storm_flips": (p_storm >= thr) & (p < thr)}, index=ev.index)


def main():
    import joblib

    from src.ingest import load_all

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=0, help="train on this many random events")
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    ap.add_argument("--conformal-only", action="store_true",
                    help="only write conformal.json (seconds, no model training)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ev = build_event_table(load_all())
    args.out.mkdir(parents=True, exist_ok=True)
    calib = calibrate_interval(ev)
    calib["storm_levels"] = storm_levels(ev)
    (args.out / "conformal.json").write_text(json.dumps(calib, indent=2))
    log.info("risk interval: %.0f%% target, %.1f%% measured out of fold %s", 100 * (1 - calib["alpha"]),
             100 * calib["coverage"], {g: f"{d['coverage']:.1%}" for g, d in calib["groups"].items()})
    if args.conformal_only:
        return
    ev = ev[ev["has_target"]]
    if args.sample:
        ev = ev.sample(min(args.sample, len(ev)), random_state=SEED)
    log.info("training on %d events (%d Critical)", len(ev), (ev["tier_actual"] == CRITICAL).sum())
    bundle, cv = train(ev)

    joblib.dump(bundle, args.out / "forecaster.joblib")
    (args.out / "cv_metrics.json").write_text(json.dumps(cv["model"], indent=2, default=float))
    cv["oof_table"].to_parquet(args.out / "oof_predictions.parquet", index=False)
    for name in ["baseline", "model"]:
        m = cv[name]
        log.info("%-8s recall_crit %.3f precision_crit %.3f f2 %.3f mse_hr %.3f loss %.3f",
                 name, m["recall_critical"], m["precision_critical"], m["f2"], m["mse_hr"], m["loss"])
    log.info("saved %s", args.out / "forecaster.joblib")


if __name__ == "__main__":
    main()
