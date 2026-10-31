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

Targets are the organisers' labels (final = last CDM of every train event
with a CDM at the cutoff). The reported numbers come from a fixed 80/20
event_id holdout (models/holdout_*.json/parquet): models, calibration and the
alert threshold are fit on the 80% only. A second classifier gives
P(Elevated or worse), so predict() returns p_safe / p_elevated / p_critical.

The balanced classifier's scores are not probabilities, so they are mapped
through an isotonic calibration fit on the OOF scores (--calibrate-only redoes
just that in seconds); models/calibration.json reports Brier and reliability.

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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.features import FEATURES, build_event_table
from src.metrics import evaluate
from src.tiering import CRITICAL, ELEVATED, RISK_ELEVATED, RISK_FLOOR, tier_array, worse

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET_RECALL = 0.95
N_FOLDS = 5
# Reported numbers come from a fixed 80/20 split of train events by event_id
# (the organisers' protocol); the deployed bundle is then refit on all events.
HOLDOUT_FRAC = 0.2
HOLDOUT_SEED = 42
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
RELIABILITY_BINS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
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


# early_stopping='auto' switches on above 10,000 samples: the final fit on all
# events would then stop early while the CV folds don't, and its scores would
# no longer match the OOF scale the calibration and alert threshold come from.
def _clf():
    return HistGradientBoostingClassifier(max_iter=MAX_ITER, learning_rate=0.05,
                                          class_weight="balanced", early_stopping=False,
                                          random_state=SEED)


def _reg(**kw):
    return HistGradientBoostingRegressor(max_iter=MAX_ITER, learning_rate=0.05,
                                         early_stopping=False, random_state=SEED, **kw)


class _Constant:
    """Stand-in classifier when a fold has only one class."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def _targets(ev: pd.DataFrame) -> dict:
    return {
        "crit": (ev["tier_actual"] == CRITICAL).to_numpy(),
        "elev": ev["tier_actual"].isin([ELEVATED, CRITICAL]).to_numpy(),
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
        "clf_elev": _classifier(y["elev"])(X, y["elev"]),
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
        # bundles saved before the three-tier probabilities have no clf_elev
        "p_elevated_up": (models["clf_elev"].predict_proba(X)[:, 1] if "clf_elev" in models
                          else np.full(len(X), np.nan)),
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
           ["risk_forecast", "risk_forecast_q90", "miss_forecast_m", "p_critical", "p_elevated_up"]}
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
TIER_PROB_COLS = ["p_safe", "p_elevated", "p_critical"]
PREDICT_COLS = ["risk_forecast", "risk_forecast_q90", "miss_forecast_m", *TIER_PROB_COLS, "drivers"]


def tier_probs(p_critical, p_elevated_up) -> dict:
    """Three tier probabilities that sum to 1 from two calibrated ordinal ones:
    P(Critical) and P(Elevated or worse). Where the two cross, P(Elevated) is 0."""
    pc = np.asarray(p_critical, float)
    pe = np.maximum(np.asarray(p_elevated_up, float), pc)
    return {"p_safe": 1 - pe, "p_elevated": pe - pc, "p_critical": pc}


def oof_forecast(fc: pd.DataFrame, ev: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Swap in out-of-fold forecasts for train events the final model was fit on,
    so nothing scored on train rows is in-sample. fc and ev share an index."""
    fc = fc.copy()
    # All three tier probabilities move together, or the row no longer sums to 1
    cols = OOF_COLS + [c for c in TIER_PROB_COLS if c in oof and c not in OOF_COLS and c in fc]
    lookup = oof.set_index("event_id")[cols]
    rows = (ev["split"] == "train") & ev["event_id"].isin(lookup.index)
    fc.loc[rows, cols] = lookup.loc[ev.loc[rows, "event_id"]].to_numpy()
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
        return pd.DataFrame(columns=PREDICT_COLS, index=ev_df.index)
    pred = _raw_predict(bundle["models"], X, ev_df["risk_latest"], ev_df["log_miss"])
    pred = _apply_fallback(pred, bundle["use_risk_model"], bundle["use_miss_model"],
                           ev_df["risk_latest"], ev_df["miss_latest_m"])
    # Drivers rank on the raw score; calibration is monotone so the ranking is the same
    pred["drivers"] = _drivers(bundle["models"], X, bundle["medians"], pred["p_critical"])
    pred.update(tier_probs(calibrate_p(pred["p_critical"], bundle.get("calibration")),
                           calibrate_p(pred.pop("p_elevated_up"), bundle.get("calibration_elev"))))
    return pd.DataFrame(pred, index=ev_df.index)[PREDICT_COLS]


def _isotonic_map(p: np.ndarray, y: np.ndarray) -> dict:
    """Isotonic fit of y on p, as a strictly increasing piecewise-linear map
    through the centre of each constant block (a step function would tie
    events and reshuffle the queue).

    An end block where every event had the same outcome would map to exactly
    0 or 1, a certainty a finite sample can't support; those two blocks get
    the Jeffreys estimate (k + 1/2) / (n + 1) instead, kept strictly monotone."""
    from sklearn.isotonic import IsotonicRegression
    fit = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit_transform(p, y)
    order = np.argsort(p, kind="stable")
    ps, fs = np.asarray(p, float)[order], fit[order]
    block = np.r_[0, np.cumsum(np.diff(fs) != 0)]
    x = np.array([ps[block == b].mean() for b in np.unique(block)])
    v = np.array([fs[block == b][0] for b in np.unique(block)])
    n = np.array([(block == b).sum() for b in np.unique(block)])
    k = v * n   # isotonic block value = observed rate in the block
    if len(v) > 1:
        # never past the neighbouring block (halfway at most), so the map stays strictly increasing
        v[0] = min((k[0] + 0.5) / (n[0] + 1), (v[0] + v[1]) / 2)
        v[-1] = max((k[-1] + 0.5) / (n[-1] + 1), (v[-2] + v[-1]) / 2)
    elif len(v) == 1:
        v[0] = (k[0] + 0.5) / (n[0] + 1)
    return {"x": x.tolist(), "y": v.tolist()}


def calibrate_p(p, calibration: dict | None) -> np.ndarray:
    """Map raw classifier scores to calibrated P(Critical); identity without a map."""
    p = np.asarray(p, float)
    if not calibration:
        return p
    return np.interp(p, calibration["x"], calibration["y"])


def _reliability(p: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(RELIABILITY_BINS[:-1], RELIABILITY_BINS[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        rows.append({"bin": [lo, hi], "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else None,
                     "observed": float(y[m].mean()) if m.any() else None})
    return rows


def fit_calibration(p_raw, y, groups, n_folds: int = N_FOLDS) -> dict:
    """Isotonic calibration of out-of-fold P(crit) scores.

    The reported Brier score / reliability are cross-fitted (each fold mapped
    by a calibrator fit on the other folds), so they are not flattered by
    fitting and scoring on the same events. The stored map is fit on all
    events and is strictly increasing, so ranking and recall are unchanged;
    the alert threshold is re-derived on that calibrated scale.
    """
    p_raw, y = np.asarray(p_raw, float), np.asarray(y, bool)
    p_cv = np.zeros(len(p_raw))
    folds = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    for tr, va in folds.split(p_raw, groups=groups):
        p_cv[va] = calibrate_p(p_raw[va], _isotonic_map(p_raw[tr], y[tr]))
    brier = lambda q: float(np.mean((q - y) ** 2))
    cal_map = _isotonic_map(p_raw, y)
    p_cal = calibrate_p(p_raw, cal_map)
    return {
        "method": "isotonic, linear between block centres",
        "map": cal_map,
        "p_calibrated": p_cal,
        "crit_threshold": _threshold_for_recall(p_cal, y),
        "crit_threshold_raw": _threshold_for_recall(p_raw, y),
        "n_events": int(len(y)), "n_critical": int(y.sum()),
        "brier_raw": brier(p_raw), "brier_calibrated": brier(p_cv),
        "brier_climatology": brier(np.full(len(y), y.mean())),
        "reliability_raw": _reliability(p_raw, y),
        "reliability_calibrated": _reliability(p_cv, y),
    }


def apply_calibration(bundle: dict, oof: pd.DataFrame, cal: dict,
                      cal_elev: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Bundle and OOF table that output calibrated P(crit) (raw score kept as
    p_critical_raw) and, with cal_elev, the three tier probabilities."""
    bundle = {**bundle, "calibration": cal["map"], "crit_threshold": cal["crit_threshold"],
              "crit_threshold_raw": cal["crit_threshold_raw"]}
    oof = oof.copy()
    oof["p_critical_raw"] = oof.get("p_critical_raw", oof["p_critical"])
    oof["p_critical"] = cal["p_calibrated"]
    if cal_elev is not None:
        bundle["calibration_elev"] = cal_elev["map"]
        oof["p_elevated_up_raw"] = oof.get("p_elevated_up_raw", oof["p_elevated_up"])
        probs = tier_probs(oof["p_critical"], cal_elev["p_calibrated"])
        oof["p_safe"], oof["p_elevated"] = probs["p_safe"], probs["p_elevated"]
    return bundle, oof


def calibrate_both(oof: pd.DataFrame, ev: pd.DataFrame, n_folds: int = N_FOLDS) -> tuple[dict, dict]:
    """Isotonic calibration of P(Critical) and P(Elevated or worse) on OOF scores.
    ev holds the truth for oof's event_ids (train events with a target)."""
    truth = ev.set_index("event_id")["tier_actual"].loc[oof["event_id"]]
    groups = oof["event_id"].to_numpy()
    cal = fit_calibration(oof["p_critical"].to_numpy(), (truth == CRITICAL).to_numpy(), groups, n_folds)
    cal_elev = fit_calibration(oof["p_elevated_up"].to_numpy(),
                               truth.isin([ELEVATED, CRITICAL]).to_numpy(), groups, n_folds)
    return cal, cal_elev


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


def holdout_mask(ev: pd.DataFrame, frac: float = HOLDOUT_FRAC, seed: int = HOLDOUT_SEED) -> pd.Series:
    """True for the events held out for evaluation. Split by event_id, so no
    event has CDMs on both sides; deterministic for a given event set."""
    ids = ev["event_id"].to_numpy()
    out = pd.Series(False, index=ev.index)
    if len(np.unique(ids)) < 2:
        return out
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
                 .split(ids, groups=ids))
    out.iloc[te] = True
    return out


HOLDOUT_COLS = ["event_id", "final_ttca", "risk_final", "miss_final_m", "tier_actual", "tier_now",
                "risk_latest", "tier_final", *TIER_PROB_COLS, "risk_forecast", "risk_forecast_q90",
                "miss_forecast_m"]


def evaluate_holdout(ev: pd.DataFrame, n_folds: int = N_FOLDS) -> tuple[pd.DataFrame, dict]:
    """Fit everything (models, fallback choice, calibration, alert threshold) on
    the 80% part only, then predict the untouched 20%. Returns the held-out
    predictions and baseline-vs-model metrics on them."""
    ev = ev[ev["has_target"]].reset_index(drop=True)
    held = holdout_mask(ev).to_numpy()
    dev, test = ev[~held], ev[held].copy()
    bundle, cv = train(dev, n_folds)
    bundle, _ = apply_calibration(bundle, cv["oof_table"], *calibrate_both(cv["oof_table"], dev, n_folds))
    pred = predict(bundle, test)
    for c in [*TIER_PROB_COLS, "risk_forecast", "risk_forecast_q90", "miss_forecast_m"]:
        test[c] = pred[c]
    test["tier_final"] = _tier_final(test["tier_now"], test["risk_forecast_q90"],
                                     test["miss_forecast_m"], test["p_critical"],
                                     bundle["crit_threshold"])
    rf, mf = test["risk_final"], test["miss_final_m"]
    metrics = {
        "split": {"frac": HOLDOUT_FRAC, "seed": HOLDOUT_SEED, "by": "event_id",
                  "n_dev": int(len(dev)), "n_holdout": int(len(test))},
        "crit_threshold": bundle["crit_threshold"],
        "baseline": evaluate(rf, mf, test["risk_latest"], test["tier_now"]),
        "model": evaluate(rf, mf, test["risk_forecast"], test["tier_final"]),
    }
    return test[HOLDOUT_COLS].reset_index(drop=True), metrics


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
    cal = bundle.get("calibration")
    p = calibrate_p(clf.predict_proba(X)[:, 1], cal)
    p_storm = calibrate_p(clf.predict_proba(Xs)[:, 1], cal)
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
    ap.add_argument("--calibrate-only", action="store_true",
                    help="only (re)fit P(crit) calibration on the saved OOF scores (seconds)")
    ap.add_argument("--holdout-only", action="store_true",
                    help="only the 80/20 held-out evaluation (no deployed bundle)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ev = build_event_table(load_all())
    if args.calibrate_only:
        bundle = joblib.load(args.out / "forecaster.joblib")
        oof = pd.read_parquet(args.out / "oof_predictions.parquet")
        for c in ["p_critical", "p_elevated_up"]:
            if f"{c}_raw" in oof:
                oof[c] = oof[f"{c}_raw"]
        _calibrate_and_save(bundle, oof, ev, args.out)
        return
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
    test, hm = evaluate_holdout(ev)
    test.to_parquet(args.out / "holdout_predictions.parquet", index=False)
    (args.out / "holdout_ids.json").write_text(json.dumps(sorted(int(i) for i in test["event_id"])))
    (args.out / "holdout_metrics.json").write_text(json.dumps(hm, indent=2, default=float))
    for name in ["baseline", "model"]:
        m = hm[name]
        log.info("HOLDOUT %-8s recall_crit %.3f %s precision_crit %.3f %s (n=%d, %d Critical)", name,
                 m["recall_critical"], m["ci95"].get("recall_critical"), m["precision_critical"],
                 m["ci95"].get("precision_critical"), m["n_events"], m["n_critical"])
    if args.holdout_only:
        return
    log.info("training on %d events (%d Critical)", len(ev), (ev["tier_actual"] == CRITICAL).sum())
    bundle, cv = train(ev)

    (args.out / "cv_metrics.json").write_text(json.dumps(cv["model"], indent=2, default=float))
    _calibrate_and_save(bundle, cv["oof_table"], ev, args.out)
    for name in ["baseline", "model"]:
        m = cv[name]
        log.info("%-8s recall_crit %.3f precision_crit %.3f f2 %.3f mse_hr %.3f loss %.3f",
                 name, m["recall_critical"], m["precision_critical"], m["f2"], m["mse_hr"], m["loss"])
    log.info("saved %s", args.out / "forecaster.joblib")



def _calibrate_and_save(bundle: dict, oof: pd.DataFrame, ev: pd.DataFrame, out: Path):
    import joblib

    cal, cal_elev = calibrate_both(oof, ev[(ev["split"] == "train") & ev["has_target"]])
    bundle, oof = apply_calibration(bundle, oof, cal, cal_elev)
    joblib.dump(bundle, out / "forecaster.joblib")
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    report = {k: v for k, v in cal.items() if k not in ("map", "p_calibrated")}
    report["elevated_or_worse"] = {k: v for k, v in cal_elev.items()
                                   if k not in ("map", "p_calibrated", "crit_threshold", "crit_threshold_raw")}
    (out / "calibration.json").write_text(json.dumps(report, indent=2))
    log.info("P(crit) calibration: Brier %.4f raw -> %.4f calibrated (climatology %.4f); "
             "alert threshold %.4f raw -> %.4f calibrated", cal["brier_raw"],
             cal["brier_calibrated"], cal["brier_climatology"], cal["crit_threshold_raw"],
             cal["crit_threshold"])


if __name__ == "__main__":
    main()
