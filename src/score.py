"""Score every event and write the files the dashboard reads.

    python -m src.score

Writes data/processed/{cdms.parquet, events_scored.parquet, metrics.json}.
Uses the trained forecaster in models/forecaster.joblib when present,
otherwise the latest-risk baseline (final risk = latest risk).
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.cascade import COLUMNS as CASCADE_OUT
from src.cascade import cascade
from src.features import build_event_table
from src.ingest import CASCADE_COLUMNS, COLUMNS, PROCESSED_DIR, load_all
from src.physics import combined_sigma, dilution, radial_maneuver
from src.tiering import (CRITICAL, TIERS, format_miss, format_pc, format_prob, rule_fired,
                         tier_array, worse)

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "forecaster.joblib"
CONFORMAL_PATH = MODELS_DIR / "conformal.json"
OOF_PATH = MODELS_DIR / "oof_predictions.parquet"
HOLDOUT_PATH = MODELS_DIR / "holdout_predictions.parquet"

CDM_COLS = ["split"] + COLUMNS + ["tier"]
EVENT_COLS = [
    "split", "event_id", "c_object_type", "n_cdms", "time_to_tca",
    "risk_latest", "miss_latest_m", "max_risk_estimate", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "risk_slope", "risk_forecast", "risk_forecast_q90", "miss_forecast_m", "p_critical",
    "tier_now", "tier_forecast", "tier_final", "reason", "drivers", "forecast_source",
    "risk_final", "miss_final_m", "tier_actual",
    "maneuver_dr_m", "maneuver_dv_mps", "pc_diluted", "risk_potential",
    "needs_review", "review_reason",
    "risk_lo", "risk_hi", "forecast_in_sample", "p_critical_storm", "storm_flips",
] + CASCADE_COLUMNS + CASCADE_OUT


def baseline_forecast(ev: pd.DataFrame) -> pd.DataFrame:
    """Final state = latest state. The floor every model must beat."""
    return pd.DataFrame({
        "risk_forecast": ev["risk_latest"],
        "risk_forecast_q90": ev["risk_latest"],
        "miss_forecast_m": ev["miss_latest_m"],
        "p_critical": (ev["tier_now"] == CRITICAL).astype(float),
        "drivers": "",
    }, index=ev.index)


def _load_calib() -> dict | None:
    return json.loads(CONFORMAL_PATH.read_text()) if CONFORMAL_PATH.exists() else None


def model_forecast(ev: pd.DataFrame, calib: dict | None):
    """Trained forecast (+ storm stress test) and its P(Critical) threshold,
    or None without a model.

    The final model was fit on train events with a known outcome, so its
    scores for them are in-sample. When train.py's out-of-fold predictions
    exist they replace those rows; otherwise the rows are marked in-sample.
    """
    if not MODEL_PATH.exists():
        return None
    import joblib

    from src import train
    bundle = joblib.load(MODEL_PATH)
    fc = train.predict(bundle, ev)
    fitted_on = (ev["split"] == "train") & ev["has_target"]
    if OOF_PATH.exists() and hasattr(train, "oof_forecast"):
        oof = pd.read_parquet(OOF_PATH)
        fc = train.oof_forecast(fc, ev, oof)
        fc["forecast_in_sample"] = (fitted_on & ~ev["event_id"].isin(oof["event_id"])).to_numpy()
    else:
        fc["forecast_in_sample"] = fitted_on.to_numpy()
    levels = (calib or {}).get("storm_levels")
    fc = fc.join(train.storm_stress(bundle, ev, levels))
    return fc, bundle.get("crit_threshold", 0.5)


def _withhold(ev: pd.DataFrame, mask: pd.Series) -> None:
    """No forecast on thin evidence: forecast = latest state, no P(crit)."""
    ev.loc[mask, "risk_forecast"] = ev.loc[mask, "risk_latest"]
    ev.loc[mask, "risk_forecast_q90"] = ev.loc[mask, "risk_latest"]
    ev.loc[mask, "miss_forecast_m"] = ev.loc[mask, "miss_latest_m"]
    ev.loc[mask, ["p_critical", "p_critical_storm"]] = np.nan
    ev.loc[mask, "storm_flips"] = False
    ev.loc[mask, "drivers"] = ""
    ev.loc[mask, "forecast_source"] = "withheld"


def _reason(row, threshold: float) -> str:
    now = row.tier_now.capitalize()
    fc = row.tier_forecast.capitalize()
    text = f"{now} now ({rule_fired(row.risk_latest, row.miss_latest_m)})"
    if row.forecast_source == "model":
        text += f"; forecast {fc} (q90 Pc {format_pc(row.risk_forecast_q90)}, " \
                f"miss {format_miss(row.miss_forecast_m)}, P(crit) {format_prob(row.p_critical)}"
        if row.p_critical >= threshold:
            # The threshold is set low on purpose to catch 95% of Critical events
            text += f" ≥ alert threshold {format_prob(threshold)}, set for 95% recall"
        text += ")"
    if pd.notna(row.risk_hi):
        # risk_hi is the top of an equal-tailed 90% split-conformal interval, so
        # 5% of outcomes fall above it: a one-sided 95% upper bound. The lower end
        # is uninformative (many events collapse to the -30 floor), so omit it.
        text += f"; 95% upper bound on Pc at TCA: {format_pc(row.risk_hi)}"
    if abs(row.risk_slope) >= 0.1 and row.n_cdms > 1:
        direction = "rising" if row.risk_slope > 0 else "falling"
        text += f"; Pc {direction} {abs(row.risk_slope):.1f} orders of magnitude per day"
    return text


def _add_maneuver(ev: pd.DataFrame) -> None:
    """Suggested radial offset and burn for Critical events, planned on the
    worse of the latest and forecast state."""
    risk = np.maximum(ev["risk_latest"], ev["risk_forecast_q90"])
    miss = np.minimum(ev["miss_latest_m"], ev["miss_forecast_m"])
    dr, dv = radial_maneuver(risk, miss, ev["relative_position_r"], ev["relative_position_t"],
                             ev["relative_position_n"],
                             combined_sigma(ev["c_sigma_r"], ev["t_sigma_r"]))
    critical = ev["tier_final"] == CRITICAL
    ev["maneuver_dr_m"] = np.where(critical, dr, np.nan)
    ev["maneuver_dv_mps"] = np.where(critical, dv, np.nan)


def _add_dilution(ev: pd.DataFrame) -> None:
    """Flag events whose covariance is large relative to the miss distance.

    On train outcomes this flag does NOT predict danger (flagged Safe events
    never ended Critical), so it is a tracking-quality note, not a warning.
    """
    diluted, potential = dilution(ev["risk_latest"], ev["miss_latest_m"],
                                  combined_sigma(ev["c_sigma_r"], ev["t_sigma_r"]))
    ev["pc_diluted"] = diluted
    ev["risk_potential"] = potential


# On train outcomes the latest tier is wrong about twice as often for these
# events (11.7% vs 6.9%), so no forecast is offered and they can't be auto-cleared.
STALE_DAYS = 5.0


def _add_review(ev: pd.DataFrame) -> None:
    """Refuse to forecast when the evidence is too thin; an analyst must review."""
    checks = [
        (ev["n_cdms"] == 1, "only one CDM before the decision point, so no trend"),
        (ev["time_to_tca"] > STALE_DAYS,
         f"latest CDM is over {STALE_DAYS:.0f} days before TCA (stale)"),
        ((ev["c_sigma_r_missing"] == 1) | (ev["t_sigma_r_missing"] == 1),
         "covariance missing from the latest CDM"),
    ]
    ev["needs_review"] = np.logical_or.reduce([m.to_numpy() for m, _ in checks])
    ev["review_reason"] = [
        "; ".join(text for (m, text) in checks if m.iat[i]) for i in range(len(ev))]


def score(cdms: pd.DataFrame) -> pd.DataFrame:
    ev = build_event_table(cdms)
    if ev.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    calib = _load_calib()
    out = model_forecast(ev, calib)
    if out is None:
        fc, threshold, source = baseline_forecast(ev), 0.5, "baseline"
        fc["forecast_in_sample"] = False
        fc["p_critical_storm"], fc["storm_flips"] = np.nan, False
    else:
        (fc, threshold), source = out, "model"
    ev = ev.join(fc)
    ev["forecast_source"] = source
    if calib is not None:
        from src import train
        ev = ev.join(train.risk_interval(ev, calib))
    else:
        ev["risk_lo"], ev["risk_hi"] = np.nan, np.nan

    _add_review(ev)
    _withhold(ev, ev["needs_review"])

    ev["tier_forecast"] = tier_array(ev["risk_forecast_q90"], ev["miss_forecast_m"])
    ev.loc[ev["p_critical"] >= threshold, "tier_forecast"] = CRITICAL
    ev["tier_final"] = [worse(a, b) for a, b in zip(ev["tier_now"], ev["tier_forecast"])]
    ev["reason"] = [_reason(r, threshold) for r in ev.itertuples()]
    _add_maneuver(ev)
    _add_dilution(ev)
    ev.loc[ev["pc_diluted"], "reason"] += (
        "; note: combined radial σ exceeds miss/√2, so Pc is covariance-limited "
        "(tracking-quality note, not a risk signal)")
    review = ev["needs_review"]
    ev.loc[review, "reason"] = ("INSUFFICIENT EVIDENCE, analyst must review ("
                                + ev.loc[review, "review_reason"] + "). "
                                + ev.loc[review, "reason"])

    # "If this collision happened": fragments and chain risk to other missions in the
    # shell. Run on all events (both splits) so every mission is known to the shell model.
    ev = ev.join(cascade(ev))

    # Queue order, same as the dashboard's: worst tier, then P(crit), then
    # forecast risk, then soonest TCA (withheld events have no P(crit): last in tier)
    ev["_rank"] = ev["tier_final"].map({t: i for i, t in enumerate(TIERS)})
    ev = ev.sort_values(["_rank", "p_critical", "risk_forecast", "time_to_tca"],
                        ascending=[False, False, False, True], na_position="last")
    # reindex: cascade inputs may be absent in older extracts and test fixtures
    return ev.reindex(columns=EVENT_COLS).reset_index(drop=True)


def evaluate_baseline_and_model(ev: pd.DataFrame) -> dict:
    """Metrics on train events with a known outcome (test outcomes are hidden)."""
    try:
        from src.metrics import evaluate
    except ImportError:
        log.warning("src.metrics not available yet; skipping metrics")
        return {}
    known = ev[ev["tier_actual"].notna()]
    results = {"baseline": evaluate(known["risk_final"], known["miss_final_m"],
                                    known["risk_latest"], known["tier_now"])}
    cv_path = MODEL_PATH.with_name("cv_metrics.json")
    if cv_path.exists():
        # Model scores on train would be in-sample; report out-of-fold CV instead
        results["model"] = json.loads(cv_path.read_text())
    if not known["forecast_in_sample"].any():
        results["queue_depth"] = _queue_depth(known)
    results.update(holdout_metrics())
    return results


def holdout_metrics(path: Path = HOLDOUT_PATH) -> dict:
    """Confusion matrices on the fixed 80/20 event_id holdout (train.py writes the predictions).

    'holdout' covers every held-out event. 'holdout_forecast_only' keeps events whose final CDM
    is less than 2 days out: for the rest the final tier is already known at the decision point
    (tier_actual == tier_now trivially), which inflates every score.
    """
    if not path.exists():
        return {}
    from src.metrics import confusion
    hold = pd.read_parquet(path)
    hold = hold[hold["tier_actual"].notna()]

    def block(df, note):
        return {"n": int(len(df)), "note": note,
                "baseline": confusion(df["tier_actual"], df["tier_now"]),
                "model": confusion(df["tier_actual"], df["tier_final"])}

    out = {"holdout": block(hold, "all held-out events (organisers' usable set), 20% by event_id")}
    if "final_ttca" in hold:
        hard = hold[hold["final_ttca"] < 2.0]
        out["holdout_forecast_only"] = block(
            hard, "final CDM < 2 d out: the outcome is not yet visible at the decision point")
    return out


def _queue_depth(known: pd.DataFrame) -> dict:
    """Reviews needed to catch 95% of Criticals: the queue order (ev is already
    sorted) vs today's order (latest tier, then latest Pc, then soonest TCA)."""
    from src.metrics import queue_depth
    rank_model = np.arange(len(known))
    base = known.assign(_r=known["tier_now"].map({t: i for i, t in enumerate(TIERS)}),
                        _pos=rank_model)
    base = base.sort_values(["_r", "risk_latest", "time_to_tca"], ascending=[False, False, True])
    rank_base = np.empty(len(known), int)
    rank_base[base["_pos"].to_numpy()] = np.arange(len(known))
    return queue_depth((known["tier_actual"] == CRITICAL).to_numpy(), rank_model, rank_base)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cdms = load_all()
    cdms[CDM_COLS].to_parquet(PROCESSED_DIR / "cdms.parquet", index=False)

    ev = score(cdms)
    ev.to_parquet(PROCESSED_DIR / "events_scored.parquet", index=False)
    log.info("scored %d events (%s); tier_final counts: %s", len(ev),
             ev["forecast_source"].iat[0],
             ev.groupby("split")["tier_final"].value_counts().to_dict())

    metrics = evaluate_baseline_and_model(ev)
    (PROCESSED_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("wrote outputs to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
