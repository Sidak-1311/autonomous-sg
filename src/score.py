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

from src.features import build_event_table
from src.ingest import COLUMNS, PROCESSED_DIR, load_all
from src.tiering import CRITICAL, TIERS, rule_fired, tier_array, worse

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "forecaster.joblib"

CDM_COLS = ["split"] + COLUMNS + ["tier"]
EVENT_COLS = [
    "split", "event_id", "c_object_type", "n_cdms", "time_to_tca",
    "risk_latest", "miss_latest_m", "max_risk_estimate", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "risk_slope", "risk_forecast", "risk_forecast_q90", "miss_forecast_m", "p_critical",
    "tier_now", "tier_forecast", "tier_final", "reason", "drivers", "forecast_source",
    "risk_final", "miss_final_m", "tier_actual",
]


def baseline_forecast(ev: pd.DataFrame) -> pd.DataFrame:
    """Final state = latest state. The floor every model must beat."""
    return pd.DataFrame({
        "risk_forecast": ev["risk_latest"],
        "risk_forecast_q90": ev["risk_latest"],
        "miss_forecast_m": ev["miss_latest_m"],
        "p_critical": (ev["tier_now"] == CRITICAL).astype(float),
        "drivers": "",
    }, index=ev.index)


def model_forecast(ev: pd.DataFrame):
    """Trained forecast plus its P(Critical) threshold, or None without a model."""
    if not MODEL_PATH.exists():
        return None
    import joblib

    from src import train
    bundle = joblib.load(MODEL_PATH)
    return train.predict(bundle, ev), bundle.get("crit_threshold", 0.5)


def _reason(row) -> str:
    now = row.tier_now.capitalize()
    fc = row.tier_forecast.capitalize()
    text = f"{now} now ({rule_fired(row.risk_latest, row.miss_latest_m)})"
    if row.forecast_source == "model":
        text += f"; forecast {fc} (q90 Pc 1e{row.risk_forecast_q90:.1f}, " \
                f"miss {row.miss_forecast_m / 1000:.2f} km, P(crit) {row.p_critical:.0%})"
    if abs(row.risk_slope) >= 0.1 and row.n_cdms > 1:
        direction = "up" if row.risk_slope > 0 else "down"
        text += f"; risk trending {direction} {abs(row.risk_slope):.1f} orders/day"
    return text


def score(cdms: pd.DataFrame) -> pd.DataFrame:
    ev = build_event_table(cdms)
    out = model_forecast(ev)
    if out is None:
        fc, threshold, source = baseline_forecast(ev), 0.5, "baseline"
    else:
        (fc, threshold), source = out, "model"
    ev = ev.join(fc)
    ev["forecast_source"] = source

    ev["tier_forecast"] = tier_array(ev["risk_forecast_q90"], ev["miss_forecast_m"])
    ev.loc[ev["p_critical"] >= threshold, "tier_forecast"] = CRITICAL
    ev["tier_final"] = [worse(a, b) for a, b in zip(ev["tier_now"], ev["tier_forecast"])]
    ev["reason"] = [_reason(r) for r in ev.itertuples()]

    # Queue order: worst tier, then highest forecast risk, then soonest TCA
    ev["_rank"] = ev["tier_final"].map({t: i for i, t in enumerate(TIERS)})
    ev = ev.sort_values(["_rank", "risk_forecast_q90", "time_to_tca"],
                        ascending=[False, False, True])
    return ev[EVENT_COLS].reset_index(drop=True)


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
    return results


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
