"""Per-event feature table built from the CDMs available at the decision point.

Operators must commit to a maneuver at least 2 days before TCA, so inputs use
only CDMs with time_to_tca >= CUTOFF_DAYS. Targets come from the event's last
CDM (train only; test data stops at the cutoff).
"""
import numpy as np
import pandas as pd

from src.ingest import OBJECT_TYPES
from src.physics import combined_sigma, log10_pc_approx
from src.tiering import RISK_FLOOR, tier_array

CUTOFF_DAYS = 2.0
# Train events whose last CDM is still >= this far out never show the final
# state, so they can't supply a target.
TARGET_MAX_TTCA = 1.0

LATEST_COLS = [
    "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "c_sigma_r", "t_sigma_r", "F10", "AP", "c_object_type",
    "c_sigma_r_missing", "t_sigma_r_missing", "sw_missing",
]

FEATURES = [
    "risk_latest", "max_risk_estimate", "miss_latest_m", "log_miss", "relative_speed",
    "time_to_tca", "risk_is_floor",
    "risk_slope", "risk_delta_last", "risk_min", "risk_max", "risk_std", "n_cdms",
    "miss_slope", "miss_delta_last",
    "relative_position_r", "relative_position_t", "relative_position_n", "radial_share",
    "c_sigma_r", "t_sigma_r", "sigma_comb", "sigma_shrink",
    "gap_worst", "F10", "AP", "F10_mean", "AP_mean",
    "log10_pc_approx", "pc_disagreement",
    "c_sigma_r_missing", "sw_missing",
] + [f"obj_{t.replace(' ', '_')}" for t in OBJECT_TYPES]


TREND_COLS = ["n_cdms", "risk_slope", "risk_delta_last", "risk_min", "risk_max", "risk_std",
              "miss_slope", "miss_delta_last", "sigma_shrink", "F10_mean", "AP_mean",
              "trend_days"]


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    """Change in y per day as TCA approaches (positive = rising)."""
    if len(t) < 2 or np.ptp(t) == 0:
        return 0.0
    return float(-np.polyfit(t, y, 1)[0])


def _event_trends(g: pd.DataFrame) -> pd.Series:
    # g is sorted by time_to_tca descending, so the last row is the latest CDM
    t = g["time_to_tca"].to_numpy()
    risk = g["risk"].to_numpy()
    miss = g["miss_distance"].to_numpy()
    sig = combined_sigma(g["c_sigma_r"], g["t_sigma_r"])
    return pd.Series({
        "n_cdms": len(g),
        "risk_slope": _slope(t, risk),
        "risk_delta_last": risk[-1] - risk[-2] if len(g) > 1 else 0.0,
        "risk_min": risk.min(),
        "risk_max": risk.max(),
        "risk_std": risk.std() if len(g) > 1 else 0.0,
        "miss_slope": _slope(t, miss),
        "miss_delta_last": miss[-1] - miss[-2] if len(g) > 1 else 0.0,
        # orders of magnitude the combined sigma fell over the event (>0 = shrinking)
        "sigma_shrink": float(np.log10(max(sig[0], 1e-3) / max(sig[-1], 1e-3))),
        "F10_mean": g["F10"].mean(),
        "AP_mean": g["AP"].mean(),
        "trend_days": t[0] - t[-1],
    })


def build_event_table(cdms: pd.DataFrame, cutoff: float = CUTOFF_DAYS) -> pd.DataFrame:
    """One row per (split, event_id) with features, latest state and targets."""
    keys = ["split", "event_id"]
    cdms = cdms.sort_values(keys + ["time_to_tca"], ascending=[True, True, False])
    hist = cdms[cdms["time_to_tca"] >= cutoff]

    latest = hist.groupby(keys).last()[LATEST_COLS]
    latest = latest.rename(columns={"risk": "risk_latest", "miss_distance": "miss_latest_m"})
    if hist.empty:
        # groupby-apply on an empty frame returns the input columns, not ours
        trends = pd.DataFrame(columns=TREND_COLS, index=latest.index, dtype=float)
    else:
        trends = hist.groupby(keys)[["time_to_tca", "risk", "miss_distance", "c_sigma_r",
                                     "t_sigma_r", "F10", "AP"]].apply(_event_trends)
    ev = latest.join(trends)
    ev["n_cdms"] = ev["n_cdms"].astype(int)

    ev["log_miss"] = np.log10(ev["miss_latest_m"].clip(lower=1.0))
    ev["risk_is_floor"] = (ev["risk_latest"] <= RISK_FLOOR).astype(np.int8)
    ev["radial_share"] = ev["relative_position_r"].abs() / ev["miss_latest_m"].clip(lower=1.0)
    ev["sigma_comb"] = combined_sigma(ev["c_sigma_r"], ev["t_sigma_r"])
    ev["gap_worst"] = ev["max_risk_estimate"] - ev["risk_latest"]
    ev["log10_pc_approx"] = log10_pc_approx(ev["miss_latest_m"], ev["c_sigma_r"], ev["t_sigma_r"])
    ev["pc_disagreement"] = ev["log10_pc_approx"] - ev["risk_latest"]
    for t in OBJECT_TYPES:
        ev[f"obj_{t.replace(' ', '_')}"] = (ev["c_object_type"] == t).astype(np.int8)

    # Targets: state at the last CDM of the full history
    final = cdms.groupby(keys).last()[["time_to_tca", "risk", "miss_distance"]]
    final.columns = ["final_ttca", "risk_final", "miss_final_m"]
    ev = ev.join(final)
    has_target = (ev.index.get_level_values("split") == "train") & (ev["final_ttca"] < TARGET_MAX_TTCA)
    ev["has_target"] = has_target
    ev.loc[~has_target, ["risk_final", "miss_final_m"]] = np.nan
    ev["tier_actual"] = np.where(has_target, tier_array(ev["risk_final"], ev["miss_final_m"]), None)
    ev["tier_now"] = tier_array(ev["risk_latest"], ev["miss_latest_m"])
    return ev.drop(columns="final_ttca").reset_index()
