"""Shared data loading, decision log and chart helpers for the dashboard sections."""
# Imports below are re-exported to the section modules (they star-import this one).
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dashboard import brief, style  # noqa: E402
from src.physics import LEO_MEAN_MOTION, combined_sigma, dilution  # noqa: E402
from src.score import _add_maneuver, _add_review  # noqa: E402  (the pipeline's own rules, reused by the stub)
from src.tiering import (  # noqa: E402
    CRITICAL, ELEVATED, MISS_CRITICAL_M, MISS_ELEVATED_M, RISK_CRITICAL, RISK_ELEVATED, RISK_FLOOR, SAFE,
    TIER_COLORS, TIER_RANK, TIERS, format_miss, format_pc, format_prob, format_ttca, rule_fired, tier_array,
)

def data_dir() -> Path:
    """Read on every call: tests and users switch DATA_DIR between runs."""
    return Path(os.environ.get("DATA_DIR", ROOT / "data"))


def processed_dir() -> Path:
    return data_dir() / "processed"
DECISION_DAYS = 2.0  # the latest CDM the model may see is at time_to_tca >= 2 days
RAW_COLS = ["event_id", "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
            "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type",
            "c_sigma_r", "t_sigma_r"]


def stub_from_raw(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contract-shaped tables from the raw test CSV, so the dashboard works before the pipeline has run."""
    cdms = pd.read_csv(path, usecols=RAW_COLS).assign(split="test")
    cdms["tier"] = tier_array(cdms.risk, cdms.miss_distance)
    seen = cdms[cdms.time_to_tca >= DECISION_DAYS]
    latest = seen.loc[seen.groupby("event_id").time_to_tca.idxmin()]
    ev = latest.rename(columns={"risk": "risk_latest", "miss_distance": "miss_latest_m"}).drop(columns="tier")
    ev["n_cdms"] = ev.event_id.map(seen.groupby("event_id").size())  # CDMs up to the decision point
    ev["risk_forecast"] = ev["risk_forecast_q90"] = ev.risk_latest
    ev["miss_forecast_m"] = ev.miss_latest_m
    ev["tier_now"] = ev["tier_forecast"] = ev["tier_final"] = tier_array(ev.risk_latest, ev.miss_latest_m)
    ev["reason"] = [rule_fired(r, m) for r, m in zip(ev.risk_latest, ev.miss_latest_m)]
    ev = ev.assign(p_critical=np.nan, drivers="", forecast_source="stub", risk_final=np.nan, miss_final_m=np.nan,
                   tier_actual=None)
    ev["pc_diluted"], ev["risk_potential"] = dilution(ev.risk_latest, ev.miss_latest_m,
                                                      combined_sigma(ev.c_sigma_r, ev.t_sigma_r))
    _add_maneuver(ev)
    ev["c_sigma_r_missing"], ev["t_sigma_r_missing"] = ev.c_sigma_r.isna().astype(int), ev.t_sigma_r.isna().astype(int)
    _add_review(ev)
    return ev.reset_index(drop=True), cdms


@st.cache_data
def load(data_root: str, stamp: tuple) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """data_root and stamp (file mtimes) are the cache key: rewritten files are picked up on the next rerun."""
    if (processed_dir() / "events_scored.parquet").exists() and (processed_dir() / "cdms.parquet").exists():
        events = pd.read_parquet(processed_dir() / "events_scored.parquet")
        cdms = pd.read_parquet(processed_dir() / "cdms.parquet")
        sources = ", ".join(sorted(events.forecast_source.dropna().unique()))
        return with_defaults(events), cdms, f"pipeline output ({sources})"
    events, cdms = stub_from_raw(data_dir() / "test_data.csv")
    return with_defaults(events), cdms, "stub from test_data.csv (no pipeline output yet)"


# Columns the pipeline added over time, with what an older output (or the stub) means by their absence.
LATER_COLUMNS = {
    "pc_diluted": lambda ev: False, "risk_potential": lambda ev: ev.risk_latest,
    "maneuver_dr_m": lambda ev: np.nan, "maneuver_dv_mps": lambda ev: np.nan,
    "needs_review": lambda ev: False, "review_reason": lambda ev: "",
    "risk_lo": lambda ev: ev.risk_forecast, "risk_hi": lambda ev: ev.risk_forecast_q90,
    "p_critical_storm": lambda ev: np.nan, "storm_flips": lambda ev: False,
    "forecast_in_sample": lambda ev: ev.forecast_source == "model",  # before the flag, train rows were in-sample
}


def with_defaults(events: pd.DataFrame) -> pd.DataFrame:
    return events.assign(**{col: fill(events) for col, fill in LATER_COLUMNS.items() if col not in events})


# Analyst sign-off: the system recommends, a named person decides, and every decision is logged with what
# the system showed at the time. ponytail: an append-only CSV per laptop; move it to a shared store (the
# room or a database) if several analysts sign the same queue.
def decisions_path() -> Path:
    return data_dir() / "decisions.csv"
DECISION_COLS = ["at_utc", "split", "event_id", "analyst", "decision", "note", "tier_final", "p_critical",
                 "forecast_source"]
CHOICES = ["Maneuver", "Monitor", "Dismiss"]


def read_decisions() -> pd.DataFrame:
    return pd.read_csv(decisions_path()) if decisions_path().exists() else pd.DataFrame(columns=DECISION_COLS)


def decision_problem(analyst: str, decision: str, note: str, e: pd.Series) -> str | None:
    if not analyst.strip():
        return "Enter your name: every decision is signed."
    if decision == "Dismiss" and (e.tier_final == CRITICAL or e.needs_review) and not note.strip():
        return "Dismissing a Critical or not-yet-reviewable event needs a note saying why."
    return None


def record_decision(row: dict):
    pd.DataFrame([row], columns=DECISION_COLS).to_csv(decisions_path(), mode="a", header=not decisions_path().exists(),
                                                      index=False)


@st.cache_data(show_spinner="Asking Gemini…")
def cached_brief(fields_json: str) -> str:
    """Keyed on the exact fields sent, so reruns and repeat clicks don't call the API again."""
    return brief.explain(json.loads(fields_json), brief.api_key())


def stamp() -> tuple:
    files = [processed_dir() / "events_scored.parquet", processed_dir() / "cdms.parquet", data_dir() / "test_data.csv"]
    return tuple(f.stat().st_mtime if f.exists() else None for f in files)


def load_metrics() -> dict | None:
    path = processed_dir() / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


# metric -> (label, higher is better)
METRICS = {"recall_critical": ("Critical recall", True), "precision_critical": ("Critical precision", True),
           "f2": ("F2 (high risk)", True), "mse_hr": ("MSE on high-risk events", False), "loss": ("loss = MSE / F2", False)}


def metrics_table(metrics: dict) -> pd.DataFrame:
    """One row per metric: value and 95% CI for each scorer present (baseline, model), and which is better."""
    def ci(run, key):
        lo_hi = run.get("ci95", {}).get(key)
        return f"[{lo_hi[0]:.3g}, {lo_hi[1]:.3g}]" if lo_hi else "—"

    rows = []
    for key, (label, higher) in METRICS.items():
        row = {"metric": label}
        for name in ("baseline", "model"):
            if name in metrics and key in metrics[name]:
                row[name], row[f"{name} 95% CI"] = metrics[name][key], ci(metrics[name], key)
        if "baseline" in row and "model" in row:
            better = row["model"] > row["baseline"] if higher else row["model"] < row["baseline"]
            row["model better?"] = "same" if np.isclose(row["model"], row["baseline"]) else "yes" if better else "no"
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def tier_text(tier) -> str:
    """Table cell style: the tier word in its tier colour (the word itself stays, so colour is never the only cue)."""
    return f"color: {TIER_COLORS[tier]}; font-weight: 600" if tier in TIER_COLORS else ""


def queue_order(df: pd.DataFrame) -> pd.DataFrame:
    """Worst first: tier, then probability of Critical, then forecast risk."""
    return (df.assign(_rank=df.tier_final.map(TIER_RANK))
            .sort_values(["_rank", "p_critical", "risk_forecast"], ascending=False).drop(columns="_rank"))


def tier_bands(fig, axis: str, critical: tuple, elevated: tuple):
    add = fig.add_hrect if axis == "y" else fig.add_vrect
    for (lo, hi), t in ((critical, CRITICAL), (elevated, ELEVATED)):
        add(**{f"{axis}0": lo, f"{axis}1": hi}, fillcolor=TIER_COLORS[t], opacity=0.12, line_width=0, layer="below")


def sphere(radius: float, tier: str, name: str) -> go.Surface:
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    return go.Surface(x=radius * np.cos(u) * np.sin(v), y=radius * np.sin(u) * np.sin(v), z=radius * np.cos(v),
                      opacity=0.15, showscale=False, colorscale=[[0, TIER_COLORS[tier]], [1, TIER_COLORS[tier]]],
                      name=name, hoverinfo="name")

