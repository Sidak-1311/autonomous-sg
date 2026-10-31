"""End-to-end check of the data pipeline. Re-run after every change.

    .venv/bin/python scripts/verify_pipeline.py

Runs the unit tests, scores the real data, then checks the written files
against the room's data contract. Exits non-zero on the first failure.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ingest import PROCESSED_DIR  # noqa: E402
from src.score import CDM_COLS, EVENT_COLS  # noqa: E402
from src.tiering import CRITICAL, TIER_RANK, TIERS, tier_array  # noqa: E402

failures = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        failures.append(msg)


def run(cmd):
    print("$", " ".join(cmd))
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"verify: '{' '.join(cmd)}' failed")


run([sys.executable, "-m", "pytest", "-q", "tests"])
run([sys.executable, "-m", "src.score"])

cdms = pd.read_parquet(PROCESSED_DIR / "cdms.parquet")
ev = pd.read_parquet(PROCESSED_DIR / "events_scored.parquet")
metrics = json.loads((PROCESSED_DIR / "metrics.json").read_text())

print("cdms.parquet")
check(list(cdms.columns) == CDM_COLS, "columns match contract")
check(set(cdms["split"]) == {"train", "test"}, "both splits present")
check(cdms["miss_distance"].ge(0).all() and cdms["miss_distance"].max() > 5_000,
      "miss_distance non-negative and in metres")
check(cdms["risk"].between(-30, 0).all(), "risk within [-30, 0]")
check((cdms["tier"] == tier_array(cdms["risk"], cdms["miss_distance"])).all(),
      "per-CDM tier matches tier rules")

print("events_scored.parquet")
check(list(ev.columns) == EVENT_COLS, "columns match contract")
check(not ev.duplicated(["split", "event_id"]).any(), "one row per (split, event_id)")
check(ev["time_to_tca"].ge(2.0).all(), "inputs only from CDMs >= 2 days before TCA")
for col in ["tier_now", "tier_forecast", "tier_final"]:
    check(ev[col].isin(TIERS).all(), f"{col} only SAFE/ELEVATED/CRITICAL")
required = ["risk_latest", "miss_latest_m", "risk_forecast", "risk_forecast_q90",
            "miss_forecast_m", "reason"]
check(not ev[required].isna().any().any(), "no NaN in forecast/tier columns")
withheld = ev["forecast_source"] == "withheld"
check((withheld == ev["needs_review"]).all(), "forecast withheld exactly for needs_review events")
check(ev.loc[withheld, "p_critical"].isna().all() and (ev.loc[withheld, "drivers"] == "").all()
      and (ev.loc[withheld, "risk_forecast"] == ev.loc[withheld, "risk_latest"]).all(),
      "withheld events carry no model forecast")
check(ev.loc[~withheld, "p_critical"].between(0, 1).all(), "p_critical in [0, 1] otherwise")
check(ev["forecast_source"].isin(["baseline", "model", "withheld"]).all(), "forecast_source values")
if (ROOT / "models" / "conformal.json").exists():
    check(ev["risk_lo"].notna().all() and (ev["risk_lo"] <= ev["risk_latest"]).all()
          and (ev["risk_latest"] <= ev["risk_hi"]).all() and ev["risk_hi"].le(0).all(),
          "risk_lo <= risk_latest <= risk_hi <= 0")
if (ev["forecast_source"] == "model").any():
    in_sample = ev["forecast_in_sample"]
    check(not (in_sample & (ev["split"] == "test")).any(), "test rows never in-sample")
    check(ev.loc[~withheld, "p_critical_storm"].ge(ev.loc[~withheld, "p_critical"] - 1e-9).mean() > 0.5,
          "storm stress mostly raises P(crit)")
rank = lambda s: s.map(TIER_RANK)  # noqa: E731
check((rank(ev["tier_final"]) >= rank(ev["tier_now"])).all()
      and (rank(ev["tier_final"]) >= rank(ev["tier_forecast"])).all(),
      "tier_final is the worse of now and forecast")
check(rank(ev["tier_final"]).is_monotonic_decreasing, "queue sorted worst tier first")
test = ev[ev["split"] == "test"]
check(test["risk_final"].isna().all() and test["tier_actual"].isna().all(),
      "test outcomes stay hidden")
known = ev[ev["tier_actual"].notna()]
check(len(known) > 1000 and known["risk_final"].notna().all(), "train outcomes present")
crit = ev["tier_final"] == CRITICAL
check(ev.loc[crit, "maneuver_dv_mps"].notna().all() and ev.loc[~crit, "maneuver_dv_mps"].isna().all(),
      "maneuver suggested exactly for Critical events")
check(np.isfinite(ev.loc[crit, "maneuver_dv_mps"]).all() and ev.loc[crit, "maneuver_dv_mps"].ge(0).all(),
      "maneuver dv finite and non-negative")

check(ev["needs_review"].dtype == bool and (ev["review_reason"] != "").eq(ev["needs_review"]).all(),
      "needs_review set exactly when a review reason exists")
check(ev.loc[ev["needs_review"], "reason"].str.startswith("INSUFFICIENT EVIDENCE").all(),
      "refused events say so in the reason")
check(not ev["reason"].str.contains("could reach").any(), "dilution is a note, not a warning")

print("metrics.json")
check("baseline" in metrics or not (ROOT / "src" / "metrics.py").exists(),
      "baseline metrics written once src/metrics.py exists")

print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
