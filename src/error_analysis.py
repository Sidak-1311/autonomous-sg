"""Why the triage misses Criticals and raises false alarms.

    .venv/bin/python -m src.error_analysis [--holdout models/holdout_ids.json]

Reads data/processed/events_scored.parquet + cdms.parquet, takes the true final
tier from the last train CDM (smallest time_to_tca, the organisers' definition),
and compares it with the tier we issue at the 2-day decision point (tier_final).
Every missed Critical gets one primary cause (first that applies, in CAUSES
order) plus all flags that hold; the same for false alarms. Writes
models/error_analysis.json. Pass --holdout (a JSON list of train event_ids) to
restrict to a held-out split.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.tiering import CRITICAL, ELEVATED, MISS_CRITICAL_M, RISK_CRITICAL, RISK_ELEVATED, SAFE, TIERS, tier_array

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "error_analysis.json"

CUTOFF_DAYS = 2.0
JUMP_ORDERS = 2.0      # Pc rising by >= 2 orders of magnitude after the decision point
NEAR_FACTOR = 2.0      # P(crit) within a factor 2 below the alert threshold

# Primary causes for a missed Critical, checked in this order.
FN_CAUSES = {
    "withheld_for_review": "forecast withheld (stale or single CDM) and sent to an analyst: refused, not silent",
    "no_evidence_after_cutoff": "Pc jumped from the -30 floor (no measurable risk) at the decision point",
    "late_jump": f"Pc rose >= {JUMP_ORDERS:g} orders of magnitude after the decision point",
    "miss_driven": "Critical only because the final miss < 1,000 m; Pc stayed < 1.0×10⁻⁴",
    "near_threshold": f"P(Critical) within {NEAR_FACTOR:g}x below the alert threshold",
    "other": "none of the above",
}
FP_CAUSES = {
    "critical_now": "already Critical at the decision point, then resolved before TCA",
    "pc_fell": f"Pc fell >= {JUMP_ORDERS:g} orders of magnitude after the decision point",
    "miss_opened": "miss < 1,000 m at the decision point, > 1,000 m at TCA",
    "alert_threshold": "raised by P(Critical) above the 95%-recall alert threshold",
    "other": "none of the above",
}


def final_outcomes(cdms: pd.DataFrame) -> pd.DataFrame:
    """Per train event: the last CDM (smallest time_to_tca), its tier and time."""
    tr = cdms[cdms["split"] == "train"]
    last = tr.loc[tr.groupby("event_id")["time_to_tca"].idxmin()]
    out = pd.DataFrame({
        "event_id": last["event_id"].to_numpy(),
        "risk_true": last["risk"].to_numpy(dtype=float),
        "miss_true_m": last["miss_distance"].to_numpy(dtype=float),
        "final_ttca": last["time_to_tca"].to_numpy(dtype=float),
    })
    out["tier_true"] = tier_array(out["risk_true"], out["miss_true_m"])
    return out


def _fn_flags(d: pd.DataFrame, threshold: float) -> pd.DataFrame:
    jump = d["risk_true"] - d["risk_latest"]
    return pd.DataFrame({
        "withheld_for_review": d["needs_review"].fillna(False).astype(bool),
        "no_evidence_after_cutoff": d["risk_latest"] <= -30 + 1e-9,
        "late_jump": jump >= JUMP_ORDERS,
        "miss_driven": (d["miss_true_m"] < MISS_CRITICAL_M) & (d["risk_true"] < RISK_CRITICAL),
        "near_threshold": (d["p_critical"] >= threshold / NEAR_FACTOR) & (d["p_critical"] < threshold),
    }, index=d.index)


def _fp_flags(d: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.DataFrame({
        "critical_now": d["tier_now"] == CRITICAL,
        "pc_fell": (d["risk_latest"] - d["risk_true"]) >= JUMP_ORDERS,
        "miss_opened": (d["miss_latest_m"] < MISS_CRITICAL_M) & (d["miss_true_m"] >= MISS_CRITICAL_M),
        "alert_threshold": d["p_critical"] >= threshold,
    }, index=d.index)


def _primary(flags: pd.DataFrame, causes: dict) -> pd.Series:
    out = pd.Series("other", index=flags.index, dtype=object)
    for c in reversed([c for c in causes if c != "other"]):
        out[flags[c].to_numpy(dtype=bool)] = c
    return out


def _summary(d: pd.DataFrame, flags: pd.DataFrame, causes: dict, n_ref: int) -> dict:
    prim = _primary(flags, causes)
    by_cause = {c: {"n": int((prim == c).sum()),
                    "flag_n": int(flags[c].sum()) if c in flags else int((prim == c).sum()),
                    "meaning": causes[c]} for c in causes}
    obj = d["c_object_type"].fillna("UNKNOWN") if len(d) else pd.Series(dtype=object)
    return {
        "n": int(len(d)),
        "share_of_reference": float(len(d) / n_ref) if n_ref else float("nan"),
        "by_cause": by_cause,
        "by_object_type": {str(k): int(v) for k, v in obj.value_counts().items()},
        "predicted_tier": {t: int((d["tier_final"] == t).sum()) for t in TIERS},
        "median_gap_days": float((d["time_to_tca"] - d["final_ttca"]).median()) if len(d) else float("nan"),
    }


def _examples(d: pd.DataFrame, prim: pd.Series, k: int = 3) -> list:
    rows = []
    for cause, g in d.assign(cause=prim).groupby("cause"):
        for _, r in g.sort_values("risk_true", ascending=False).head(k).iterrows():
            rows.append({"event_id": int(r["event_id"]), "cause": cause,
                         "risk_decision": float(r["risk_latest"]), "risk_final": float(r["risk_true"]),
                         "miss_decision_m": float(r["miss_latest_m"]), "miss_final_m": float(r["miss_true_m"]),
                         "ttca_decision": float(r["time_to_tca"]), "ttca_final": float(r["final_ttca"]),
                         "p_critical": float(r["p_critical"]), "tier_issued": r["tier_final"]})
    return rows


def analyse(ev: pd.DataFrame, finals: pd.DataFrame, threshold: float) -> dict:
    """ev: train rows of events_scored; finals: final_outcomes(cdms). Returns the report dict."""
    d = ev.merge(finals, on="event_id", how="inner")
    cm = pd.crosstab(pd.Categorical(d["tier_true"], TIERS), pd.Categorical(d["tier_final"], TIERS),
                     dropna=False)
    is_crit = d["tier_true"] == CRITICAL
    fn = d[is_crit & (d["tier_final"] != CRITICAL)]
    fp = d[~is_crit & (d["tier_final"] == CRITICAL)]
    fn_flags, fp_flags = _fn_flags(fn, threshold), _fp_flags(fp, threshold)
    late = d["final_ttca"] < CUTOFF_DAYS
    n_crit = int(is_crit.sum())
    silent = int((~fn_flags["withheld_for_review"]).sum())
    return {
        "n_events": int(len(d)),
        "n_critical": n_crit,
        "alert_threshold": float(threshold),
        "confusion_matrix": {"rows_actual_cols_issued": TIERS,
                             "counts": cm.to_numpy().astype(int).tolist()},
        "recall_critical": float((is_crit & (d["tier_final"] == CRITICAL)).sum() / n_crit) if n_crit else float("nan"),
        "recall_critical_final_cdm_under_2d": float(
            ((is_crit & late) & (d["tier_final"] == CRITICAL)).sum() / (is_crit & late).sum())
            if (is_crit & late).any() else float("nan"),
        "n_final_cdm_under_2d": int(late.sum()),
        "silent_missed_critical": silent,
        "recall_critical_incl_review": float(1 - silent / n_crit) if n_crit else float("nan"),
        "missed_critical": _summary(fn, fn_flags, FN_CAUSES, n_crit),
        "false_alarm": _summary(fp, fp_flags, FP_CAUSES, int((d["tier_final"] == CRITICAL).sum())),
        "missed_examples": _examples(fn, _primary(fn_flags, FN_CAUSES)),
        "false_alarm_examples": _examples(fp, _primary(fp_flags, FP_CAUSES), k=2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=Path, help="JSON list of held-out train event_ids")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    import joblib
    threshold = float(joblib.load(ROOT / "models" / "forecaster.joblib")["crit_threshold"])
    ev = pd.read_parquet(PROCESSED / "events_scored.parquet")
    ev = ev[ev["split"] == "train"]
    if a.holdout:
        ev = ev[ev["event_id"].isin(json.loads(a.holdout.read_text()))]
    report = analyse(ev, final_outcomes(pd.read_parquet(PROCESSED / "cdms.parquet")), threshold)
    report["scope"] = f"holdout ({a.holdout.name})" if a.holdout else "all train events (OOF where trained)"
    a.out.write_text(json.dumps(report, indent=2, default=lambda x: None if x != x else x))
    m = report["missed_critical"]
    print(f"{report['n_critical']} Criticals, missed {m['n']} (recall {report['recall_critical']:.3f}); "
          f"{report['silent_missed_critical']} silent (not sent to review); "
          f"false alarms {report['false_alarm']['n']}")
    for c, v in m["by_cause"].items():
        print(f"  {c:26s} {v['n']:4d}  (flag on {v['flag_n']})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
