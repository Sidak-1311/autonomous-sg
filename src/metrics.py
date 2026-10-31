"""Scores for a forecast of each event's final state, against the truth at TCA.

Two views:
- Critical tier (what operators act on): recall and precision of tier CRITICAL.
- High-risk regression (ESA Kelvins challenge style): F2 on "risk >= -6" and
  MSE on the truly high-risk events; loss = mse_hr / f2 (lower is better).
Confidence intervals come from a bootstrap over events.
"""
import numpy as np

from src.tiering import CRITICAL, RISK_ELEVATED, RISK_FLOOR, TIERS, tier_array

HIGH_RISK = RISK_ELEVATED
METRICS = ["recall_critical", "precision_critical", "f2", "mse_hr", "loss"]


def _scores(true_crit, pred_crit, true_hr, pred_hr, sq_err):
    """Metric values for boolean/float arrays of shape (..., n_events)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        tp = (true_crit & pred_crit).sum(-1)
        recall = tp / true_crit.sum(-1)
        precision = tp / pred_crit.sum(-1)
        tp_hr = (true_hr & pred_hr).sum(-1)
        p_hr = tp_hr / pred_hr.sum(-1)
        r_hr = tp_hr / true_hr.sum(-1)
        f2 = np.nan_to_num(5 * p_hr * r_hr / (4 * p_hr + r_hr))
        mse_hr = (sq_err * true_hr).sum(-1) / true_hr.sum(-1)
        loss = mse_hr / f2
    return {"recall_critical": recall, "precision_critical": precision,
            "f2": f2, "mse_hr": mse_hr, "loss": loss}


def evaluate(risk_final, miss_final_m, risk_pred, tier_pred, n_boot=1000, seed=0) -> dict:
    """Score predictions (risk_pred: log10 Pc, tier_pred: tier) for each event."""
    risk_final = np.asarray(risk_final, float)
    risk_pred = np.clip(np.asarray(risk_pred, float), RISK_FLOOR, 0.0)
    true_crit = tier_array(risk_final, miss_final_m) == CRITICAL
    pred_crit = np.asarray(tier_pred, dtype=object) == CRITICAL
    true_hr = risk_final >= HIGH_RISK
    pred_hr = risk_pred >= HIGH_RISK
    sq_err = (risk_pred - risk_final) ** 2

    point = _scores(true_crit, pred_crit, true_hr, pred_hr, sq_err)
    out = {k: float(v) for k, v in point.items()}
    out["n_critical"] = int(true_crit.sum())
    out["n_events"] = int(len(risk_final))

    ci = {}
    if n_boot and len(risk_final):
        rng = np.random.default_rng(seed)
        ci = {k: [] for k in METRICS}
        # Chunked so memory stays small on the full train set
        for start in range(0, n_boot, 100):
            idx = rng.integers(0, len(risk_final), size=(min(100, n_boot - start), len(risk_final)))
            boot = _scores(true_crit[idx], pred_crit[idx], true_hr[idx], pred_hr[idx], sq_err[idx])
            for k in METRICS:
                ci[k].append(boot[k])
        ci = {k: np.concatenate(v) for k, v in ci.items()}
        ci = {k: [float(x) for x in np.nanpercentile(v[np.isfinite(v)], [2.5, 97.5])]
              if np.isfinite(v).any() else [float("nan")] * 2 for k, v in ci.items()}
    out["ci95"] = ci
    return out


def _depth(y_sorted: np.ndarray, recall: float) -> np.ndarray:
    """Events reviewed top-down until `recall` of the Criticals are found, per row."""
    hits = np.cumsum(y_sorted, axis=-1)
    need = np.ceil(recall * hits[..., -1:] - 1e-9)
    return (hits < need).sum(-1) + 1


def queue_depth(is_critical, rank_model, rank_baseline, recall=0.95, n_boot=1000, seed=0) -> dict:
    """Queue depth needed to catch `recall` of the Criticals under two orderings
    (rank 0 = top of the queue), with a paired bootstrap over events.

    Each resample draws events with replacement and reviews them in each
    ordering, so the model/baseline difference is measured on the same events.
    Depths are reported as counts on the original number of events.
    """
    y = np.asarray(is_critical, bool)
    n = len(y)
    out = {"recall": recall, "n_events": int(n), "n_critical": int(y.sum())}
    if n == 0 or not y.any():
        return {**out, "model": None, "baseline": None, "reduction": None, "ci95": {}}
    order_m, order_b = np.argsort(rank_model, kind="stable"), np.argsort(rank_baseline, kind="stable")
    d_m, d_b = int(_depth(y[order_m], recall)), int(_depth(y[order_b], recall))
    out.update(model=d_m, baseline=d_b, model_share=d_m / n, baseline_share=d_b / n,
               reduction=1 - d_m / d_b)

    rng = np.random.default_rng(seed)
    pos_m = np.empty(n, int)
    pos_m[order_m] = np.arange(n)
    pos_b = np.empty(n, int)
    pos_b[order_b] = np.arange(n)
    boots = {"model": [], "baseline": [], "reduction": []}
    for start in range(0, n_boot, 50):
        idx = rng.integers(0, n, size=(min(50, n_boot - start), n))
        keep = y[idx].any(axis=1)
        idx = idx[keep]
        if not len(idx):
            continue
        rows = np.arange(len(idx))[:, None]
        bm = _depth(y[idx[rows, np.argsort(pos_m[idx], axis=1)]], recall)
        bb = _depth(y[idx[rows, np.argsort(pos_b[idx], axis=1)]], recall)
        boots["model"].append(bm)
        boots["baseline"].append(bb)
        boots["reduction"].append(1 - bm / bb)
    out["ci95"] = {k: [float(x) for x in np.percentile(np.concatenate(v), [2.5, 97.5])]
                   for k, v in boots.items() if v}
    return out


def confusion(tier_true, tier_pred) -> dict:
    """3x3 tier confusion matrix (rows = actual, columns = predicted, SAFE/ELEVATED/CRITICAL)
    with per-class recall, precision and counts. A class with no events gets None, not 0."""
    labels = list(TIERS)
    t, p = np.asarray(tier_true, dtype=object), np.asarray(tier_pred, dtype=object)
    m = np.array([[int(np.sum((t == a) & (p == b))) for b in labels] for a in labels])
    rows, cols = m.sum(axis=1), m.sum(axis=0)
    return {
        "labels": labels,
        "matrix": m.tolist(),
        "recall": {c: (float(m[i, i] / rows[i]) if rows[i] else None) for i, c in enumerate(labels)},
        "precision": {c: (float(m[i, i] / cols[i]) if cols[i] else None) for i, c in enumerate(labels)},
        "n": {c: int(rows[i]) for i, c in enumerate(labels)},
    }
