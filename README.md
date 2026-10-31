# Triage-TCA: orbital collision risk evaluator

A triage tool for **conjunction analysts**, the people who read the thousands of
conjunction data messages (CDMs) a satellite operator receives each day and decide
which close approaches need a maneuver. Today that means checking warnings one by
one against Pc and miss-distance thresholds, and committing to a burn at least
2 days before closest approach (TCA), before the most reliable CDMs arrive.

Triage-TCA reads the CDM history of every encounter and, at that 2-day decision point:

1. **Tiers it by rule** (Safe / Elevated / Critical, NASA/SP-20205011318 and ESA SDO
   thresholds). Deterministic and auditable.
2. **Forecasts where it is heading** with a gradient-boosted model, so an event that is
   Elevated today but trending to Critical is flagged while there is still time to act.
3. **Explains, bounds and refuses**: every event gets a reason, its top drivers, a 90%
   interval on its final Pc, and an explicit "insufficient evidence, analyst must review"
   when the data are too thin to forecast.
4. **Suggests a maneuver** for Critical events: the radial offset at TCA that clears both
   rules, and the approximate burn.

The analyst stays in control: the tool ranks, explains and suggests; it never commands a burn.
Each event has an **Analyst decision** (Maneuver / Monitor / Dismiss, signed; dismissing a
Critical or a refusal requires a note), logged to `data/decisions.csv` with what the system
showed at the time.

## Risk tiers (dataset units: `risk` = log10 Pc, `miss_distance` in metres)

| Tier | Rule | Action |
|---|---|---|
| Critical | risk ≥ -4.0 **or** miss < 1,000 m | Maneuver required |
| Elevated | -6.0 ≤ risk < -4.0 **or** 1,000 m ≤ miss ≤ 5,000 m | Watch, re-check each CDM |
| Safe | risk < -6.0 **and** miss > 5,000 m | Archive |

Critical is checked first, so overlapping rules resolve to the worse tier. The rules live
only in [`src/tiering.py`](src/tiering.py).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the challenge data in data/: data/train_data.csv, data/test_data.csv (git-ignored)
.venv/bin/python -m src.score                 # ~15 s: writes data/processed/*.parquet + metrics.json
.venv/bin/streamlit run dashboard/app.py      # the dashboard
.venv/bin/python scripts/verify_pipeline.py   # tests + scoring + data-contract checks
```

The trained model is committed in `models/`, so no training is needed to run the demo.
To retrain (~5 min on a laptop): `.venv/bin/python -m src.train`.

Optional: the dashboard can write a plain-language event brief with Gemini. Set
`GEMINI_API_KEY` in the environment or a git-ignored `.env`. The LLM only rephrases the
reason, drivers and numbers already computed; it never sets a tier or a probability, and
the dashboard works fully without a key.

## How it works

```
CSV ──► ingest ──► features ──► rules ─────────────┐
       validate    per event    tier_now           ├─► tier_final = worse of the two ─► queue
       (drop bad   at the 2-day                    │      + reason, drivers, 90% interval,
        rows, flag cutoff      model ─► tier_forecast      refusal, maneuver
        gaps)                  P(crit), q90 risk,
                               miss forecast
```

| Module | Role |
|---|---|
| `src/ingest.py` | Load the 13 chosen columns (+ relative velocity), validate ranges, fill and flag missing sigmas / space weather |
| `src/features.py` | One row per event from CDMs with `time_to_tca ≥ 2` d: latest state, risk/miss trends, sigma shrink, geometry, physics Pc prior; targets from the last CDM |
| `src/physics.py` | Short-encounter Pc approximation, maneuver sizing, covariance-dilution check |
| `src/tiering.py` | Tier rules, reason text |
| `src/train.py` | scikit-learn HistGradientBoosting, 5-fold GroupKFold on `event_id`; P(Critical) classifier, isotonic-calibrated on out-of-fold scores, threshold set for 95% recall; risk and miss regressors that must beat the latest-value baseline or fall back to it; split-conformal interval; storm stress test |
| `src/metrics.py` | Critical recall/precision, F2, MSE on high-risk events, ESA loss, queue depth at 95% recall; bootstrap 95% CIs |
| `src/score.py` | Applies all of the above, writes `events_scored.parquet` / `cdms.parquet` / `metrics.json` |
| `dashboard/app.py` | Streamlit + Plotly: operations queue, event detail (risk/miss history, 3D encounter geometry), fleet overview, 24-hour danger-zone replay, evaluation |

Train rows in the dashboard carry **out-of-fold** forecasts (`models/oof_predictions.parquet`),
so nothing shown on train events was predicted by a model that saw them.

## Results (8,293 train events with a known outcome, 551 Critical; out-of-fold)

| | Latest-value baseline | Triage-TCA | |
|---|---|---|---|
| Critical recall | 0.820 [0.79, 0.85] | **0.951** [0.93, 0.97] | catches 13 points more Criticals |
| Critical precision | 0.800 [0.77, 0.83] | 0.488 [0.46, 0.52] | price: about 2x the Critical list |
| Queue depth to catch 95% of Criticals | 1,514 [1,362, 1,798] (18.3%) | **1,198 [999, 1,456] (14.4%)** | 21% fewer reviews [6%, 37%] |
| P(Critical) Brier score | 0.062 (climatology) | **0.019** (0.022 before calibration) | P(crit) is a calibrated probability |
| F2 / MSE on high-risk / ESA loss | 0.411 / 8.01 / 19.5 | same | risk forecast falls back to baseline (below) |

Brackets are bootstrap 95% CIs (queue depth: paired bootstrap over events, so the
reduction interval is wide; quote it with the interval). Queue depth is measured in the
dashboard's queue order (tier, then P(crit), then forecast risk) on out-of-fold train rows;
the baseline orders by latest tier, then latest Pc, then soonest TCA. Never quote the 6%
figure from an earlier in-sample run. The 90% conformal interval on Pc at TCA covers 90.2%
of out-of-fold outcomes; because it is equal-tailed, its top is shown as a **one-sided 95%
upper bound** and the uninformative lower end is not displayed. On `test_data.csv` the outcomes are hidden (data stop at
the 2-day cutoff), so test events are scored but not evaluated: 394 Critical, 307
Elevated, 1,466 Safe of 2,167, with 91 refused for insufficient evidence.

## Where it fails, and how you'd know

- **The risk forecast does not beat "final risk = latest risk".** The regressor's ESA loss
  was 278 vs the baseline's 19.5 out of fold: most events drift down or collapse to the
  -30 floor while the few dangerous ones stay flat, so a learned change mispredicts exactly
  the events that matter (the same finding as ESA's Kelvins challenge, where 12 of 97
  teams beat this baseline). The pipeline measures this and falls back automatically;
  the dashboard's Evaluation tab shows "same" for those metrics. The model's value is in
  the P(Critical) classifier, not in the point forecast.
- **The covariance-dilution flag does not predict danger.** We expected events whose Pc is
  low only because tracking is poor to be hidden risks. On train outcomes none of the
  279 flagged Safe events ended Critical, and flagged events overall ended Critical less
  often than unflagged ones (1.9% vs 6.8%), because tracking improves far past the Pc peak. It is shown as a tracking-quality note with this measured result,
  never as a warning.
- **Low alert threshold.** Catching 95% of Criticals needs a calibrated P(crit) ≥ 3.3%, so
  many events are raised at single-digit probabilities. Every such reason states the
  threshold explicitly ("P(crit) 64.2% ≥ alert threshold 3.3%, set for 95% recall").
- **Storm sensitivity is not a storm forecast.** With F10/AP at the data's 99th percentile,
  350 test events flip to Critical; this likely reflects solar-cycle correlations in
  2015–2019 data, not drag physics. Shown as sensitivity only.
- **Refusals.** Events with one CDM, a stale latest CDM (> 5 days before TCA) or missing
  covariance get no forecast; their latest tier is wrong 11.7% of the time vs 6.9% otherwise.

## Limitations

- Decision support only: a qualified analyst signs off every Critical call.
- Very few events reach Pc ≥ 1e-4; most Criticals come from the 1 km rule.
- Anonymised data: no absolute times, orbits or identities. The 3D view is relative
  encounter geometry (RTN frame) and the 24-hour view is a replay.
- Pc is inherited from the CDMs (ESA's computation); we forecast it, we don't recompute it.
  Our physics Pc uses radial sigmas only and is a feature and a maneuver-sizing aid, never
  a tier input.
- Maneuver sizes assume a ~95-minute LEO orbit and a tangential burn half an orbit before
  TCA (dv ≈ n·|dr|/4); typical suggestions are 0.16 to 0.27 m/s.
- Data are 2015–2019 ESA CDMs; LEO is far more crowded now, so a deployment would retrain.

## Sources

- Uriot et al., *Spacecraft Collision Avoidance Challenge: design and results of a machine
  learning competition* (ESA): dataset, metric, latest-risk baseline.
- ESA Kelvins Collision Avoidance Challenge: column definitions, 2-day cutoff.
- NASA Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook
  (NASA/SP-20205011318) and ESA Space Debris Office practice: tier thresholds.
