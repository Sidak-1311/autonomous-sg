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
| `src/features.py` | One row per event from CDMs with `time_to_tca ≥ 2` d: latest state, risk/miss trends, sigma shrink, geometry, physics Pc prior; target = tier of the event's last CDM (the organisers' labels; `scripts/check_labels.py` checks them event by event) |
| `src/physics.py` | Short-encounter Pc approximation, maneuver sizing, covariance-dilution check |
| `src/tiering.py` | Tier rules, reason text |
| `src/train.py` | scikit-learn HistGradientBoosting; fixed 80/20 holdout by `event_id` for the reported numbers, 5-fold GroupKFold on `event_id` inside it; P(Critical) and P(Elevated or worse) classifiers, isotonic-calibrated on out-of-fold scores, giving P(Safe) / P(Elevated) / P(Critical) that sum to 1; Critical threshold set for 95% recall; risk and miss regressors that must beat the latest-value baseline or fall back to it; split-conformal interval; storm stress test |
| `src/metrics.py` | Critical recall/precision, F2, MSE on high-risk events, ESA loss, queue depth at 95% recall; bootstrap 95% CIs |
| `src/score.py` | Applies all of the above, writes `events_scored.parquet` / `cdms.parquet` / `metrics.json` |
| `dashboard/app.py` | Streamlit + Plotly: operations queue, event detail (risk/miss history, 3D encounter geometry), fleet overview, 24-hour danger-zone replay, evaluation |

Train rows in the dashboard carry **out-of-fold** forecasts (`models/oof_predictions.parquet`),
so nothing shown on train events was predicted by a model that saw them.

## Results (organisers' labels: 11,942 usable train events, 834 Critical)

Target: the tier of each event's last CDM, predicted from CDMs at least 2 days before TCA.
Headline numbers come from a **fixed 80/20 split of train events by `event_id`** (seed 42):
models, calibration and the alert threshold are fit on the 80% only, then scored once on
the untouched 2,389 events (154 Critical). The baseline is today's practice: the tier of the
latest CDM at the 2-day decision point.

| Held out (2,389 events) | Latest-tier baseline | Triage-TCA | |
|---|---|---|---|
| Critical recall | 0.877 [0.82, 0.93] | **0.981** [0.96, 1.00] | 151 of 154 Criticals caught, vs 135 |
| Critical precision | 0.808 [0.74, 0.87] | 0.487 [0.43, 0.54] | price: about 2x the Critical list |
| Critical recall, hard subset (final CDM < 2 d, 1,741 events) | 0.829 | **0.973** | where the tier actually changes |

Confusion matrices on the holdout (rows = actual tier at TCA, columns = issued tier):

| Model | Safe | Elevated | Critical | | Baseline | Safe | Elevated | Critical |
|---|---|---|---|---|---|---|---|---|
| **Safe** | 1,714 | 48 | 23 | | **Safe** | 1,721 | 57 | 7 |
| **Elevated** | 29 | 285 | 136 | | **Elevated** | 31 | 394 | 25 |
| **Critical** | 2 | 1 | **151** | | **Critical** | 3 | 16 | 135 |

Across all 11,942 events, out of fold:

| | Latest-tier baseline | Triage-TCA | |
|---|---|---|---|
| Critical recall | 0.866 [0.84, 0.89] | **0.953** [0.94, 0.97] | |
| Critical precision | 0.851 [0.83, 0.88] | 0.553 [0.53, 0.58] | |
| Queue depth to catch 95% of Criticals | 2,176 [1,883, 2,341] | **1,440 [1,331, 1,763]** | 34% fewer reviews [22%, 39%] |
| P(Critical) Brier score | 0.065 (climatology) | **0.016** (0.018 before calibration) | P(crit) is a calibrated probability |
| P(Elevated or worse) Brier score | 0.188 (climatology) | **0.030** | |
| F2 / MSE on high-risk / ESA loss | 0.806 / 3.50 / 4.35 | same | risk forecast falls back to baseline (below) |

**Hidden test set (lockbox).** The ESA Kelvins challenge later published the final CDM of
every test event (Zenodo 4463683). We use it only to score the frozen model, never to train,
tune, pick thresholds or choose between models: Critical recall **0.927** vs 0.824 baseline,
precision 0.452 vs 0.855 (2,167 events, 193 Critical; 7 Criticals issued Safe, listed with
causes in the Evaluation tab).

Brackets are bootstrap 95% CIs (queue depth: paired bootstrap over events). Queue depth
is measured in the dashboard's queue order (tier, then P(crit), then forecast risk) on
out-of-fold train rows; the baseline orders by latest tier, then latest Pc, then soonest TCA.
Events whose last CDM is itself 2 or more days out have a final tier equal to the
decision-point tier, so they flatter both columns; the hard subset leaves them out.
Before the organisers' label file we trained on the ESA protocol's subset (final CDM within
1 day of TCA, 8,293 events); `final_ttca < 1` in `events_scored` recovers it.
The 90% conformal interval on Pc at TCA covers 90.1%
of out-of-fold outcomes; because it is equal-tailed, its top is shown as a **one-sided 95%
upper bound** and the uninformative lower end is not displayed. On `test_data.csv` the outcomes are hidden (data stop at
the 2-day cutoff), so test events are scored without them: 396 Critical, 305
Elevated, 1,466 Safe of 2,167, with 91 refused for insufficient evidence.
Tried and rejected on the holdout: ten extra trend features (last-3-CDM slopes,
extrapolated miss, miss/σ) gave average precision 0.872 vs 0.871 and lower recall, so the
simpler model stays.

## Where it fails, and how you'd know

- **The risk forecast does not beat "final risk = latest risk".** The regressor's ESA loss
  was 9.47 vs the baseline's 4.35 out of fold: most events drift down or collapse to the
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
- **Low alert threshold.** Catching 95% of Criticals needs a calibrated P(crit) ≥ 4.9%, so
  many events are raised at single-digit probabilities. Every such reason states the
  threshold explicitly ("P(crit) 33.4% ≥ alert threshold 4.9%, set for 95% recall").
- **Misses are geometry closing late.** All 3 Criticals missed on the holdout (43 of 44 out
  of fold) became Critical by geometry alone: the miss distance closed below 1,000 m after
  the 2-day decision point while Pc stayed below 10⁻⁴ (e.g. event 1936: 13,748 m → 657 m,
  issued Safe; `models/error_analysis_holdout.json`). Raising every event with a forecast
  miss under 1,500 m would catch 12 more out of fold for 92 more alerts (~8 reviews per
  catch), so that dial is left to the analyst.
- **Storm sensitivity is not a storm forecast.** With F10/AP at the data's 99th percentile,
  38 test events flip to Critical; this likely reflects solar-cycle correlations in
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
