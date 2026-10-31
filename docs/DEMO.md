# Demo script (under 3 minutes)

Setup before going on stage: `git pull`, `.venv/bin/python -m src.score`,
`.venv/bin/streamlit run dashboard/app.py`, dashboard filtered to the **test** split.
All event ids below are test events.

## 0:00 to 0:25: the user and the status quo

"A conjunction analyst gets thousands of collision warnings a day and must decide on a
maneuver at least two days before closest approach, before the best data arrive. Today
that's thresholds checked by hand. Triage-TCA ranks every encounter, explains it, and says
when it doesn't know."

## 0:25 to 0:55: Operations queue

- KPI strip: 2,167 encounters; 394 Critical, 307 Elevated, 1,466 Safe archived; 91 need
  analyst review.
- "Safe events leave the queue. Refusals never do, whatever their tier."
- Point at the alert-fatigue curve: "To catch 95% of the dangerous events, today's
  thresholds make you review the top 18% of the list; ours, 14%. That's about 21% fewer
  reviews, measured out of fold."

## 0:55 to 1:35: a Critical event, **event 891**

- Event detail: risk history above the -4 band, miss distance 0.50 km inside the 1 km line,
  3D encounter view with the debris passing inside the 1 km sphere.
- Reason: *"Critical now (Pc 1e-2.9 ≥ 1e-4 and miss 0.50 km < 1 km) … final Pc at most
  1e-1.6 with 90% confidence."* Drivers: *miss distance small, debris position precise.*
- Suggested maneuver: shift radial separation by ~830 m, about a 0.23 m/s burn half an orbit
  before TCA. "A suggestion; the analyst signs off."
- Analyst decision: choose **Maneuver**, sign, submit; it's logged with what the system showed.
  (Optional, if a key is set: open the Gemini brief, which only rephrases these numbers
  and shows exactly which fields were sent.)

## 1:35 to 2:05: what a threshold filter misses, **event 170**

- "Elevated now: miss 1.09 km, Pc tiny. A threshold filter archives the debate here."
- Forecast Critical: P(crit) 95%, miss forecast 1.07 km, *"last CDM raised risk"*.
- "This is why the model exists: it flags the event while there's still time to plan a burn."

## 2:05 to 2:25: refusal, **event 1152**

- *"INSUFFICIENT EVIDENCE, analyst must review (only one CDM before the decision point)."*
- "Safe by the rules, but with one CDM we won't forecast. On past data the tier is wrong
  twice as often for events like this, so it stays with a human." Dismissing it needs a note.

## 2:25 to 2:55: honest evaluation (Evaluation tab)

- Critical recall 0.82 → 0.95, precision 0.80 → 0.49: "We trade a longer Critical list for
  catching 13 points more of the real ones."
- The visible failure: "Our risk regressor lost to 'final = latest'. Most events collapse
  to zero risk, the dangerous ones stay flat, so a learned change mispredicts exactly those.
  The pipeline measured it and fell back to the baseline. Our model earns its keep in
  P(Critical), not in the point forecast."
- Close: "Where does it fail? On the cases that list says, and the dashboard tells the
  analyst so."

## Backup (only if asked)

- Next 24 hours tab: replay of encounters entering the 1 km / 5 km zones.
- Pc-dilution check: an idea we tested and that failed on outcomes; shown as a note, not a warning.
- Storm sensitivity: 350 events flip at 99th-percentile space weather; sensitivity, not a forecast.
