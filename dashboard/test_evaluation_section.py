"""Verification loop for the Evaluation section's honesty guards and error analysis: run `pytest dashboard -q`."""
import json

from dashboard.sections import evaluation
from dashboard.test_app import METRIC_RUN, isolated, page_text, render, write_pipeline  # noqa: F401  (autouse fixture)

EA = {"n_events": 11942, "n_critical": 834, "scope": "all train events (OOF where trained)",
      "confusion_matrix": {"rows_actual_cols_issued": ["SAFE", "ELEVATED", "CRITICAL"],
                           "counts": [[8601, 218, 127], [149, 1439, 574], [12, 25, 797]]},
      "recall_critical": 0.9556, "recall_critical_final_cdm_under_2d": 0.9373, "silent_missed_critical": 27,
      "missed_critical": {"n": 37, "by_cause": {
          "miss_driven": {"n": 15, "meaning": "Critical only because the final miss < 1,000 m"},
          "withheld_for_review": {"n": 10, "meaning": "sent to an analyst"},
          "near_threshold": {"n": 0, "meaning": "never shown: zero misses"}}},
      "miss_watch_tradeoff": [{"miss_below_m": 1500.0, "recovered_critical": 7, "extra_alerts": 82,
                               "alerts_per_recovered": 11.7}]}


def test_baseline_and_model_on_different_events_get_no_verdict(tmp_path, monkeypatch):
    base = {**METRIC_RUN, "n_events": 11942}
    model = {**METRIC_RUN, "recall_critical": 0.95, "n_events": 8293}
    text = page_text(render(write_pipeline(tmp_path, metrics={"baseline": base, "model": model}), monkeypatch))
    assert "Not comparable yet: the baseline is scored on 11,942 events and the model on 8,293" in text
    assert "baseline 80.0%" not in text and "Model better?" not in text   # no comparison card, no verdict column


def test_same_events_keep_the_comparison(tmp_path, monkeypatch):
    model = {**METRIC_RUN, "recall_critical": 0.95}
    text = page_text(render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN, "model": model}), monkeypatch))
    assert "Not comparable yet" not in text and "baseline 80.0%" in text and "Model better?" in text


def test_confusion_matrix_and_miss_causes(tmp_path, monkeypatch):
    path = tmp_path / "error_analysis.json"
    path.write_text(json.dumps(EA))
    monkeypatch.setattr(evaluation, "ERROR_ANALYSIS", path)
    at = render(write_pipeline(tmp_path / "data"), monkeypatch)
    grid = next(d.value for d in at.dataframe if "issued CRITICAL" in d.value)
    assert grid.loc["actual CRITICAL", "issued SAFE"] == 12 and grid.loc["actual CRITICAL", "issued CRITICAL"] == 797
    causes = next(d.value for d in at.dataframe if "why" in d.value)
    assert list(causes.why) == ["miss driven", "withheld for review"]   # biggest first, zero-count causes dropped
    text = page_text(at)
    assert "37 missed, 27 of them silently" in text and "(12 per catch)" in text


LOCKBOX = {"n": 2167, "note": "lockbox",
           "baseline": {**METRIC_RUN, "recall_critical": 0.824, "precision_critical": 0.855,
                        "confusion": {"labels": ["SAFE", "ELEVATED", "CRITICAL"],
                                      "matrix": [[1435, 56, 3], [38, 418, 24], [8, 26, 159]]}},
           "model": {**METRIC_RUN, "recall_critical": 0.933, "precision_critical": 0.457, "n_critical": 193,
                     "ci95": {"recall_critical": [0.89, 0.97]},
                     "confusion": {"labels": ["SAFE", "ELEVATED", "CRITICAL"],
                                   "matrix": [[1426, 47, 21], [33, 254, 193], [7, 6, 180]]}}}
MISSED = {"missed_examples": [{"event_id": 1640, "cause": "late_jump", "risk_decision": -23.0, "risk_final": -4.1,
                               "miss_decision_m": 8883.0, "miss_final_m": 117.0, "tier_issued": "SAFE"},
                              {"event_id": 93, "cause": "no_evidence_after_cutoff", "risk_decision": -30.0,
                               "risk_final": -30.0, "miss_decision_m": 15629.0, "miss_final_m": 330.0,
                               "tier_issued": "SAFE"}]}


def test_hidden_test_set_block(tmp_path, monkeypatch):
    path = tmp_path / "lockbox.json"
    path.write_text(json.dumps(MISSED))
    monkeypatch.setattr(evaluation, "LOCKBOX_ANALYSIS", path)
    at = render(write_pipeline(tmp_path / "data", metrics={"baseline": METRIC_RUN, "test_lockbox": LOCKBOX}),
                monkeypatch)
    text = page_text(at)
    assert "<b>2,167 events</b>, <b>193</b> truly Critical" in text and "baseline 82.4% · 95% CI 89.0%–97.0%" in text
    model_cm = [d.value for d in at.dataframe if "predicted CRITICAL" in d.value][1]
    assert model_cm.loc["actual CRITICAL", "predicted SAFE"] == 7
    missed = next(d.value for d in at.dataframe if "tier_issued" in d.value)
    assert list(missed.event_id) == [1640, 93] and missed.miss[0] == "8,883 m → 117 m"
    assert missed.pc[1] == "<10⁻³⁰ → <10⁻³⁰" and missed.cause[1] == "no evidence after cutoff"


def test_official_holdout_block_leads_with_the_organisers_split(tmp_path, monkeypatch):
    conf = lambda m: {"labels": ["SAFE", "ELEVATED", "CRITICAL"], "matrix": m}  # noqa: E731
    hm, ea = tmp_path / "holdout_metrics.json", tmp_path / "ea_holdout.json"
    hm.write_text(json.dumps({"baseline": {**METRIC_RUN, "recall_critical": 0.877, "n_critical": 154},
                              "model": {**METRIC_RUN, "recall_critical": 0.981, "precision_critical": 0.487,
                                        "n_critical": 154, "ci95": {"recall_critical": [0.957, 1.0]}}}))
    ea.write_text(json.dumps({"missed_examples": [
        {"event_id": 1936, "cause": "miss_driven", "risk_decision": -9.0, "risk_final": -6.5,
         "miss_decision_m": 13748.0, "miss_final_m": 657.0, "tier_issued": "SAFE"}]}))
    monkeypatch.setattr(evaluation, "HOLDOUT_METRICS", hm)
    monkeypatch.setattr(evaluation, "HOLDOUT_ANALYSIS", ea)
    metrics = {"baseline": METRIC_RUN, "holdout": {"n": 2389, "note": "20% by event_id",
               "baseline": conf([[1700, 60, 25], [40, 380, 30], [5, 14, 135]]),
               "model": conf([[1714, 48, 23], [29, 285, 136], [2, 1, 151]])}}
    text = page_text(render(write_pipeline(tmp_path / "data", metrics=metrics), monkeypatch))
    assert "Official holdout" in text and "<b>2,389 events</b>, <b>154</b> truly Critical" in text
    assert ">98.1%<" in text and "baseline 87.7% · 95% CI 95.7%–100.0%" in text
    assert "13.7 km → 657 m" in text   # format_miss switches to km at 10 km
    assert text.index("Official holdout") < text.index("Alert fatigue")   # the headline comes first


def test_no_holdout_metrics_file_no_holdout_block(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "HOLDOUT_METRICS", tmp_path / "absent.json")
    metrics = {"baseline": METRIC_RUN, "holdout": {"n": 1, "baseline": {}, "model": {}}}
    assert "Official holdout" not in page_text(render(write_pipeline(tmp_path / "d", metrics=metrics), monkeypatch))


def test_final_tier_probability_bar(tmp_path, monkeypatch):
    import pandas as pd
    from dashboard.test_app import by_label
    root = write_pipeline(tmp_path)
    path = root / "processed" / "events_scored.parquet"
    ev = pd.read_parquet(path)
    ev["p_safe"], ev["p_elevated"] = 0.04, 0.06   # event 0: p_critical 0.9 in the fixture
    ev.to_parquet(path)
    at = render(root, monkeypatch)
    text = page_text(at)
    assert "Final tier at TCA" in text and "Safe</span> 4.0%" in text and "Critical</span> 90.0%" in text
    by_label(at.selectbox, "Event (worst first)").set_value(1).run()   # needs review: no forecast, no bar
    assert "Final tier at TCA" not in page_text(at)


def test_no_probability_columns_no_bar(tmp_path, monkeypatch):
    assert "Final tier at TCA" not in page_text(render(write_pipeline(tmp_path), monkeypatch))


def test_threshold_rows_keep_rule_criticals_and_never_raise_refusals():
    import pandas as pd
    train = pd.DataFrame({
        "tier_actual": ["CRITICAL", "CRITICAL", "SAFE", "ELEVATED", "CRITICAL"],
        "tier_now": ["CRITICAL", "SAFE", "SAFE", "SAFE", "SAFE"],
        "p_critical": [0.01, 0.12, 0.30, 0.16, 0.90],
        "needs_review": [False, False, False, False, True]})
    rows = evaluation.threshold_rows(train).set_index("threshold")
    # at 15%: rule-Critical 0 kept, 2 (0.30) and 3 (0.16) raised; 1 (0.12) not; 4 never (refusal)
    assert rows.loc[0.15, "flagged"] == 3 and rows.loc[0.15, "false_alarms"] == 2 and rows.loc[0.15, "missed"] == 2
    assert rows.loc[0.10, "flagged"] == 4 and abs(rows.loc[0.10, "recall"] - 2 / 3) < 1e-9
    assert rows.loc[0.50, "flagged"] == 1 and rows.loc[0.50, "precision"] == 1.0


def test_tradeoff_table_and_reach_card(tmp_path, monkeypatch):
    model = {**METRIC_RUN, "recall_reach": 0.958}
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN, "model": model}), monkeypatch)
    text = page_text(at)
    assert ">95.8%<" in text and "Real Criticals that reach an analyst" in text
    assert "Alert threshold · fewer false alarms cost recall" in text and "15%</span> · chosen" in text


def test_watch_event_gets_a_watch_card_not_a_maneuver(tmp_path, monkeypatch):
    import pandas as pd
    from dashboard.test_app import by_label
    root = write_pipeline(tmp_path)
    path = root / "processed" / "events_scored.parquet"
    ev = pd.read_parquet(path)
    ev["watch"] = ev.event_id == 2     # event 2: SAFE, enough evidence, P(crit) 0.1 in the fixture
    ev.to_parquet(path)
    at = render(root, monkeypatch)
    by_label(at.selectbox, "Event (worst first)").set_value(2).run()
    text = page_text(at)
    assert "under the 15% needed to call it Critical" in text and "Suggested maneuver" not in text
    by_label(at.selectbox, "Event (worst first)").set_value(1).run()   # not on watch
    assert "under the 15% needed" not in page_text(at)


def test_no_lockbox_no_block(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path, metrics={"baseline": METRIC_RUN}), monkeypatch)
    assert "Hidden test set" not in page_text(at)


def test_no_error_analysis_file_shows_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ERROR_ANALYSIS", tmp_path / "absent.json")
    at = render(write_pipeline(tmp_path / "data"), monkeypatch)
    assert not any("issued CRITICAL" in d.value for d in at.dataframe)
