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


def test_no_error_analysis_file_shows_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ERROR_ANALYSIS", tmp_path / "absent.json")
    at = render(write_pipeline(tmp_path / "data"), monkeypatch)
    assert not any("issued CRITICAL" in d.value for d in at.dataframe)
