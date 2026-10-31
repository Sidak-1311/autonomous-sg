"""Verification loop for dashboard/sections/fleet.py and window.py: run `pytest dashboard/test_fleet_window.py -q`.

Edge cases: empty input, misses exactly on 1 km / 5 km, TCAs exactly on hour and window edges, now = 0 and 24,
measured vs predicted outcomes, the -30 floor, no storm-sensitive events.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.sections import window  # noqa: E402
from dashboard.test_app import by_label, render, write_pipeline  # noqa: E402,F401  (fixtures + helpers)
from dashboard.test_app import isolated  # noqa: E402,F401  (autouse fixture: no cache or API leaks)
from src.tiering import CRITICAL, ELEVATED, SAFE  # noqa: E402


def _ev(rows):
    base = dict(time_to_tca=2.0, miss_final_m=np.nan, miss_forecast_m=50_000.0, tier_final=SAFE)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_hourly_buckets_and_thresholds():
    ev = _ev([
        dict(time_to_tca=2.0, miss_forecast_m=999.0, tier_final=CRITICAL),         # hour 0, inside 1 km
        dict(time_to_tca=2.0 + 1.5 / 24, miss_forecast_m=1_000.0, tier_final=ELEVATED),   # hour 1, exactly 1 km
        dict(time_to_tca=2.0 + 5 / 24, miss_final_m=5_000.0),                        # hour 5, exactly 5 km, measured
        dict(time_to_tca=2.0 + 5 / 24, miss_final_m=5_001.0, miss_forecast_m=10.0),  # measured beats predicted
        dict(time_to_tca=3.0, miss_forecast_m=10.0, tier_final=CRITICAL),            # window end is exclusive
    ])
    win, inside, near, crit = window._hourly(ev, 2.0)
    assert len(win) == 4 and int(win.real.sum()) == 2
    assert inside.tolist()[:2] == [1, 0] and inside.sum() == 1
    assert near[1] == 1 and near[5] == 1 and near.sum() == 2
    assert crit.iloc[0] == 1 and crit.iloc[-1] == 1 and len(crit) == 24


def test_hourly_empty_and_out_of_window():
    for ev in (_ev([dict()]).iloc[:0], _ev([dict(time_to_tca=9.0)])):
        win, inside, near, crit = window._hourly(ev, 2.0)
        assert win.empty and inside.sum() == near.sum() == crit.sum() == 0


def test_chart_fades_the_future():
    zeros = pd.Series(0, index=window.HOURS)
    for now, faded in ((0, 24), (6, 18), (24, 0)):
        fig = window._chart(zeros, zeros, zeros, now)
        assert list(fig.data[0].marker.opacity).count(window.FUTURE_OPACITY) == faded
        assert fig.layout.shapes[-1].x0 == now - 0.5          # the NOW line


def test_fleet_section_renders(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    for title in ("Fleet overview", "Where every event sits", "Tier mix by object type", "Pc above the floor",
                  "Space-weather sensitivity", "Next 24 hours", "hour by hour"):
        assert title in md
    assert "Events in view" in md and "bm-lede" in md
    assert "<b>40%</b> have a Pc so small" in md                # 2 of 5 events at the floor
    assert any(c.value.startswith("2 events (40%) sit at the floor") for c in at.caption)


def test_fleet_position_toggle(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    toggle = by_label(at.get("button_group"), "Position events by")
    toggle.set_value("Latest CDM").run()
    assert not at.exception, at.exception


def test_window_sliders_and_empty_filter(tmp_path, monkeypatch):
    at = render(write_pipeline(tmp_path), monkeypatch)
    by_label(at.slider, "Now (hours into the window)").set_value(0).run()
    assert not at.exception, at.exception
    assert "Critical events so far" in " ".join(m.value for m in at.markdown)
    at.sidebar.multiselect[0].set_value([]).run()              # filter that matches nothing
    assert not at.exception, at.exception
