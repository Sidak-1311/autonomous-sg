"""Verification loop for dashboard/hero.py: run `pytest dashboard/test_hero.py -q` after every change.

Edge cases: empty input, NaN positions, a zero-distance encounter, distances exactly on the 1 km / 5 km
thresholds, unknown tiers, missing tier / model columns, single-CDM events, other splits only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import hero  # noqa: E402
from src.tiering import CRITICAL, ELEVATED, SAFE, TIER_COLORS  # noqa: E402


def _events(rows):
    base = dict(split="test", event_id=0, c_object_type="DEBRIS", risk_latest=-8.0, miss_latest_m=1.0,
                p_critical=0.01, tier_final=SAFE, relative_position_r=0.0, relative_position_t=0.0,
                relative_position_n=0.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def _markers(fig, tier):
    return [t for t in fig.data if t.mode == "markers" and t.name == tier]


def test_project_keeps_direction_and_maps_thresholds():
    xyz = np.array([[1000.0, 0, 0], [0, 0, -5000.0], [0, 0, 0], [3.0, 4.0, 0]])
    p = hero.project(xyz)
    assert np.allclose(np.linalg.norm(p, axis=1), hero.radius([1000, 5000, 0, 5]))
    assert p[0, 0] > 0 and p[1, 2] < 0 and np.allclose(p[2], 0)          # zero distance stays at the origin
    assert np.allclose(p[3, :2] / np.linalg.norm(p[3, :2]), [0.6, 0.8])  # direction unchanged
    assert np.allclose(hero.project(xyz, "linear")[1], [0, 0, -5.0])     # km
    assert np.all(np.diff(hero.radius([0, 40, 1000, 5000, 65000])) > 0)  # monotone


def test_real_shape_scene():
    ev = _events([
        dict(event_id=1, tier_final=CRITICAL, relative_position_t=999.0, miss_latest_m=999.0, p_critical=0.9),
        dict(event_id=2, tier_final=ELEVATED, relative_position_n=5000.0, miss_latest_m=5000.0),
        dict(event_id=3, tier_final=SAFE, relative_position_r=-200.0, relative_position_t=30_000.0,
             miss_latest_m=30_001.0, risk_latest=-30.0),
        dict(event_id=4, split="train", tier_final=CRITICAL, relative_position_t=10.0),   # other split: dropped
    ])
    fig = hero.hero_figure(ev)
    assert fig.layout.meta["n_events"] == 3
    crit = _markers(fig, CRITICAL)[0]
    assert crit.marker.color == TIER_COLORS[CRITICAL] and len(crit.x) == 1
    assert "Event 1" in crit.text[0] and "999 m" in crit.text[0] and "90.0%" in crit.text[0]
    assert "10⁻³⁰" in _markers(fig, SAFE)[0].text[0]                       # floor shown, not "1e-30"
    labels = [t.text[0] for t in fig.data if t.mode == "text"]
    assert labels == ["1 km", "5 km"]
    assert fig.layout.scene.xaxis.visible is False and fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    # the 5 km sphere fits inside the axis range
    assert fig.layout.scene.xaxis.range[1] >= hero.radius(5000)


def test_empty_and_missing_columns():
    for ev in [_events([]).iloc[:0], pd.DataFrame(), pd.DataFrame({"event_id": [1]})]:
        fig = hero.hero_figure(ev)
        assert fig.layout.meta["n_events"] == 0
        assert not [t for t in fig.data if t.mode == "markers" and t.name != "PRIMARY" and len(t.x) > 1]


def test_nan_positions_and_unknown_tier_and_no_model_columns():
    ev = _events([
        dict(event_id=1, relative_position_t=np.nan),
        dict(event_id=2, tier_final="BOGUS", relative_position_t=700.0),
    ]).drop(columns=["p_critical", "c_object_type"])
    fig = hero.hero_figure(ev)
    assert fig.layout.meta["n_events"] == 1
    unk = _markers(fig, "unknown tier")[0]
    assert unk.marker.color == hero.UNKNOWN_COLOR and "n/a" in unk.text[0]


def test_tier_computed_when_no_tier_columns():
    ev = _events([dict(relative_position_t=500.0, miss_latest_m=500.0)]).drop(columns="tier_final")
    assert len(_markers(hero.hero_figure(ev), CRITICAL)[0].x) == 1      # 500 m < 1 km


def test_falls_back_to_all_rows_when_split_missing():
    ev = _events([dict(split="train", relative_position_t=2000.0)])
    assert hero.hero_figure(ev, split="test").layout.meta["n_events"] == 1


def test_critical_paths_from_cdms():
    ev = _events([dict(event_id=7, tier_final=CRITICAL, relative_position_t=800.0, p_critical=0.99),
                  dict(event_id=8, tier_final=CRITICAL, relative_position_t=900.0)])
    cdms = pd.DataFrame({"split": "test", "event_id": [7, 7, 7, 8],          # event 8 has one CDM: no path
                         "time_to_tca": [5.0, 3.0, 2.1, 4.0],
                         "relative_position_t": [3000.0, 1500.0, 800.0, 900.0],
                         "relative_position_n": [0.0, 0.0, 0.0, 0.0], "relative_position_r": [0.0, 0.0, 0.0, 0.0]})
    paths = [t for t in hero.hero_figure(ev, cdms).data if t.name == "Critical approach paths"]
    assert len(paths) == 1 and sum(v is None for v in paths[0].x) == 1
    assert paths[0].x[0] > paths[0].x[2]                                   # drawn from far to near
    assert not [t for t in hero.hero_figure(ev, cdms.iloc[:0]).data if t.name == "Critical approach paths"]


@pytest.mark.parametrize("frames", [0, 12])
def test_camera_orbit(frames):
    ev = _events([dict(relative_position_t=1000.0)])
    fig = hero.hero_figure(ev, angle=90, orbit_frames=frames)
    eye = fig.layout.scene.camera.eye
    assert abs(eye.x) < 1e-9 and eye.y == pytest.approx(hero.CAMERA_RADIUS)
    assert len(fig.frames) == frames and bool(fig.layout.updatemenus) == (frames > 0)


def test_hero_html_orbit_script():
    ev = _events([dict(relative_position_t=1000.0, tier_final=CRITICAL)])
    fig = hero.hero_figure(ev, orbit_frames=6)
    html = hero.hero_html(fig, height=480, period_s=60, fps=10, div_id="x-hero")
    assert html.count('id="x-hero"') == 1 and "480px" in html
    assert "Plotly.relayout" in html and "prefers-reduced-motion" in html and "document.hidden" in html
    assert "2 * Math.PI / (60.0 * 10)" in html
    assert html.index("<style>") < html.index("</head>") and html.rstrip().endswith("</html>")
    assert len(fig.frames) == 6                          # caller's figure untouched
    assert hero.hero_html(hero.hero_figure(pd.DataFrame())).count("Plotly.relayout") == 1   # empty scene


# ---- WebGL render (hero_render / render_payload) ------------------------------------------------

def _payload(html):
    import json
    start = html.index('type="application/json">') + len('type="application/json">')
    return json.loads(html[start:html.index("</script>", start)])


def test_render_payload_axes_tiers_and_true_values_in_tips():
    ev = _events([
        dict(event_id=1, tier_final=CRITICAL, relative_position_t=999.0, miss_latest_m=999.0, p_critical=0.9),
        dict(event_id=2, tier_final=ELEVATED, relative_position_r=5000.0, miss_latest_m=5000.0),
        dict(event_id=3, tier_final="BOGUS", relative_position_n=-1000.0, risk_latest=-30.0),
        dict(event_id=4, split="train", tier_final=CRITICAL, relative_position_t=10.0),   # other split: dropped
    ])
    p = hero.render_payload(ev)
    pts = np.array(p["pts"]).reshape(-1, 3)
    assert p["tier"] == [2, 1, -1] and len(p["tip"]) == 3
    assert pts[0, 0] == pytest.approx(hero.radius(999), abs=1e-4)          # along-track -> x
    assert pts[1, 1] == pytest.approx(hero.radius(5000), abs=1e-4)         # radial -> y (up)
    assert pts[2, 2] == pytest.approx(-hero.radius(1000), abs=1e-4)        # cross-track -> z
    assert "999 m" in p["tip"][0] and "90.0%" in p["tip"][0] and "<br>" not in p["tip"][0]
    assert "10⁻³⁰" in p["tip"][2]                                         # floor, not "1e-30"
    assert p["shells"] == pytest.approx([hero.radius(1000), hero.radius(5000)])
    assert p["extent"] >= p["shells"][1] and p["colors"][2] == TIER_COLORS[CRITICAL]


def test_render_payload_empty_nan_and_missing_columns():
    for ev in [_events([]).iloc[:0], pd.DataFrame(), pd.DataFrame({"event_id": [1]}),
               _events([dict(relative_position_t=np.nan)])]:
        p = hero.render_payload(ev)
        assert p["pts"] == [] and p["tier"] == [] and p["paths"] == []
        assert p["extent"] == pytest.approx(hero.radius(5000))


def test_render_paths_only_for_critical_with_two_cdms():
    ev = _events([dict(event_id=7, tier_final=CRITICAL, relative_position_t=800.0, p_critical=0.99),
                  dict(event_id=8, tier_final=CRITICAL, relative_position_t=900.0),
                  dict(event_id=9, tier_final=SAFE, relative_position_t=9000.0)])
    cdms = pd.DataFrame({"split": "test", "event_id": [7, 7, 7, 8, 9, 9],
                         "time_to_tca": [5.0, 3.0, 2.1, 4.0, 5.0, 3.0],
                         "relative_position_t": [3000.0, 1500.0, 800.0, 900.0, 9000.0, 8000.0],
                         "relative_position_n": 0.0, "relative_position_r": 0.0})
    paths = hero.render_payload(ev, cdms)["paths"]
    assert len(paths) == 1 and len(paths[0]) == 9                         # 3 CDMs x (x, y, z)
    assert paths[0][0] > paths[0][6]                                       # far to near
    assert hero.render_payload(ev, cdms, n_paths=0)["paths"] == []


def test_hero_render_html_is_self_contained_and_safe():
    ev = _events([dict(event_id=1, tier_final=CRITICAL, relative_position_t=500.0,
                       c_object_type="</script><script>alert(1)</script>")])
    html = hero.hero_render(ev, period_s=90)
    assert hero.THREE_JS in html and "cdnjs.cloudflare.com" in html and "/r128/" in html
    assert html.count("</script>") == 3                     # payload, three.js, scene: data can't close a tag
    assert _payload(html)["tier"] == [2]
    assert "prefers-reduced-motion" in html and "WebGL" in html and "background:transparent" in html
    assert "const RESUME = 4000, PERIOD = 90.0;" in html and "__" not in html.replace("__proto__", "")
    assert _payload(hero.hero_render(pd.DataFrame()))["pts"] == []    # empty scene still renders
