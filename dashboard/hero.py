"""Hero scene: every test encounter in the primary's RTN frame, primary at the origin.

Axes follow the Event detail view: x = along-track (t), y = cross-track (n),
z = radial (r). Encounter distances span ~40 m to ~65 km, so by default each
point keeps its true direction but its distance d is drawn at
    rho = log10(1 + d / LOG_SCALE_M)
which keeps the 1 km and 5 km spheres visible next to 60 km encounters. The
hover text always shows the true values. scale="linear" draws kilometres.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.tiering import (CRITICAL, ELEVATED, MISS_CRITICAL_M, MISS_ELEVATED_M, SAFE, TIER_COLORS,
                         format_miss, format_pc, format_prob, tier_array)

LOG_SCALE_M = 250.0
POS = ["relative_position_t", "relative_position_n", "relative_position_r"]
# Quieter tiers recede so the Critical encounters read first
STYLE = {SAFE: dict(size=2.0, opacity=0.45), ELEVATED: dict(size=2.6, opacity=0.75),
         CRITICAL: dict(size=3.2, opacity=0.9)}
UNKNOWN_COLOR = "#7a7a7a"
N_PATHS = 40          # Critical approach paths drawn when CDMs are given
CAMERA_RADIUS = 1.2
CAMERA_HEIGHT = 0.42


def radius(d_m, scale: str = "log"):
    """Display radius for a true distance in metres."""
    d = np.asarray(d_m, float)
    return np.log10(1 + d / LOG_SCALE_M) if scale == "log" else d / 1000.0


def project(xyz_m: np.ndarray, scale: str = "log") -> np.ndarray:
    """Rescale (n, 3) positions in metres to display units, keeping each direction."""
    xyz_m = np.asarray(xyz_m, float).reshape(-1, 3)
    d = np.linalg.norm(xyz_m, axis=1, keepdims=True)
    unit = np.divide(xyz_m, d, out=np.zeros_like(xyz_m), where=d > 0)
    return unit * radius(d, scale)


def camera_eye(angle_deg: float) -> dict:
    """Camera position for a slow orbit: app.py can advance angle_deg between reruns."""
    a = np.radians(angle_deg)
    return dict(x=CAMERA_RADIUS * np.cos(a), y=CAMERA_RADIUS * np.sin(a), z=CAMERA_HEIGHT)


def _tiers(ev: pd.DataFrame) -> np.ndarray:
    for col in ("tier_final", "tier_now"):
        if col in ev:
            return ev[col].to_numpy(object)
    return tier_array(ev["risk_latest"], ev["miss_latest_m"])


def _wire_sphere(r: float, color: str, label: str, n: int = 48) -> list[go.Scatter3d]:
    """Faint wireframe: 3 latitude rings and 4 meridians, one trace."""
    u = np.linspace(0, 2 * np.pi, n)
    xs, ys, zs = [], [], []
    for lat in np.radians([-45, 0, 45]):
        xs += list(r * np.cos(lat) * np.cos(u)) + [None]
        ys += list(r * np.cos(lat) * np.sin(u)) + [None]
        zs += list(np.full(n, r * np.sin(lat))) + [None]
    v = np.linspace(-np.pi / 2, np.pi / 2, n // 2)
    for lon in np.radians([0, 45, 90, 135]):
        for sign in (1, -1):
            xs += list(sign * r * np.cos(v) * np.cos(lon)) + [None]
            ys += list(sign * r * np.cos(v) * np.sin(lon)) + [None]
            zs += list(r * np.sin(v)) + [None]
    return [
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=1),
                     opacity=0.25, hoverinfo="skip", showlegend=False, name=label),
        go.Scatter3d(x=[r * 0.72], y=[-r * 0.72], z=[r * 0.62], mode="text", text=[label],
                     textfont=dict(color=color, size=11), hoverinfo="skip", showlegend=False),
    ]


def _hover(ev: pd.DataFrame, tiers: np.ndarray) -> list[str]:
    def col(name):
        return ev[name].to_numpy() if name in ev else np.full(len(ev), np.nan)
    miss, pc, pcrit = col("miss_latest_m"), col("risk_latest"), col("p_critical")
    obj = ev["c_object_type"].to_numpy() if "c_object_type" in ev else np.full(len(ev), "")
    ids = ev["event_id"].to_numpy() if "event_id" in ev else np.arange(len(ev))
    return [f"Event {i}<br>{t}<br>Miss {format_miss(m)}<br>Pc {format_pc(p)}<br>"
            f"P(Critical at TCA) {format_prob(q)}<br>{o}"
            for i, t, m, p, q, o in zip(ids, tiers, miss, pc, pcrit, obj)]


def _paths(events: pd.DataFrame, cdms: pd.DataFrame, tiers: np.ndarray, scale: str) -> list[go.Scatter3d]:
    """Approach paths (CDM by CDM, towards TCA) of the most likely Critical events."""
    crit = events[tiers == CRITICAL]
    if "p_critical" in crit:
        crit = crit.sort_values("p_critical", ascending=False, na_position="last")
    ids = crit["event_id"].head(N_PATHS)
    if "split" in cdms and "split" in events:
        cdms = cdms[cdms["split"].isin(events["split"].unique())]
    xs, ys, zs = [], [], []
    for _, g in cdms[cdms["event_id"].isin(ids)].groupby("event_id"):
        g = g.sort_values("time_to_tca", ascending=False).dropna(subset=POS)
        if len(g) < 2:
            continue
        p = project(g[POS].to_numpy(), scale)
        xs += list(p[:, 0]) + [None]
        ys += list(p[:, 1]) + [None]
        zs += list(p[:, 2]) + [None]
    if not xs:
        return []
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip", showlegend=False,
                         line=dict(color=TIER_COLORS[CRITICAL], width=1.5), opacity=0.45,
                         name="Critical approach paths")]


def hero_figure(events_df: pd.DataFrame, cdms_df: pd.DataFrame | None = None, split: str = "test",
                scale: str = "log", angle: float = 30.0, orbit_frames: int = 0) -> go.Figure:
    """3D scene of the encounters at the decision point, markers coloured by tier.

    events_df: rows of events_scored.parquet (filtered to `split` when that
    column is present and the split has rows). cdms_df: optional cdms.parquet
    rows to draw Critical approach paths. orbit_frames > 0 adds a camera-orbit
    animation (a play button); otherwise set `angle` per rerun for a slow orbit.
    """
    ev = events_df
    if "split" in ev and (ev["split"] == split).any():
        ev = ev[ev["split"] == split]
    ev = ev.dropna(subset=[c for c in POS if c in ev]) if set(POS) <= set(ev.columns) else ev.iloc[:0]
    tiers = _tiers(ev) if len(ev) else np.array([], object)

    r_crit, r_elev = radius(MISS_CRITICAL_M, scale), radius(MISS_ELEVATED_M, scale)
    traces = _wire_sphere(float(r_crit), TIER_COLORS[CRITICAL], "1 km")
    traces += _wire_sphere(float(r_elev), TIER_COLORS[ELEVATED], "5 km")
    if cdms_df is not None and len(ev) and set(POS) <= set(cdms_df.columns):
        traces += _paths(ev, cdms_df, tiers, scale)

    xyz = project(ev[POS].to_numpy(), scale) if len(ev) else np.zeros((0, 3))
    hover = _hover(ev, tiers)
    for t in [SAFE, ELEVATED, CRITICAL, None]:   # Critical drawn last, on top
        m = (tiers == t) if t else ~np.isin(tiers, [SAFE, ELEVATED, CRITICAL])
        if not m.any():
            continue
        style = STYLE.get(t, dict(size=2.5, opacity=0.35))
        traces.append(go.Scatter3d(
            x=xyz[m, 0], y=xyz[m, 1], z=xyz[m, 2], mode="markers", name=t or "unknown tier",
            marker=dict(size=style["size"], color=TIER_COLORS.get(t, UNKNOWN_COLOR), line=dict(width=0)),
            opacity=style["opacity"], text=[h for h, k in zip(hover, m) if k],
            hovertemplate="%{text}<extra></extra>", showlegend=False))
    traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="PRIMARY",
                               marker=dict(size=6, color="#ffffff", line=dict(color="#101010", width=2)),
                               hovertemplate="Primary satellite<extra></extra>", showlegend=False))

    lim = float(max(np.abs(xyz).max(initial=0.0), r_elev) * 1.05)
    hidden = dict(visible=False, range=[-lim, lim], showspikes=False)
    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="cube",
                   bgcolor="rgba(0,0,0,0)", camera=dict(eye=camera_eye(angle), up=dict(x=0, y=0, z=1))),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hoverlabel=dict(font_size=12),
        meta={"scale": scale, "log_scale_m": LOG_SCALE_M, "n_events": int(len(ev))},
    )
    if orbit_frames > 0:
        steps = np.linspace(angle, angle + 360, orbit_frames, endpoint=False)
        fig.frames = [go.Frame(layout=dict(scene_camera=dict(eye=camera_eye(a))), name=str(i))
                      for i, a in enumerate(steps)]
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
            buttons=[dict(label="Orbit", method="animate",
                          args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True,
                                           transition=dict(duration=0), mode="immediate")])])])
    return fig


ORBIT_PERIOD_S = 120.0   # one full turn; slow enough to read the points
ORBIT_FPS = 20
RESUME_AFTER_MS = 4000   # pause while the viewer drags or zooms, then carry on

_ORBIT_JS = """
<script>
(function () {
  const gd = document.getElementById("%(div_id)s");
  if (!gd || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const eye = ((gd.layout || {}).scene || {}).camera ? gd.layout.scene.camera.eye : {x: 1, y: 1, z: 0.6};
  const R = Math.hypot(eye.x, eye.y), Z = eye.z;
  let a = Math.atan2(eye.y, eye.x), paused = false, timer = null;
  const step = 2 * Math.PI / (%(period)s * %(fps)s);
  const hold = () => { paused = true; clearTimeout(timer); timer = setTimeout(() => {
    const e = gd.layout.scene.camera.eye; a = Math.atan2(e.y, e.x); paused = false; }, %(resume)s); };
  ["mousedown", "wheel", "touchstart"].forEach(ev => gd.addEventListener(ev, hold, {passive: true}));
  setInterval(() => {
    if (paused || document.hidden) return;
    a += step;
    Plotly.relayout(gd, {"scene.camera.eye": {x: R * Math.cos(a), y: R * Math.sin(a), z: Z}});
  }, 1000 / %(fps)s);
})();
</script>
"""


def hero_html(fig: go.Figure, height: int = 640, period_s: float = ORBIT_PERIOD_S, fps: int = ORBIT_FPS,
              include_plotlyjs: str | bool = "cdn", div_id: str = "triage-hero") -> str:
    """Standalone HTML for the scene with a slow, continuous camera orbit.

    st.plotly_chart can't animate on its own, so embed this with
    streamlit.components.v1.html(hero_html(fig), height=...). The orbit pauses
    while the viewer interacts, stops when the tab is hidden, and is off for
    viewers who ask their OS for reduced motion.
    """
    fig = go.Figure(fig)
    fig.frames, fig.layout.updatemenus = [], []   # the script drives the camera instead
    html = fig.to_html(full_html=True, include_plotlyjs=include_plotlyjs, div_id=div_id,
                       default_width="100%", default_height=f"{int(height)}px",
                       config={"displayModeBar": False, "responsive": True})
    js = _ORBIT_JS % dict(div_id=div_id, period=float(period_s), fps=int(fps), resume=RESUME_AFTER_MS)
    style = "<style>html,body{margin:0;background:transparent;overflow:hidden}</style>"
    return html.replace("<head>", "<head>" + style, 1).replace("</body>", js + "</body>", 1)
