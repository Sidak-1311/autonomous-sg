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


# ---- WebGL render (three.js): the buttermax-style hero ---------------------------------
# Same frame and log-radius mapping as hero_figure, rendered with lighting, soft glowing
# points, translucent threshold shells, a depth-of-field fade and a slow eased camera drift.
THREE_JS = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
RENDER_SIZE = {SAFE: 6.5, ELEVATED: 8.0, CRITICAL: 11.0}
RENDER_ALPHA = {SAFE: 0.75, ELEVATED: 0.85, CRITICAL: 1.0}
TIER_CODE = {SAFE: 0, ELEVATED: 1, CRITICAL: 2}


def render_payload(events_df: pd.DataFrame, cdms_df: pd.DataFrame | None = None,
                   split: str = "test", n_paths: int = N_PATHS) -> dict:
    """Everything the WebGL scene needs, in display units (JSON-serialisable).

    pts: flat [x, y, z, ...] with three.js axes (x = along-track t, y = radial r,
    z = cross-track n, so radial is "up"); tier: 0/1/2 (-1 unknown); tip: hover
    text with true values; shells: display radii of the 1 km / 5 km spheres;
    paths: approach polylines of the most likely Critical events.
    """
    ev = events_df
    if "split" in ev and (ev["split"] == split).any():
        ev = ev[ev["split"] == split]
    ev = ev.dropna(subset=POS) if set(POS) <= set(ev.columns) else ev.iloc[:0]
    tiers = _tiers(ev) if len(ev) else np.array([], object)
    xyz = project(ev[POS].to_numpy(), "log") if len(ev) else np.zeros((0, 3))
    # POS order is (t, n, r); three.js is y-up, so draw (t, r, n)
    pts = xyz[:, [0, 2, 1]]
    tips = [h.replace("<br>", " · ") for h in _hover(ev, tiers)]

    paths = []
    if cdms_df is not None and len(ev) and set(POS) <= set(cdms_df.columns):
        crit = ev[tiers == CRITICAL]
        if "p_critical" in crit:
            crit = crit.sort_values("p_critical", ascending=False, na_position="last")
        cd = cdms_df
        if "split" in cd and "split" in ev:
            cd = cd[cd["split"].isin(ev["split"].unique())]
        for _, g in cd[cd["event_id"].isin(crit["event_id"].head(n_paths))].groupby("event_id"):
            g = g.sort_values("time_to_tca", ascending=False).dropna(subset=POS)
            if len(g) >= 2:
                paths.append(np.round(project(g[POS].to_numpy(), "log")[:, [0, 2, 1]], 4).ravel().tolist())

    r_elev = float(radius(MISS_ELEVATED_M))
    return {
        "pts": np.round(pts, 4).ravel().tolist(),
        "tier": [TIER_CODE.get(t, -1) for t in tiers],
        "tip": tips,
        "shells": [float(radius(MISS_CRITICAL_M)), r_elev],
        "extent": float(max(np.abs(pts).max(initial=0.0), r_elev)),
        "paths": paths,
        "colors": [TIER_COLORS[SAFE], TIER_COLORS[ELEVATED], TIER_COLORS[CRITICAL], UNKNOWN_COLOR],
        "size": [RENDER_SIZE[SAFE], RENDER_SIZE[ELEVATED], RENDER_SIZE[CRITICAL], 4.0],
        "alpha": [RENDER_ALPHA[SAFE], RENDER_ALPHA[ELEVATED], RENDER_ALPHA[CRITICAL], 0.4],
    }


_RENDER_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden}
#c{position:absolute;inset:0;cursor:grab}#c:active{cursor:grabbing}
#tip{position:absolute;pointer-events:none;padding:6px 9px;border:1px solid #262626;background:#000c;
 color:#F2F2F0;font:12px/1.4 "JetBrains Mono",ui-monospace,monospace;white-space:nowrap;opacity:0;transition:opacity .15s}
#fallback{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#8E8E88;
 font:13px system-ui,sans-serif}
</style></head><body>
<div id="c"></div><div id="tip"></div><div id="fallback">3D view needs WebGL</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script src="__THREE__"></script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("payload").textContent);
  const host = document.getElementById("c"), tip = document.getElementById("tip");
  if (!window.THREE) { document.getElementById("fallback").style.display = "flex"; return; }
  let renderer;
  try { renderer = new THREE.WebGLRenderer({antialias: true, alpha: true}); }
  catch (e) { document.getElementById("fallback").style.display = "flex"; return; }
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const ext = Math.max(D.extent, 1e-3);
  const camera = new THREE.PerspectiveCamera(32, 1, ext * 0.01, ext * 20);
  const world = new THREE.Group(); scene.add(world);

  // Lighting: soft key, cool rim, warm fill, so the primary reads as a glossy object
  scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x0b0b0b, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(3, 4, 2); scene.add(key);
  const rim = new THREE.DirectionalLight(0x9fb8ff, 1.1); rim.position.set(-4, 1, -3); scene.add(rim);
  const fill = new THREE.PointLight(0xffd9b0, 0.6, ext * 4); fill.position.set(0, -ext * 0.6, ext * 0.8); scene.add(fill);

  // Primary satellite: bus + solar wings, a few pixels across at the origin
  const s = D.shells[0] * 0.13;
  const sat = new THREE.Group();
  const gloss = new THREE.MeshStandardMaterial({color: 0xd9dbe0, metalness: 0.75, roughness: 0.22});
  const foil = new THREE.MeshStandardMaterial({color: 0xc9a24a, metalness: 0.9, roughness: 0.35});
  const cell = new THREE.MeshStandardMaterial({color: 0x15213d, metalness: 0.3, roughness: 0.55,
                                               emissive: 0x0a1430, emissiveIntensity: 0.5});
  sat.add(new THREE.Mesh(new THREE.BoxGeometry(s, s * 1.1, s), gloss));
  const band = new THREE.Mesh(new THREE.BoxGeometry(s * 1.02, s * 0.35, s * 1.02), foil); sat.add(band);
  [-1, 1].forEach(k => {
    const w = new THREE.Mesh(new THREE.BoxGeometry(s * 2.6, s * 0.04, s * 0.9), cell);
    w.position.x = k * s * 1.95; sat.add(w);
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(s * 0.04, s * 0.04, s * 0.6, 8), gloss);
    arm.rotation.z = Math.PI / 2; arm.position.x = k * s * 0.6; sat.add(arm);
  });
  world.add(sat);

  // Threshold shells: fresnel rim so they read as glass, not wireframe
  const shellMat = (hex, power, alpha) => new THREE.ShaderMaterial({
    uniforms: {c: {value: new THREE.Color(hex)}, p: {value: power}, a: {value: alpha}},
    vertexShader: "varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); " +
      "vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }",
    fragmentShader: "uniform vec3 c; uniform float p; uniform float a; varying vec3 vN; varying vec3 vV; " +
      "void main(){ float f = pow(1.-abs(dot(vN,vV)), p); gl_FragColor = vec4(c, f*a); }",
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide});
  world.add(new THREE.Mesh(new THREE.SphereGeometry(D.shells[0], 64, 48), shellMat(D.colors[2], 3.2, 0.32)));
  world.add(new THREE.Mesh(new THREE.SphereGeometry(D.shells[1], 64, 48), shellMat(D.colors[1], 4.0, 0.26)));
  const ring = (r, hex, a) => {
    const g = new THREE.RingGeometry(r * 0.995, r * 1.005, 160);
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({color: hex, transparent: true, opacity: a,
                                                             side: THREE.DoubleSide, depthWrite: false}));
    m.rotation.x = Math.PI / 2; world.add(m);
  };
  ring(D.shells[0], D.colors[2], 0.35); ring(D.shells[1], D.colors[1], 0.22);

  // Approach paths of the most likely Critical events, fading in toward TCA
  D.paths.forEach(p => {
    const g = new THREE.BufferGeometry(), n = p.length / 3, col = new Float32Array(n * 3);
    const c = new THREE.Color(D.colors[2]);
    for (let i = 0; i < n; i++) { const t = 0.15 + 0.85 * i / Math.max(n - 1, 1);
      col[3*i] = c.r * t; col[3*i+1] = c.g * t; col[3*i+2] = c.b * t; }
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    world.add(new THREE.Line(g, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 0.55,
                                                             blending: THREE.AdditiveBlending, depthWrite: false})));
  });

  // Encounters: soft round sprites; Critical glow and breathe; depth-of-field fade off the focus plane
  const n = D.tier.length, geo = new THREE.BufferGeometry();
  const col = new Float32Array(n * 3), size = new Float32Array(n), alpha = new Float32Array(n), crit = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = D.tier[i] < 0 ? 3 : D.tier[i], c = new THREE.Color(D.colors[t]);
    col[3*i] = c.r; col[3*i+1] = c.g; col[3*i+2] = c.b;
    size[i] = D.size[t]; alpha[i] = D.alpha[t]; crit[i] = t === 2 ? 1 : 0;
  }
  geo.setAttribute("position", new THREE.Float32BufferAttribute(D.pts, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setAttribute("size", new THREE.BufferAttribute(size, 1));
  geo.setAttribute("alpha", new THREE.BufferAttribute(alpha, 1));
  geo.setAttribute("crit", new THREE.BufferAttribute(crit, 1));
  const ptsMat = new THREE.ShaderMaterial({
    uniforms: {time: {value: 0}, focus: {value: 1}, depth: {value: 1}, px: {value: renderer.getPixelRatio()}},
    vertexShader: [
      "attribute float size; attribute float alpha; attribute float crit; attribute vec3 color;",
      "uniform float time; uniform float focus; uniform float depth; uniform float px;",
      "varying vec3 vC; varying float vA; varying float vBlur; varying float vCrit;",
      "void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); float d = -mv.z;",
      " vBlur = clamp(abs(d-focus)/depth, 0., 1.); vC = color; vCrit = crit;",
      " float pulse = crit > .5 ? 1. + .18*sin(time*1.6 + position.x*9.) : 1.;",
      " vA = alpha*(1. - .55*vBlur);",
      " gl_PointSize = size*px*pulse*(1. + 1.4*vBlur)*(3./max(d,1e-3))*" + (ext * 0.33).toFixed(4) + ";",
      " gl_Position = projectionMatrix*mv; }"].join("\n"),
    fragmentShader: [
      "varying vec3 vC; varying float vA; varying float vBlur; varying float vCrit;",
      "void main(){ vec2 q = gl_PointCoord-.5; float r = length(q)*2.; if (r > 1.) discard;",
      " float core = smoothstep(1., mix(.25, .0, vBlur), r);",
      " float halo = vCrit > .5 ? .55*exp(-r*r*3.) : 0.;",
      " gl_FragColor = vec4(vC*(1.+.6*vCrit), (core*.9 + halo)*vA); }"].join("\n"),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending});
  const cloud = new THREE.Points(geo, ptsMat); world.add(cloud);

  // Camera: eased intro dolly, then a slow drift (azimuth turn + gentle bob); drag to rotate
  const R0 = ext * 3.1, R1 = ext * 2.35;
  let az = 0.6, el = 0.32, dragAz = 0, dragEl = 0, dragging = false, lastX = 0, lastY = 0, idleAt = 0;
  const RESUME = __RESUME__, PERIOD = __PERIOD__;
  const ease = t => t < 0 ? 0 : t > 1 ? 1 : 1 - Math.pow(1 - t, 3);
  host.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY;
                                             host.setPointerCapture(e.pointerId); });
  host.addEventListener("pointerup", () => { dragging = false; idleAt = performance.now(); });
  host.addEventListener("pointermove", e => {
    if (dragging) { dragAz -= (e.clientX - lastX) * 0.006; dragEl += (e.clientY - lastY) * 0.004;
                    dragEl = Math.max(-0.9, Math.min(0.9, dragEl)); lastX = e.clientX; lastY = e.clientY;
                    idleAt = performance.now(); }
    hover(e);
  });
  host.addEventListener("pointerleave", () => { tip.style.opacity = 0; });

  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
  ray.params.Points.threshold = ext * 0.012;
  function hover(e) {
    const b = host.getBoundingClientRect();
    mouse.set((e.clientX - b.left) / b.width * 2 - 1, -(e.clientY - b.top) / b.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObject(cloud)[0];
    if (!hit || dragging) { tip.style.opacity = 0; return; }
    tip.textContent = D.tip[hit.index];
    const x = e.clientX - b.left + 14, y = e.clientY - b.top + 14;
    tip.style.left = Math.min(x, b.width - tip.offsetWidth - 8) + "px"; tip.style.top = Math.min(y, b.height - 34) + "px";
    tip.style.opacity = 1;
  }

  function resize() { const w = host.clientWidth, h = host.clientHeight || 1;
    renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
    camera.aspect = w / h; camera.updateProjectionMatrix(); }
  window.addEventListener("resize", resize); resize();

  const t0 = performance.now();
  function frame(now) {
    const t = (now - t0) / 1000, intro = still ? 1 : ease(t / 2.4);
    const idle = !dragging && now - idleAt > RESUME;
    if (idle && !still) { az += (2 * Math.PI / PERIOD) / 60; dragEl *= 0.985; }
    const r = R0 + (R1 - R0) * intro, e = el + dragEl + (still ? 0 : 0.05 * Math.sin(t * 0.21));
    const a = az + dragAz - (1 - intro) * 0.8;
    camera.position.set(r * Math.cos(e) * Math.cos(a), r * Math.sin(e), r * Math.cos(e) * Math.sin(a));
    camera.lookAt(0, 0, 0);
    sat.rotation.y = still ? 0.5 : 0.5 + t * 0.15; sat.rotation.x = 0.35;
    ptsMat.uniforms.time.value = still ? 0 : t;
    ptsMat.uniforms.focus.value = r; ptsMat.uniforms.depth.value = ext * 1.6;
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script></body></html>"""


def hero_render(events_df: pd.DataFrame, cdms_df: pd.DataFrame | None = None, split: str = "test",
                period_s: float = ORBIT_PERIOD_S) -> str:
    """Self-contained HTML (three.js) for streamlit.components.v1.html(..., height=...).

    Transparent background; slow camera drift that pauses while the viewer drags and
    resumes after RESUME_AFTER_MS; still for prefers-reduced-motion; hover shows the
    true miss / Pc / P(Critical); a text fallback when WebGL or the CDN is unavailable.
    """
    import json
    payload = json.dumps(render_payload(events_df, cdms_df, split), separators=(",", ":"),
                         allow_nan=False)
    payload = payload.replace("</", "<\\/")   # never let data close the script tag
    return (_RENDER_HTML.replace("__PAYLOAD__", payload).replace("__THREE__", THREE_JS)
            .replace("__RESUME__", str(int(RESUME_AFTER_MS))).replace("__PERIOD__", str(float(period_s))))
