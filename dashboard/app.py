"""Collision-risk dashboard: operations queue, event detail, fleet overview and a 24-hour danger-zone replay.

Run from the repo root: streamlit run dashboard/app.py
Reads only the pipeline's output (DATA CONTRACT in the room): data/processed/events_scored.parquet and
data/processed/cdms.parquet, written by `python -m src.score`. Until they exist it builds a stub with the
same columns from data/test_data.csv (latest CDM as the forecast, no model). DATA_DIR overrides data/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.tiering import (  # noqa: E402
    CRITICAL, ELEVATED, MISS_CRITICAL_M, MISS_ELEVATED_M, RISK_CRITICAL, RISK_ELEVATED, RISK_FLOOR, SAFE,
    TIER_COLORS, TIER_RANK, TIERS, rule_fired, tier_array,
)

DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
PROCESSED = DATA / "processed"
DECISION_DAYS = 2.0  # the latest CDM the model may see is at time_to_tca >= 2 days
RAW_COLS = ["event_id", "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
            "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type"]


def stub_from_raw(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contract-shaped tables from the raw test CSV, so the dashboard works before the pipeline has run."""
    cdms = pd.read_csv(path, usecols=RAW_COLS).assign(split="test")
    cdms["tier"] = tier_array(cdms.risk, cdms.miss_distance)
    seen = cdms[cdms.time_to_tca >= DECISION_DAYS]
    latest = seen.loc[seen.groupby("event_id").time_to_tca.idxmin()]
    ev = latest.rename(columns={"risk": "risk_latest", "miss_distance": "miss_latest_m"}).drop(columns="tier")
    ev["n_cdms"] = ev.event_id.map(cdms.groupby("event_id").size())
    ev["risk_forecast"] = ev["risk_forecast_q90"] = ev.risk_latest
    ev["miss_forecast_m"] = ev.miss_latest_m
    ev["tier_now"] = ev["tier_forecast"] = ev["tier_final"] = tier_array(ev.risk_latest, ev.miss_latest_m)
    ev["reason"] = [rule_fired(r, m) for r, m in zip(ev.risk_latest, ev.miss_latest_m)]
    ev = ev.assign(p_critical=np.nan, drivers="", forecast_source="stub", risk_final=np.nan, miss_final_m=np.nan,
                   tier_actual=None)
    return ev.reset_index(drop=True), cdms


@st.cache_data
def load(data_dir: str, stamp: tuple) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """data_dir and stamp (file mtimes) are the cache key: rewritten files are picked up on the next rerun."""
    if (PROCESSED / "events_scored.parquet").exists() and (PROCESSED / "cdms.parquet").exists():
        events = pd.read_parquet(PROCESSED / "events_scored.parquet")
        cdms = pd.read_parquet(PROCESSED / "cdms.parquet")
        sources = ", ".join(sorted(events.forecast_source.dropna().unique()))
        return events, cdms, f"pipeline output ({sources})"
    events, cdms = stub_from_raw(DATA / "test_data.csv")
    return events, cdms, "stub from test_data.csv (no pipeline output yet)"


def stamp() -> tuple:
    files = [PROCESSED / "events_scored.parquet", PROCESSED / "cdms.parquet", DATA / "test_data.csv"]
    return tuple(f.stat().st_mtime if f.exists() else None for f in files)


def queue_order(df: pd.DataFrame) -> pd.DataFrame:
    """Worst first: tier, then probability of Critical, then forecast risk."""
    return (df.assign(_rank=df.tier_final.map(TIER_RANK))
            .sort_values(["_rank", "p_critical", "risk_forecast"], ascending=False).drop(columns="_rank"))


def tier_bands(fig, axis: str, critical: tuple, elevated: tuple):
    add = fig.add_hrect if axis == "y" else fig.add_vrect
    for (lo, hi), t in ((critical, CRITICAL), (elevated, ELEVATED)):
        add(**{f"{axis}0": lo, f"{axis}1": hi}, fillcolor=TIER_COLORS[t], opacity=0.12, line_width=0, layer="below")


def sphere(radius: float, tier: str, name: str) -> go.Surface:
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    return go.Surface(x=radius * np.cos(u) * np.sin(v), y=radius * np.sin(u) * np.sin(v), z=radius * np.cos(v),
                      opacity=0.15, showscale=False, colorscale=[[0, TIER_COLORS[tier]], [1, TIER_COLORS[tier]]],
                      name=name, hoverinfo="name")


st.set_page_config(page_title="Collision risk", layout="wide")
st.title("Satellite collision-risk evaluator")
events, cdms, source = load(str(DATA), stamp())

with st.sidebar:
    st.caption(f"Data: {source}")
    splits = sorted(events.split.unique())
    split = st.selectbox("Events", splits, index=splits.index("test") if "test" in splits else 0)
    tiers = st.multiselect("Tier (final)", TIERS, default=TIERS)
    types = sorted(events.c_object_type.dropna().unique())
    obj = st.multiselect("Object type", types, default=types)
shown = events[(events.split == split) & events.tier_final.isin(tiers) & events.c_object_type.isin(obj)]
history = cdms[cdms.split == split]

counts = shown.tier_final.value_counts()
k = st.columns(4)
k[0].metric("Events", f"{len(shown):,}")
for col, t in zip(k[1:], reversed(TIERS)):
    col.metric(t.title(), f"{counts.get(t, 0):,}")
st.caption(f"{counts.get(CRITICAL, 0):,} events need a maneuver (CRITICAL: Pc ≥ 1e-4 or miss < 1 km at TCA).")

queue_tab, detail_tab, fleet_tab, window_tab = st.tabs(
    ["Operations queue", "Event detail", "Fleet overview", "Next 24 hours"])

with queue_tab:
    queue = queue_order(shown[shown.tier_final != SAFE])
    st.dataframe(queue[["event_id", "tier_final", "tier_now", "p_critical", "risk_latest", "risk_forecast",
                        "miss_latest_m", "miss_forecast_m", "time_to_tca", "c_object_type", "reason", "drivers"]],
                 hide_index=True, width="stretch", column_config={
                     "p_critical": st.column_config.ProgressColumn("P(critical)", min_value=0, max_value=1),
                     "risk_latest": st.column_config.NumberColumn("risk now", format="%.2f"),
                     "risk_forecast": st.column_config.NumberColumn("risk at TCA", format="%.2f"),
                     "miss_latest_m": st.column_config.NumberColumn("miss now (m)", format="%d"),
                     "miss_forecast_m": st.column_config.NumberColumn("miss at TCA (m)", format="%d"),
                     "time_to_tca": st.column_config.NumberColumn("days to TCA", format="%.2f"),
                 })

    # Alert-fatigue curve: where outcomes are known (train), how many truly Critical events an analyst
    # catches by working this queue top-down, against how many events they have to review.
    known = shown[shown.tier_actual.notna()]
    hits = (queue_order(known).tier_actual == CRITICAL).cumsum().to_numpy()
    if len(hits) and hits[-1]:
        caught = hits / hits[-1]
        k95 = int(np.argmax(caught >= 0.95)) + 1
        st.subheader("Alert fatigue: how far down the queue to go")
        curve = px.line(x=np.arange(1, len(hits) + 1), y=caught,
                        labels={"x": "events reviewed, worst first", "y": "share of actual Critical events caught"})
        curve.add_vline(x=k95, line_dash="dot", annotation_text=f"95% caught after {k95:,}")
        st.plotly_chart(curve, width="stretch")
        st.caption(f"Reviewing the top {k95:,} of {len(hits):,} events ({k95 / len(hits):.0%}) catches 95% of the "
                   f"{int(hits[-1]):,} events that really ended Critical.")

with detail_tab:
    ranked = queue_order(shown)
    if ranked.empty:
        st.info("No events match the filters.")
    else:
        by_id = ranked.set_index("event_id")
        ev = st.selectbox("Event (worst first)", ranked.event_id, format_func=lambda e: f"{e} · {by_id.tier_final[e]}")
        e = by_id.loc[ev]
        path = history[history.event_id == ev].sort_values("time_to_tca", ascending=False)
        text = lambda v: v if isinstance(v, str) else ""  # parquet gives None/NaN for missing strings
        st.markdown(f"**{e.tier_final}** · {e.c_object_type} · {text(e.reason)}"
                    + (f"  \nDrivers: {e.drivers}" if text(e.drivers) else ""))

        a, b = st.columns(2)
        risk = px.line(path, x="time_to_tca", y="risk", markers=True, title="Risk per CDM update")
        tier_bands(risk, "y", (RISK_CRITICAL, 0), (RISK_ELEVATED, RISK_CRITICAL))
        risk.add_trace(go.Scatter(x=[0], y=[e.risk_forecast], mode="markers", marker_symbol="diamond", name="forecast",
                                  error_y=dict(type="data", array=[e.risk_forecast_q90 - e.risk_forecast], symmetric=False)))
        miss = px.line(path, x="time_to_tca", y="miss_distance", markers=True, log_y=True,
                       title="Miss distance per CDM update")
        tier_bands(miss, "y", (1, MISS_CRITICAL_M), (MISS_CRITICAL_M, MISS_ELEVATED_M))
        miss.add_trace(go.Scatter(x=[0], y=[e.miss_forecast_m], mode="markers", marker_symbol="diamond", name="forecast"))
        for fig, final in ((risk, e.risk_final), (miss, e.miss_final_m)):
            if pd.notna(final):
                fig.add_trace(go.Scatter(x=[0], y=[final], mode="markers", marker_symbol="x", name="actual at TCA"))
            fig.add_vline(x=DECISION_DAYS, line_dash="dot", annotation_text="decision point")
            fig.update_xaxes(autorange="reversed", title="days to TCA")
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
        a.plotly_chart(risk, width="stretch")
        b.plotly_chart(miss, width="stretch")

        st.caption("Each point is one CDM update: where the secondary object is predicted to be, relative to the "
                   "satellite, at closest approach (radial r, along-track t, cross-track n). Spheres: 1 km and 5 km.")
        geo = go.Figure([
            sphere(MISS_CRITICAL_M, CRITICAL, "1 km"), sphere(MISS_ELEVATED_M, ELEVATED, "5 km"),
            go.Scatter3d(x=path.relative_position_t, y=path.relative_position_n, z=path.relative_position_r,
                         mode="lines+markers", name="predicted miss vector",
                         marker=dict(size=4, color=path.time_to_tca, colorscale="Viridis",
                                     colorbar=dict(title="days to TCA"))),
            go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5, color="black"), name="satellite"),
        ])
        geo.update_layout(scene=dict(xaxis_title="t (m)", yaxis_title="n (m)", zaxis_title="r (m)", aspectmode="data"),
                          height=560, legend=dict(orientation="h", y=0))
        st.plotly_chart(geo, width="stretch")

with fleet_tab:
    at_tca = st.radio("Position events by", ["forecast at TCA", "latest CDM"], horizontal=True) == "forecast at TCA"
    x, y = ("miss_forecast_m", "risk_forecast") if at_tca else ("miss_latest_m", "risk_latest")
    scatter = px.scatter(shown, x=x, y=y, color="tier_final", color_discrete_map=TIER_COLORS,
                         category_orders={"tier_final": TIERS}, log_x=True, opacity=0.7,
                         hover_data=["event_id", "c_object_type", "p_critical", "reason"],
                         labels={x: "Miss distance (m, log)", y: "Risk, log10(Pc)", "tier_final": "tier"})
    tier_bands(scatter, "y", (RISK_CRITICAL, 0), (RISK_ELEVATED, RISK_CRITICAL))
    tier_bands(scatter, "x", (1, MISS_CRITICAL_M), (MISS_CRITICAL_M, MISS_ELEVATED_M))
    st.plotly_chart(scatter, width="stretch")
    st.caption("Shaded: the tier regions (Critical: risk ≥ -4 or miss < 1 km; Elevated: risk ≥ -6 or miss ≤ 5 km). "
               "Colour is the final tier, which can also come from the model's P(critical).")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(shown, x="c_object_type", color="tier_final", color_discrete_map=TIER_COLORS,
                                   category_orders={"tier_final": TIERS}, title="Tier by object type",
                                   labels={"c_object_type": "object type", "tier_final": "tier"}), width="stretch")
    hist = px.histogram(shown, x=y, color="tier_final", color_discrete_map=TIER_COLORS, nbins=60,
                        category_orders={"tier_final": TIERS}, title="Risk scores",
                        labels={y: "Risk, log10(Pc)", "tier_final": "tier"})
    hist.add_annotation(x=RISK_FLOOR, y=0, yanchor="bottom", text="floor: Pc ≈ 0", showarrow=False)
    right.plotly_chart(hist, width="stretch")

with window_tab:
    # ponytail: events have no shared clock, so each event's decision-point CDM counts as "now" and its TCA
    # falls time_to_tca days later. Swap in real epochs if a dataset carries them.
    real = shown.miss_final_m.notna()
    tca = shown.assign(miss_tca=shown.miss_final_m.where(real, shown.miss_forecast_m), real=real)
    lo = float(tca.time_to_tca.min()) if len(tca) else DECISION_DAYS
    start = st.slider("Window starts (days after the decision point)", lo, lo + 5, lo, 0.25)
    now = st.slider("Now (hours into the window)", 0, 24, 24)
    win = tca[(tca.time_to_tca >= start) & (tca.time_to_tca < start + 1)].copy()
    win["hour"] = ((win.time_to_tca - start) * 24).astype(int)
    hours = np.arange(24)
    inside = win[win.miss_tca < MISS_CRITICAL_M].groupby("hour").size().reindex(hours, fill_value=0)
    near = win[win.miss_tca.between(MISS_CRITICAL_M, MISS_ELEVATED_M)].groupby("hour").size().reindex(hours, fill_value=0)
    crit = (win.tier_final == CRITICAL).groupby(win.hour).sum().reindex(hours, fill_value=0).cumsum()
    fade = np.where(hours < now, 1.0, 0.35)

    c = st.columns(3)
    c[0].metric("Inside 1 km so far", int(inside[:now].sum()), f"{int(inside[now:].sum())} still to come", delta_color="off")
    c[1].metric("1–5 km so far", int(near[:now].sum()), f"{int(near[now:].sum())} still to come", delta_color="off")
    c[2].metric("Critical events so far", int(crit.iloc[now - 1]) if now else 0)
    bars = go.Figure([
        go.Bar(x=hours, y=inside, name="inside 1 km", marker=dict(color=TIER_COLORS[CRITICAL], opacity=fade)),
        go.Bar(x=hours, y=near, name="1–5 km", marker=dict(color=TIER_COLORS[ELEVATED], opacity=fade)),
        go.Scatter(x=hours, y=crit, name="cumulative Critical", yaxis="y2", line=dict(color=TIER_COLORS[CRITICAL])),
    ])
    bars.add_vline(x=now - 0.5, line_dash="dot", annotation_text="now")
    bars.update_layout(barmode="stack", xaxis=dict(title=f"Hour after day {start:.2f}", dtick=1, range=[-0.5, 23.5]),
                       yaxis=dict(title="Objects reaching closest approach"),
                       yaxis2=dict(title="Cumulative Critical", overlaying="y", side="right", rangemode="tozero"),
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(bars, width="stretch")
    st.caption(f"Miss distance at TCA: {int(win.real.sum())} events measured, {int((~win.real).sum())} predicted "
               "by the forecaster (outcome unknown). Faded bars are still to come.")
