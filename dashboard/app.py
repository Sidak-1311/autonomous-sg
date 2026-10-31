"""Collision-risk dashboard: operations queue, event detail, fleet overview and a 24-hour danger-zone replay.

Run from the repo root: streamlit run dashboard/app.py
Reads only the pipeline's output (DATA CONTRACT in the room): data/processed/events_scored.parquet and
data/processed/cdms.parquet, written by `python -m src.score`. Until they exist it builds a stub with the
same columns from data/test_data.csv (latest CDM as the forecast, no model). DATA_DIR overrides data/.
"""
import json
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
from dashboard import brief, style  # noqa: E402
from src.physics import LEO_MEAN_MOTION, combined_sigma, dilution  # noqa: E402
from src.score import _add_maneuver, _add_review  # noqa: E402  (the pipeline's own rules, reused by the stub)
from src.tiering import (  # noqa: E402
    CRITICAL, ELEVATED, MISS_CRITICAL_M, MISS_ELEVATED_M, RISK_CRITICAL, RISK_ELEVATED, RISK_FLOOR, SAFE,
    TIER_COLORS, TIER_RANK, TIERS, format_miss, format_pc, format_prob, format_ttca, rule_fired, tier_array,
)

DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
PROCESSED = DATA / "processed"
DECISION_DAYS = 2.0  # the latest CDM the model may see is at time_to_tca >= 2 days
RAW_COLS = ["event_id", "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
            "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type",
            "c_sigma_r", "t_sigma_r"]


def stub_from_raw(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contract-shaped tables from the raw test CSV, so the dashboard works before the pipeline has run."""
    cdms = pd.read_csv(path, usecols=RAW_COLS).assign(split="test")
    cdms["tier"] = tier_array(cdms.risk, cdms.miss_distance)
    seen = cdms[cdms.time_to_tca >= DECISION_DAYS]
    latest = seen.loc[seen.groupby("event_id").time_to_tca.idxmin()]
    ev = latest.rename(columns={"risk": "risk_latest", "miss_distance": "miss_latest_m"}).drop(columns="tier")
    ev["n_cdms"] = ev.event_id.map(seen.groupby("event_id").size())  # CDMs up to the decision point
    ev["risk_forecast"] = ev["risk_forecast_q90"] = ev.risk_latest
    ev["miss_forecast_m"] = ev.miss_latest_m
    ev["tier_now"] = ev["tier_forecast"] = ev["tier_final"] = tier_array(ev.risk_latest, ev.miss_latest_m)
    ev["reason"] = [rule_fired(r, m) for r, m in zip(ev.risk_latest, ev.miss_latest_m)]
    ev = ev.assign(p_critical=np.nan, drivers="", forecast_source="stub", risk_final=np.nan, miss_final_m=np.nan,
                   tier_actual=None)
    ev["pc_diluted"], ev["risk_potential"] = dilution(ev.risk_latest, ev.miss_latest_m,
                                                      combined_sigma(ev.c_sigma_r, ev.t_sigma_r))
    _add_maneuver(ev)
    ev["c_sigma_r_missing"], ev["t_sigma_r_missing"] = ev.c_sigma_r.isna().astype(int), ev.t_sigma_r.isna().astype(int)
    _add_review(ev)
    return ev.reset_index(drop=True), cdms


@st.cache_data
def load(data_dir: str, stamp: tuple) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """data_dir and stamp (file mtimes) are the cache key: rewritten files are picked up on the next rerun."""
    if (PROCESSED / "events_scored.parquet").exists() and (PROCESSED / "cdms.parquet").exists():
        events = pd.read_parquet(PROCESSED / "events_scored.parquet")
        cdms = pd.read_parquet(PROCESSED / "cdms.parquet")
        sources = ", ".join(sorted(events.forecast_source.dropna().unique()))
        return with_defaults(events), cdms, f"pipeline output ({sources})"
    events, cdms = stub_from_raw(DATA / "test_data.csv")
    return with_defaults(events), cdms, "stub from test_data.csv (no pipeline output yet)"


# Columns the pipeline added over time, with what an older output (or the stub) means by their absence.
LATER_COLUMNS = {
    "pc_diluted": lambda ev: False, "risk_potential": lambda ev: ev.risk_latest,
    "maneuver_dr_m": lambda ev: np.nan, "maneuver_dv_mps": lambda ev: np.nan,
    "needs_review": lambda ev: False, "review_reason": lambda ev: "",
    "risk_lo": lambda ev: ev.risk_forecast, "risk_hi": lambda ev: ev.risk_forecast_q90,
    "p_critical_storm": lambda ev: np.nan, "storm_flips": lambda ev: False,
    "forecast_in_sample": lambda ev: ev.forecast_source == "model",  # before the flag, train rows were in-sample
}


def with_defaults(events: pd.DataFrame) -> pd.DataFrame:
    return events.assign(**{col: fill(events) for col, fill in LATER_COLUMNS.items() if col not in events})


# Analyst sign-off: the system recommends, a named person decides, and every decision is logged with what
# the system showed at the time. ponytail: an append-only CSV per laptop; move it to a shared store (the
# room or a database) if several analysts sign the same queue.
DECISIONS = DATA / "decisions.csv"
DECISION_COLS = ["at_utc", "split", "event_id", "analyst", "decision", "note", "tier_final", "p_critical",
                 "forecast_source"]
CHOICES = ["Maneuver", "Monitor", "Dismiss"]


def read_decisions() -> pd.DataFrame:
    return pd.read_csv(DECISIONS) if DECISIONS.exists() else pd.DataFrame(columns=DECISION_COLS)


def decision_problem(analyst: str, decision: str, note: str, e: pd.Series) -> str | None:
    if not analyst.strip():
        return "Enter your name: every decision is signed."
    if decision == "Dismiss" and (e.tier_final == CRITICAL or e.needs_review) and not note.strip():
        return "Dismissing a Critical or not-yet-reviewable event needs a note saying why."
    return None


def record_decision(row: dict):
    pd.DataFrame([row], columns=DECISION_COLS).to_csv(DECISIONS, mode="a", header=not DECISIONS.exists(),
                                                      index=False)


@st.cache_data(show_spinner="Asking Gemini…")
def cached_brief(fields_json: str) -> str:
    """Keyed on the exact fields sent, so reruns and repeat clicks don't call the API again."""
    return brief.explain(json.loads(fields_json), brief.api_key())


def stamp() -> tuple:
    files = [PROCESSED / "events_scored.parquet", PROCESSED / "cdms.parquet", DATA / "test_data.csv"]
    return tuple(f.stat().st_mtime if f.exists() else None for f in files)


def load_metrics() -> dict | None:
    path = PROCESSED / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


# metric -> (label, higher is better)
METRICS = {"recall_critical": ("Critical recall", True), "precision_critical": ("Critical precision", True),
           "f2": ("F2 (high risk)", True), "mse_hr": ("MSE on high-risk events", False), "loss": ("loss = MSE / F2", False)}


def metrics_table(metrics: dict) -> pd.DataFrame:
    """One row per metric: value and 95% CI for each scorer present (baseline, model), and which is better."""
    def ci(run, key):
        lo_hi = run.get("ci95", {}).get(key)
        return f"[{lo_hi[0]:.3g}, {lo_hi[1]:.3g}]" if lo_hi else "—"

    rows = []
    for key, (label, higher) in METRICS.items():
        row = {"metric": label}
        for name in ("baseline", "model"):
            if name in metrics and key in metrics[name]:
                row[name], row[f"{name} 95% CI"] = metrics[name][key], ci(metrics[name], key)
        if "baseline" in row and "model" in row:
            better = row["model"] > row["baseline"] if higher else row["model"] < row["baseline"]
            row["model better?"] = "same" if np.isclose(row["model"], row["baseline"]) else "yes" if better else "no"
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def tier_text(tier) -> str:
    """Table cell style: the tier word in its tier colour (the word itself stays, so colour is never the only cue)."""
    return f"color: {TIER_COLORS[tier]}; font-weight: 600" if tier in TIER_COLORS else ""


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


st.set_page_config(page_title="Collision risk", page_icon="🛰️", layout="wide")
style.apply()
st.markdown('<div class="hero-eyebrow">Conjunction triage · low Earth orbit</div>', unsafe_allow_html=True)
st.title("Collision-risk evaluator")
st.markdown('<p class="hero-sub">Every close approach in one queue, worst first. Rules set the tier, a forecaster '
            'looks ahead to closest approach, and an analyst signs every decision.</p>', unsafe_allow_html=True)
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
review = shown.needs_review.astype(bool)
decisions = read_decisions()
latest_decision = (decisions[decisions.split == split].sort_values("at_utc")
                   .groupby("event_id").decision.last())
shown = shown.assign(decision=shown.event_id.map(latest_decision).fillna(""))
queued = (shown.tier_final != SAFE) | review  # a refusal is never auto-cleared
k = st.columns(6)
k[5].metric("Undecided", f"{int((queued & (shown.decision == '')).sum()):,}", border=True,
            help="Queued events without a signed analyst decision.")
k[0].metric("Events", f"{len(shown):,}", border=True)
for col, t in zip(k[1:], reversed(TIERS)):
    col.metric(t.title(), f"{counts.get(t, 0):,}", border=True)
k[4].metric("Needs review", f"{int(review.sum()):,}", border=True,
            help="Too little evidence to forecast: an analyst must review these whatever their tier.")
st.caption(f"{counts.get(CRITICAL, 0):,} events need a maneuver (CRITICAL: Pc ≥ {format_pc(RISK_CRITICAL)} or miss "
           f"< {format_miss(MISS_CRITICAL_M)} at TCA). {int(review.sum()):,} have too little evidence to forecast; "
           "they stay in the queue whatever their tier.")

queue_tab, detail_tab, fleet_tab, window_tab, eval_tab = st.tabs(
    ["Operations queue", "Event detail", "Fleet overview", "Next 24 hours", "Evaluation"])

with queue_tab:
    queue = queue_order(shown[queued])
    pc_text, miss_text = (lambda s: [format_pc(v) for v in s]), (lambda s: [format_miss(v) for v in s])
    view = queue[["event_id", "tier_final", "decision", "needs_review", "tier_now", "p_critical", "risk_latest",
                  "risk_forecast", "risk_hi", "miss_latest_m", "miss_forecast_m", "time_to_tca", "c_object_type",
                  "storm_flips", "pc_diluted", "reason", "drivers"]].assign(
        risk_latest=pc_text(queue.risk_latest), risk_forecast=pc_text(queue.risk_forecast),
        risk_hi=pc_text(queue.risk_hi), miss_latest_m=miss_text(queue.miss_latest_m),
        miss_forecast_m=miss_text(queue.miss_forecast_m), time_to_tca=[format_ttca(v) for v in queue.time_to_tca])
    st.dataframe(view.style.map(tier_text, subset=["tier_final", "tier_now"]),
                 hide_index=True, width="stretch", column_config={
                     "event_id": "event", "tier_final": "tier", "tier_now": "tier now", "c_object_type": "object",
                     "p_critical": st.column_config.ProgressColumn("P(Critical at TCA)", min_value=0, max_value=1,
                                                                   format="percent"),
                     "risk_latest": "Pc now", "risk_forecast": "Pc at TCA",
                     "risk_hi": st.column_config.TextColumn(
                         "95% upper bound on Pc at TCA",
                         help="One-sided 95% upper bound from the conformal interval on the final Pc."),
                     "storm_flips": st.column_config.CheckboxColumn(
                         "storm-sensitive", help="Sensitivity, not a forecast: the tier changes if F10/AP rise "
                                                 "to geomagnetic-storm levels."),
                     "miss_latest_m": "miss now", "miss_forecast_m": "miss at TCA", "time_to_tca": "time to TCA",
                     "needs_review": st.column_config.CheckboxColumn(
                         "needs review", help="Too little evidence to forecast (one CDM, stale CDM or no "
                                              "covariance): an analyst must look, even if the tier is SAFE."),
                     "pc_diluted": st.column_config.CheckboxColumn(
                         "Pc diluted", help="Low Pc only because tracking is poor. See Fleet overview for whether "
                                            "this predicted anything on known outcomes."),
                 })

    # Alert-fatigue curve: where outcomes are known (train), how many truly Critical events an analyst
    # catches by working this queue top-down, against how many events they have to review.
    # The baseline ranks by the latest CDM alone (today's practice) and is always shown next to this queue.
    known = shown[shown.tier_actual.notna()]
    if (known.tier_actual == CRITICAL).any():
        in_sample = known.forecast_in_sample.astype(bool).any()  # the model was fit on these very events
        rankings = {"baseline: latest CDM": (known.assign(_r=known.tier_now.map(TIER_RANK))
                                             .sort_values(["_r", "risk_latest"], ascending=False)),
                    "this queue" + (" (in-sample)" if in_sample else ""): queue_order(known)}
        st.subheader("Alert fatigue: how far down the queue to go")
        curve, k95 = go.Figure(), {}
        for name, ranked in rankings.items():
            hits = (ranked.tier_actual == CRITICAL).cumsum().to_numpy()
            caught = hits / hits[-1]
            k95[name] = int(np.argmax(caught >= 0.95)) + 1
            curve.add_trace(go.Scatter(x=np.arange(1, len(hits) + 1), y=caught, name=name))
        curve.update_layout(xaxis_title="events reviewed, worst first", yaxis_title="share of actual Critical events caught",
                            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(curve, width="stretch")
        n, (base, mine) = len(known), k95.values()
        st.caption(f"To catch 95% of the {int((known.tier_actual == CRITICAL).sum()):,} events that really ended "
                   f"Critical: top {mine:,} of {n:,} ({mine / n:.0%}) with this queue, top {base:,} ({base / n:.0%}) "
                   "with the baseline ranking.")
        if depth := (load_metrics() or {}).get("queue_depth"):
            lo, hi = depth["ci95"]["reduction"]
            st.metric("Fewer reviews than today's ranking, at 95% recall", f"{depth['reduction']:.0%}", border=True,
                      help="All train events, out of fold, paired bootstrap over events (metrics.json).")
            st.caption(f"95% CI {lo:.0%}–{hi:.0%}: top {depth['model']:,} vs {depth['baseline']:,} events on the full "
                       "train split, whatever the filters above. The interval is wide; quote it with the number.")
        if in_sample:
            st.warning("In-sample: the model was trained on these events, so the queue's curve is optimistic. "
                       "Use the Evaluation tab's out-of-fold numbers for claims.")

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
        forecast = not e.needs_review  # refuse: no forecast and no maneuver plan on thin evidence
        if not forecast:
            st.error(f"Insufficient evidence: {text(e.review_reason)}. No forecast or maneuver is shown; "
                     "an analyst must review this event from its CDM history below.")
        elif pd.notna(e.maneuver_dr_m):  # the pipeline plans one for CRITICAL events only
            if e.maneuver_dr_m == 0:
                st.info("Suggested maneuver: none from geometry. The separation already clears the rules, so "
                        "the Critical tier comes from the forecast; an analyst decides.")
            else:
                half_orbit = np.pi / LEO_MEAN_MOTION / 60
                st.warning(f"Suggested maneuver: shift radial separation by {abs(e.maneuver_dr_m):,.0f} m "
                           f"({'outward' if e.maneuver_dr_m > 0 else 'inward'}), an along-track burn of about "
                           f"{e.maneuver_dv_mps * 100:.1f} cm/s roughly {half_orbit:.0f} min (half an orbit) "
                           "before TCA. A first-order suggestion for the analyst to verify, not a command.")

        with st.expander("Plain-English brief (Gemini)"):
            sent = brief.facts(e.to_dict() | {"event_id": ev})
            if why := brief.refusal(e.to_dict()):
                st.info(why)
            elif not brief.api_key():
                st.info("Brief off: set GEMINI_API_KEY in the environment (or a git-ignored .env) to enable it.")
            elif st.button("Explain this event"):
                try:
                    st.markdown(cached_brief(json.dumps(sent, sort_keys=True)))
                except RuntimeError as err:
                    st.error(f"Brief unavailable, so nothing is shown in its place: {err}")
            st.caption("The model sees only these fields; the tier itself comes from the rules and the forecaster.")
            st.json(sent, expanded=False)

        with st.form(f"decide-{split}-{ev}"):
            st.markdown("**Analyst decision** (the system recommends; you decide and sign)")
            analyst = st.text_input("Analyst", key="analyst")
            choice = st.radio("Decision", CHOICES, index=1, horizontal=True)
            note = st.text_input("Note")
            if st.form_submit_button("Sign decision"):
                if problem := decision_problem(analyst, choice, note, e):
                    st.error(problem)
                else:
                    record_decision({"at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "split": split,
                                     "event_id": ev, "analyst": analyst.strip(), "decision": choice,
                                     "note": note.strip(), "tier_final": e.tier_final, "p_critical": e.p_critical,
                                     "forecast_source": e.forecast_source})
                    st.rerun()
        signed = decisions[(decisions.split == split) & (decisions.event_id == ev)]
        if len(signed):
            st.caption("Decision log for this event (what the system showed when each was signed):")
            st.dataframe(signed.drop(columns=["split", "event_id"]), hide_index=True, width="stretch")

        a, b = st.columns(2)
        risk = px.line(path, x="time_to_tca", y="risk", markers=True, title="Pc per CDM update")
        tier_bands(risk, "y", (RISK_CRITICAL, 0), (RISK_ELEVATED, RISK_CRITICAL))
        miss = style.miss_axis(px.line(path, x="time_to_tca", y="miss_distance", markers=True,
                                       title="Miss distance per CDM update"), "y")
        tier_bands(miss, "y", (1, MISS_CRITICAL_M), (MISS_CRITICAL_M, MISS_ELEVATED_M))
        if forecast:  # the whisker is the one-sided 95% upper bound; the lower end is not shown (uninformative)
            risk.add_trace(go.Scatter(x=[0], y=[e.risk_forecast], mode="markers", marker_symbol="diamond",
                                      name="forecast, 95% upper bound",
                                      error_y=dict(type="data", symmetric=False, array=[e.risk_hi - e.risk_forecast],
                                                   arrayminus=[0])))
            miss.add_trace(go.Scatter(x=[0], y=[e.miss_forecast_m], mode="markers", marker_symbol="diamond",
                                      name="forecast"))
        for fig, final in ((risk, e.risk_final), (miss, e.miss_final_m)):
            if pd.notna(final):
                fig.add_trace(go.Scatter(x=[0], y=[final], mode="markers", marker_symbol="x", name="actual at TCA"))
            fig.add_vline(x=DECISION_DAYS, line_dash="dot", annotation_text="decision point")
            fig.update_xaxes(autorange="reversed", title="days to TCA")
        if e.pc_diluted:
            risk.add_trace(go.Scatter(x=[e.time_to_tca] * 2, y=[e.risk_latest, e.risk_potential], mode="lines+markers",
                                      line_dash="dot", name="if tracking were sharp (σ = d/√2)"))
        shown_pc = [*path.risk, e.risk_forecast, e.risk_hi, e.risk_final, e.risk_potential]
        lo = int(np.floor(np.nanmin(shown_pc))) if len(path) else int(RISK_FLOOR)
        style.pc_axis(risk, lo=lo, step=max(1, int(np.ceil(-lo / 8))))  # ~8 ticks, whatever the range
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
    st.subheader("Where every event sits")
    at_tca = st.radio("Position events by", ["forecast at TCA", "latest CDM"], horizontal=True) == "forecast at TCA"
    x, y = ("miss_forecast_m", "risk_forecast") if at_tca else ("miss_latest_m", "risk_latest")
    sky = shown.assign(Pc=[format_pc(v) for v in shown[y]], miss=[format_miss(v) for v in shown[x]],
                       p_crit=[format_prob(v) for v in shown.p_critical])
    scatter = px.scatter(sky, x=x, y=y, color="tier_final", color_discrete_map=TIER_COLORS,
                         category_orders={"tier_final": TIERS}, hover_name="event_id",
                         hover_data={x: False, y: False, "tier_final": True, "Pc": True, "miss": True,
                                     "p_crit": True, "c_object_type": True},
                         labels={"tier_final": "tier", "p_crit": "P(Critical at TCA)", "c_object_type": "object"})
    scatter.update_traces(marker=dict(size=6, opacity=0.7, line_width=0))
    tier_bands(scatter, "y", (RISK_CRITICAL, 0), (RISK_ELEVATED, RISK_CRITICAL))
    tier_bands(scatter, "x", (1, MISS_CRITICAL_M), (MISS_CRITICAL_M, MISS_ELEVATED_M))
    scatter.add_annotation(x=1, xref="paper", xanchor="right", y=RISK_FLOOR, yanchor="bottom", showarrow=False,
                           text="floor: Pc < 10⁻³⁰ (effectively zero)", font=dict(color=style.MUTED, size=11))
    style.miss_axis(style.pc_axis(scatter)).update_layout(height=520)
    st.plotly_chart(scatter, width="stretch")
    st.caption(f"Shaded: the tier regions (Critical: Pc ≥ {format_pc(RISK_CRITICAL)} or miss < "
               f"{format_miss(MISS_CRITICAL_M)}; Elevated: Pc ≥ {format_pc(RISK_ELEVATED)} or miss ≤ "
               f"{format_miss(MISS_ELEVATED_M)}). Colour is the final tier, which can also come from the forecaster.")

    left, right = st.columns(2)
    with left, st.container(border=True):
        mix = shown.groupby(["c_object_type", "tier_final"]).size().reset_index(name="events")
        order = shown.c_object_type.value_counts().index.tolist()[::-1]  # biggest group on top
        bars = px.bar(mix, x="events", y="c_object_type", color="tier_final", orientation="h",
                      color_discrete_map=TIER_COLORS, category_orders={"tier_final": TIERS, "c_object_type": order},
                      title="Tier mix by object type", labels={"c_object_type": "", "tier_final": "tier"})
        bars.update_layout(barnorm="percent", showlegend=False, height=340, xaxis_title="share of events (%)")
        st.plotly_chart(bars, width="stretch")
    with right, st.container(border=True):
        floor = shown[y] <= RISK_FLOOR
        above = shown[~floor]
        hist = px.histogram(above, x=y, color="tier_final", color_discrete_map=TIER_COLORS, nbins=40,
                            category_orders={"tier_final": TIERS}, title="Pc above the floor",
                            labels={"tier_final": "tier"})
        for v in (RISK_ELEVATED, RISK_CRITICAL):
            hist.add_vline(x=v, line_dash="dot", line_color=style.MUTED)
        lo = int(np.floor(above[y].min())) if len(above) else int(RISK_FLOOR)
        style.pc_axis(hist, "x", lo=lo - lo % 4, step=4).update_layout(showlegend=False, height=340,
                                                                       yaxis_title="events", bargap=0.05)
        st.plotly_chart(hist, width="stretch")
        st.caption(f"{int(floor.sum()):,} events ({floor.mean() if len(floor) else 0:.0%}) sit at the floor, "
                   "Pc < 10⁻³⁰, and are left out so the rest stays readable.")

    st.subheader("Space-weather sensitivity")
    flips = shown[shown.storm_flips.astype(bool)]
    st.caption(f"{len(flips):,} of {len(shown):,} events change tier if F10 / AP rise to geomagnetic-storm levels "
               "(denser upper atmosphere, more drag uncertainty). A sensitivity check, not a storm forecast: "
               "these are the events to re-check if a storm arrives.")
    if len(flips):
        ranked_flips = (flips.assign(jump=flips.p_critical_storm - flips.p_critical,
                                     miss=[format_miss(v) for v in flips.miss_latest_m])
                        .sort_values("jump", ascending=False)
                        [["event_id", "tier_final", "p_critical", "p_critical_storm", "miss", "c_object_type"]])
        storm_cols = {"event_id": "event", "tier_final": "tier", "miss": "miss now", "c_object_type": "object",
                      "p_critical": st.column_config.ProgressColumn("P(Critical) now", min_value=0, max_value=1,
                                                                    format="percent"),
                      "p_critical_storm": st.column_config.ProgressColumn("P(Critical) in a storm", min_value=0,
                                                                          max_value=1, format="percent")}
        st.dataframe(ranked_flips.head(10).style.map(tier_text, subset=["tier_final"]), hide_index=True,
                     width="stretch", column_config=storm_cols)
        if len(ranked_flips) > 10:
            with st.expander(f"All {len(ranked_flips):,} storm-sensitive events"):
                st.dataframe(ranked_flips.style.map(tier_text, subset=["tier_final"]), hide_index=True,
                             width="stretch", column_config=storm_cols)

with eval_tab:
    st.subheader("Baseline vs model (cross-validated on train events)")
    metrics = load_metrics()
    if metrics is None:
        st.info("No data/processed/metrics.json yet. Run `python -m src.score` to measure the baseline.")
    else:
        table = metrics_table(metrics)
        st.dataframe(table, width="stretch", column_config={
            name: st.column_config.NumberColumn(name, format="%.3f") for name in ("baseline", "model")})
        run = metrics.get("model") or metrics.get("baseline") or {}
        st.caption(f"{run.get('n_critical', '?'):,} actual Critical events among {run.get('n_events', '?'):,}. "
                   "Baseline: the latest CDM's risk carried to TCA. 95% CIs bootstrap over events. "
                   "Lower is better for MSE and loss.")
        if "model" not in metrics:
            st.info("No trained model yet, so these are baseline numbers only. "
                    "The forecaster's full training run is waiting for the team's go-ahead.")

    # Pc dilution is shown as a hypothesis tested against outcomes, not as a warning: on the training data
    # it does not predict danger, and the dashboard says so whenever the outcomes show it.
    st.subheader("Pc dilution: does a 'low because poorly tracked' Pc hide danger?")
    flagged = shown.pc_diluted.astype(bool)
    st.caption(f"{int(flagged.sum()):,} of {len(shown):,} events are flagged: their covariance is wider than the "
               "σ = d/√2 at which Pc peaks, so sharper tracking alone could raise Pc by 10× or more.")
    known = shown[shown.tier_actual.notna() & shown.risk_final.notna()]
    if known.empty:
        st.info("No known outcomes for these events, so the flag can't be checked here. "
                "Choose the train split to see whether it predicted anything.")
    else:
        check = (known.assign(flag=np.where(known.pc_diluted.astype(bool), "flagged", "not flagged"),
                              critical=known.tier_actual == CRITICAL, change=known.risk_final - known.risk_latest)
                 .groupby("flag").agg(events=("critical", "size"), ended_critical=("critical", "mean"),
                                      median_change=("change", "median")))
        st.dataframe(check, width="stretch", column_config={
            "ended_critical": st.column_config.NumberColumn("ended Critical", format="percent"),
            "median_change": st.column_config.NumberColumn("median change in log10 Pc to TCA", format="%.1f")})
        if {"flagged", "not flagged"} <= set(check.index):
            f, n = check.ended_critical["flagged"], check.ended_critical["not flagged"]
            if f > n:
                st.success(f"Flagged events ended Critical more often ({f:.1%} vs {n:.1%}).")
            else:
                st.warning(f"Measured failure: flagged events ended Critical no more often than the rest "
                           f"({f:.1%} vs {n:.1%}). As tracking improves, σ usually falls far below d/√2, so the "
                           "peak Pc is never reached. Treat the flag as a tracking-quality note, not a risk signal.")

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
