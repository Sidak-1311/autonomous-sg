"""Operations queue: the triage list, worst first, and the alert-fatigue curve."""
from html import escape

from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)

# Shown by default; the rest stays in the table's data (and in Event detail) without crowding the view.
VISIBLE = ["event_id", "tier_final", "p_critical", "risk_latest", "risk_hi", "miss_latest_m", "time_to_tca",
           "c_object_type", "needs_review", "decision"]


def render(ctx):
    shown, queued = ctx.shown, ctx.queued
    queue = queue_order(shown[queued])
    n_crit, n_review = int((queue.tier_final == CRITICAL).sum()), int(queue.needs_review.astype(bool).sum())
    n_open = int((queue.decision == "").sum())
    n_watch = int(queue.watch.astype(bool).sum())

    st.markdown(style.heading("01", "Operations queue",
                              "Every event that needs an analyst, worst first: by tier, then by the calibrated "
                              "probability of being Critical at closest approach."), unsafe_allow_html=True)
    st.markdown(style.cells([
        style.stat(f"{len(queue):,}", "In the queue", sub="non-Safe, Watch and every refusal"),
        style.stat(f"{n_crit:,}", "Critical", "CRITICAL", sub="rules, or P(Critical) ≥ 15%"),
        style.stat(f"{n_watch:,}", "Watch", sub="P(Critical) below 15% but above the recall threshold"),
        style.stat(f"{n_review:,}", "Insufficient evidence", sub="no forecast; an analyst decides"),
        style.stat(f"{n_open:,}", "Awaiting a decision", sub="no signed analyst decision yet"),
    ]), unsafe_allow_html=True)

    filtered = _filter(queue)
    shown_rows = filtered.head(TABLE_ROWS)
    if filtered.empty:
        st.markdown(style.card('<span class="dim">No queued events match these filters.</span>'),
                    unsafe_allow_html=True)
    else:
        st.markdown(style.html_table(shown_rows, TABLE_COLUMNS, max_height=640, rank=True, foot=(
            f"<span>Showing {len(shown_rows):,} of {len(filtered):,}"
            f"{f' (filtered from {len(queue):,})' if len(filtered) < len(queue) else ''} · worst first</span>"
            "<span>Open any event in 02 Event detail</span>")), unsafe_allow_html=True)

    with st.expander("Export or sort the full queue as a grid"):
        _grid(filtered)


TABLE_ROWS = 300
DECISIONS = ["Undecided", "Maneuver", "Monitor", "Dismiss"]
WATCH = "WATCH"


def _filter(queue):
    """Filters on the table itself. An empty pill selection means 'no filter on that field'."""
    with st.container(key="queue-filters"):
        f = st.columns([2.6, 4.4, 4.4, 5.0], gap="medium", vertical_alignment="top")
        search = f[0].text_input("Find event", placeholder="Event ID", key="queue_search")
        review_only = f[0].toggle("Review only", key="queue_review",
                                  help="Only events with too little evidence to forecast")
        tiers = f[1].pills("Tier", [CRITICAL, WATCH, ELEVATED, SAFE], selection_mode="multi", key="queue_tier",
                           help="Watch is a status, not a tier: P(Critical) between the recall threshold and 15%")
        decisions = f[2].pills("Decision", DECISIONS, selection_mode="multi", key="queue_decision")
        objects = sorted(queue.c_object_type.dropna().unique())
        picked = f[3].pills("Object", objects, selection_mode="multi", key="queue_object",
                            format_func=lambda o: str(o).title())
    out = queue
    if search.strip():
        out = out[out.event_id.astype(str).str.contains(search.strip(), regex=False)]
    if tiers:
        out = out[out.tier_final.isin(tiers) | (out.watch.astype(bool) & (WATCH in tiers))]
    if decisions:
        out = out[out.decision.replace("", "Undecided").isin(decisions)]
    if picked:
        out = out[out.c_object_type.isin(picked)]
    if review_only:
        out = out[out.needs_review.astype(bool)]
    return out


def _why(v, r) -> str:
    """The reason's first clause in the table (the rule or the refusal); the full text on hover."""
    text = v if isinstance(v, str) else ""
    if text.startswith("INSUFFICIENT"):
        head = "Insufficient evidence: " + text.split("(", 1)[1].split(")")[0] if "(" in text else text
    else:
        # 'Critical now (miss 97 m < 1,000 m); …' -> 'miss 97 m < 1,000 m': the pill already names the tier
        first = text.split(";")[0]
        head = first.split("(", 1)[1].rsplit(")", 1)[0] if "(" in first else first
    return f'<span class="why" title="{escape(text)}">{escape(head)}</span>'


def _event(v, r) -> str:
    """Event id, with the signed decision and its flags (review / storm-sensitive / σ-limited)
    as small chips underneath."""
    flags = [style.chip(f"✓ {r.decision}") if r.decision else "",
             style.chip("Review") if r.needs_review else "", style.chip("Storm", on=False) if r.storm_flips else "",
             style.chip("σ-limited", on=False) if r.pc_diluted else ""]
    chips = "".join(flags)
    return f'<span class="strong">{v}</span>' + (f'<span class="sub">{chips}</span>' if chips else "")


TABLE_COLUMNS = [
    ("event_id", "Event", "l", _event),
    ("tier_final", "Tier", "l",
     lambda v, r: style.tier_pill(v) + (f' {style.status_pill("watch")}' if r.watch else "")),
    ("p_critical", "P(Critical at TCA)", "l", lambda v, r: style.prob_bar(v)),
    ("risk_latest", "Pc now · 95% bound at TCA", "r",
     lambda v, r: f'{format_pc(v)}<span class="sub">≤ {format_pc(r.risk_hi)} at TCA</span>'),
    ("miss_latest_m", "Miss now", "r", lambda v, r: format_miss(v)),
    ("time_to_tca", "Time to TCA", "r", lambda v, r: format_ttca(v)),
    ("c_object_type", "Object", "l", lambda v, r: f'<span class="dim">{str(v).title()}</span>'),
    ("reason", "Why", "l", _why),
]


def _grid(queue):
    """The plain sortable grid, for CSV export and column sorting."""
    pc_text, miss_text = (lambda s: [format_pc(v) for v in s]), (lambda s: [format_miss(v) for v in s])
    view = queue[["event_id", "tier_final", "decision", "needs_review", "tier_now", "p_critical", "risk_latest",
                  "risk_forecast", "risk_hi", "miss_latest_m", "miss_forecast_m", "time_to_tca", "c_object_type",
                  "storm_flips", "pc_diluted", "reason", "drivers"]].assign(
        risk_latest=pc_text(queue.risk_latest), risk_forecast=pc_text(queue.risk_forecast),
        risk_hi=pc_text(queue.risk_hi), miss_latest_m=miss_text(queue.miss_latest_m),
        miss_forecast_m=miss_text(queue.miss_forecast_m), time_to_tca=[format_ttca(v) for v in queue.time_to_tca])
    st.dataframe(view.style.map(tier_text, subset=["tier_final", "tier_now"]),
                 hide_index=True, width="stretch", height=520, column_order=VISIBLE, column_config={
                     "event_id": st.column_config.NumberColumn("Event", format="%d", width="small"),
                     "tier_final": st.column_config.TextColumn("Tier", width="small"),
                     "tier_now": "Tier now", "c_object_type": "Object", "decision": "Decision",
                     "p_critical": st.column_config.ProgressColumn("P(Critical at TCA)", min_value=0, max_value=1,
                                                                   format="percent", width="medium"),
                     "risk_latest": "Pc now", "risk_forecast": "Pc at TCA",
                     "risk_hi": st.column_config.TextColumn(
                         "95% upper bound, Pc at TCA",
                         help="One-sided 95% upper bound from the conformal interval on the final Pc."),
                     "storm_flips": st.column_config.CheckboxColumn(
                         "Storm-sensitive", help="Sensitivity, not a forecast: the tier changes if F10/AP rise "
                                                 "to geomagnetic-storm levels."),
                     "miss_latest_m": "Miss now", "miss_forecast_m": "Miss at TCA", "time_to_tca": "Time to TCA",
                     "needs_review": st.column_config.CheckboxColumn(
                         "Review", help="Too little evidence to forecast (one CDM, stale CDM or no covariance): "
                                        "an analyst must look, even if the tier is SAFE."),
                     "pc_diluted": st.column_config.CheckboxColumn(
                         "Pc diluted", help="Low Pc only because tracking is poor. See Evaluation for whether "
                                            "this predicted anything on known outcomes."),
                     "reason": "Reason", "drivers": "Drivers",
                 })
    st.markdown(style.label("Click a column to sort · open an event in 02 Event detail for its reason, drivers and "
                            "history"), unsafe_allow_html=True)
