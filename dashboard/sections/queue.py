"""Operations queue: the triage list, worst first, and the alert-fatigue curve."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)

# Shown by default; the rest stays in the table's data (and in Event detail) without crowding the view.
VISIBLE = ["event_id", "tier_final", "p_critical", "risk_latest", "risk_hi", "miss_latest_m", "time_to_tca",
           "c_object_type", "needs_review", "decision"]


def render(ctx):
    shown, queued = ctx.shown, ctx.queued
    queue = queue_order(shown[queued])
    n_crit, n_review = int((queue.tier_final == CRITICAL).sum()), int(queue.needs_review.astype(bool).sum())

    st.markdown(style.section("01", "Operations queue"), unsafe_allow_html=True)
    st.markdown(f'<p class="bm-lede"><b>{len(queue):,} events</b> need an analyst, sorted by tier and then by the '
                f'calibrated probability of being Critical at closest approach. <b>{n_crit:,}</b> are Critical; '
                f'<b>{n_review:,}</b> have too little evidence to forecast and stay here whatever their tier.</p>',
                unsafe_allow_html=True)

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
