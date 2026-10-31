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

    # Alert-fatigue curve: where outcomes are known (train), how many truly Critical events an analyst
    # catches by working this queue top-down, against how many events they have to review.
    # The baseline ranks by the latest CDM alone (today's practice) and is always shown next to this queue.
    known = shown[shown.tier_actual.notna()]
    if not (known.tier_actual == CRITICAL).any():
        return
    in_sample = known.forecast_in_sample.astype(bool).any()  # the model was fit on these very events
    rankings = {"Today: latest CDM only": (known.assign(_r=known.tier_now.map(TIER_RANK))
                                           .sort_values(["_r", "risk_latest"], ascending=False)),
                "This queue" + (" (in-sample)" if in_sample else ""): queue_order(known)}
    st.markdown(style.section("01·B", "How far down the queue to go"), unsafe_allow_html=True)

    chart_col, number_col = st.columns([7, 4], gap="large")
    curve, k95 = go.Figure(), {}
    for (name, ranked), line in zip(rankings.items(), (dict(color=style.MUTED, dash="dash", width=2),
                                                       dict(color=style.TEXT, width=3))):
        hits = (ranked.tier_actual == CRITICAL).cumsum().to_numpy()
        caught = hits / hits[-1]
        k95[name] = int(np.argmax(caught >= 0.95)) + 1
        curve.add_trace(go.Scatter(x=np.arange(1, len(hits) + 1), y=caught, name=name, line=line,
                                   hovertemplate="%{x:,} reviewed → %{y:.0%} caught<extra></extra>"))
    curve.add_hline(y=0.95, line=dict(color=style.LINE, width=1), annotation_text="95% of Criticals",
                    annotation_font=dict(color=style.MUTED, size=11), annotation_position="bottom right")
    curve.update_layout(height=400, xaxis_title="Events reviewed, worst first",
                        yaxis=dict(title="Actual Critical events caught", tickformat=".0%", range=[0, 1.02]),
                        legend=dict(orientation="h", y=1.08, x=0))
    with chart_col:
        st.markdown(style.label("Train events with known outcomes"), unsafe_allow_html=True)
        st.plotly_chart(curve, width="stretch", config={"displayModeBar": False})
        n, (base, mine) = len(known), k95.values()
        st.caption(f"To catch 95% of the {int((known.tier_actual == CRITICAL).sum()):,} events that really ended "
                   f"Critical: top {mine:,} of {n:,} ({mine / n:.0%}) with this queue, top {base:,} ({base / n:.0%}) "
                   "with the baseline ranking.")
    with number_col:
        if depth := (load_metrics() or {}).get("queue_depth"):
            lo, hi = depth["ci95"]["reduction"]
            with st.container(key="kpis-queue-depth"):
                st.metric("Fewer reviews than today's ranking, at 95% recall", f"{depth['reduction']:.0%}",
                          help="All train events, out of fold, paired bootstrap over events (metrics.json).")
            st.caption(f"95% CI {lo:.0%}–{hi:.0%}: top {depth['model']:,} vs {depth['baseline']:,} events on the full "
                       "train split, whatever the filters above. The interval is wide; quote it with the number.")
        if in_sample:
            st.warning("In-sample: the model was trained on these events, so the queue's curve is optimistic. "
                       "Use the Evaluation tab's out-of-fold numbers for claims.")
