"""Operations queue: the triage list, worst first, and the alert-fatigue curve."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)


def render(ctx):
    shown, history, split = ctx.shown, ctx.history, ctx.split
    queued, decisions = ctx.queued, ctx.decisions
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
