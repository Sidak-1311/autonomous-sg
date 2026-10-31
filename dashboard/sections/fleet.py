"""Fleet overview: where every event sits, tier mix, Pc distribution, space-weather sensitivity."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)


def render(ctx):
    shown, history, split = ctx.shown, ctx.history, ctx.split
    queued, decisions = ctx.queued, ctx.decisions
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
