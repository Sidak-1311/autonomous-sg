"""Event detail: reason, refusal, maneuver, Gemini brief, analyst decision, histories, 3D geometry."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)


def render(ctx):
    shown, history, split = ctx.shown, ctx.history, ctx.split
    queued, decisions = ctx.queued, ctx.decisions
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
