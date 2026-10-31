"""Event detail: a header strip of big numbers, histories and 3D geometry on the left, and on the right the
maneuver, refusal, analyst decision and Gemini summary."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)
from dashboard.sections.cascade import cascade_card

GREYS = [[0, "#F2F2F2"], [1, "#4A4A4A"]]  # time along a path: not a tier, so no tier colour
NO_TOOLBAR = {"displayModeBar": False}  # it sat on top of the chart titles; zoom still works by drag


def stat(value: str, caption: str, tier: str | None = None) -> str:
    """A header-strip number: the shared row size, a notch smaller so six fit across."""
    # display tracking (-0.07em) swallows the spaces in '2 d 06 h' and '202 m', so loosen it here
    return style.hero_stat(value, caption, tier, size="row").replace(
        'style="color:', 'style="font-size:clamp(1.6rem,2.4vw,2.6rem);letter-spacing:-0.02em;color:', 1)


def tier_probability_bar(e) -> str:
    """The forecaster's final-tier probabilities (p_safe + p_elevated + p_critical = 1) as one stacked bar, each
    segment labelled in words. Empty when the columns are missing or not filled (older output, refusals)."""
    keys = [("p_safe", SAFE), ("p_elevated", ELEVATED), ("p_critical", CRITICAL)]
    if not all(k in e.index and pd.notna(e[k]) for k, _ in keys):
        return ""
    probs = [(t, float(e[k])) for k, t in keys]
    segs = "".join(f'<div style="width:{p * 100:.2f}%;background:{TIER_COLORS[t]}"></div>' for t, p in probs if p > 0)
    legend = " · ".join(f'<span style="color:{TIER_COLORS[t]}">{t.title()}</span> {format_prob(p)}' for t, p in probs)
    return (f'<div style="margin:0.4rem 0 0.9rem">{style.label("Final tier at TCA · forecast probability")}'
            f'<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin:0.45rem 0 0.35rem;'
            f'background:{style.LINE}">{segs}</div><div style="font-size:0.9rem">{legend}</div></div>')


def reason_lines(reason: str) -> str:
    """The pipeline's one-line reason ('Critical now (…); forecast …; 95% upper bound …; Pc falling …') as short
    labelled lines, one clause each."""
    parts = [p.strip() for p in reason.split("; ") if p.strip()]
    return "<br>".join(f"<b>{p[:1].upper()}</b>{p[1:]}" if i == 0 else p for i, p in enumerate(parts))


def _signed_at(v, _) -> str:
    t = pd.to_datetime(v, utc=True, errors="coerce")
    return t.strftime("%d %b %H:%M UTC") if t is not pd.NaT else f'<span class="dim">{v}</span>'


def _opt_text(v, _) -> str:
    """Free text (analyst, note), escaped; a dash when empty."""
    from html import escape
    return escape(str(v)) if isinstance(v, str) and v.strip() else '<span class="dim">—</span>'


# Newest first: when it was signed, who, what, and what the system showed at that moment
DECISION_LOG_COLUMNS = [
    ("at_utc", "Signed", "l", _signed_at),
    ("analyst", "Analyst", "l", _opt_text),
    ("decision", "Decision", "l", lambda v, r: style.chip(str(v), on=v == "Maneuver")),
    ("note", "Note", "l", _opt_text),
    ("tier_final", "Tier shown", "l", lambda v, r: style.tier_pill(v)),
    ("p_critical", "P(Critical) shown", "l", lambda v, r: style.prob_bar(v)),
    ("forecast_source", "Forecast", "l", lambda v, r: f'<span class="dim">{v}</span>'),
]


def note_card(title: str, body: str, tier: str | None = None) -> str:
    edge = f"border-left:3px solid {TIER_COLORS[tier]};" if tier else ""
    return (f'<div class="bm-card" style="{edge}margin-bottom:0.8rem">{style.label(title)}'
            f'<div style="margin-top:0.45rem;line-height:1.45">{body}</div></div>')


def render(ctx):
    shown, history, split, decisions = ctx.shown, ctx.history, ctx.split, ctx.decisions
    st.markdown(style.section("02", "Event detail"), unsafe_allow_html=True)
    ranked = queue_order(shown)
    if ranked.empty:
        st.info("No events match the filters.")
        return
    by_id = ranked.set_index("event_id")
    ev = st.selectbox("Event (worst first)", ranked.event_id, format_func=lambda e: f"{e} · {by_id.tier_final[e]}")
    e = by_id.loc[ev]
    path = history[history.event_id == ev].sort_values("time_to_tca", ascending=False)
    text = lambda v: v if isinstance(v, str) else ""  # noqa: E731  (parquet gives None/NaN for missing strings)
    forecast = not e.needs_review  # refuse: no forecast and no maneuver plan on thin evidence

    strip = [(e.tier_final, "Tier", e.tier_final), (format_ttca(e.time_to_tca), "Time to TCA", None),
             (format_pc(e.risk_latest), "Pc now", None),
             (format_prob(e.p_critical) if forecast else "withheld", "P(Critical at TCA)", None),
             (format_miss(e.miss_latest_m), "Miss now", None),
             (format_pc(e.risk_hi) if forecast else "withheld", "95% bound at TCA", None)]
    for col, (value, caption, tier) in zip(st.columns([1.5, 1, 1, 1, 1, 1.2]), strip):  # the tier word is widest
        col.markdown(stat(value, caption, tier), unsafe_allow_html=True)
    if forecast and (bar := tier_probability_bar(e)):
        st.markdown(bar, unsafe_allow_html=True)
    st.markdown(f'<p class="bm-lede" style="max-width:none;line-height:1.55">{e.c_object_type} · '
                f"{reason_lines(text(e.reason))}</p>", unsafe_allow_html=True)
    if text(e.drivers):
        st.markdown(f'{style.label("Drivers")}<p style="margin-top:0.2rem">{text(e.drivers)}</p>',
                    unsafe_allow_html=True)
    st.caption("95% bound at TCA: one-sided 95% upper bound on Pc at closest approach (conformal).")

    main, side = st.columns([2, 1], gap="large")
    with side:
        if not forecast:
            st.markdown(note_card("Insufficient evidence", f"{text(e.review_reason)}. No forecast or maneuver is "
                                  "shown; review this event from its CDM history.", CRITICAL), unsafe_allow_html=True)
        elif pd.notna(e.maneuver_dr_m):  # the pipeline plans one for CRITICAL events only
            if e.maneuver_dr_m == 0:
                st.markdown(note_card("Suggested maneuver", "None from geometry: the separation already clears the "
                                      "rules, so the Critical tier comes from the forecast. You decide."),
                            unsafe_allow_html=True)
            else:
                half_orbit = np.pi / LEO_MEAN_MOTION / 60
                st.markdown(note_card(
                    "Suggested maneuver",
                    f'<span style="font:800 2rem {style.FONT};letter-spacing:-0.05em">'
                    f"{abs(e.maneuver_dr_m):,.0f} m ({'outward' if e.maneuver_dr_m > 0 else 'inward'})</span><br>"
                    f"radial shift: an along-track burn of {e.maneuver_dv_mps * 100:.1f} cm/s about "
                    f"{half_orbit:.0f} min (half an orbit) before TCA. A first-order suggestion to verify.",
                    CRITICAL), unsafe_allow_html=True)
        if card := cascade_card(e):
            st.markdown(card, unsafe_allow_html=True)

        with st.form(f"decide-{split}-{ev}"):
            st.markdown(style.label("Analyst decision · you decide and sign"), unsafe_allow_html=True)
            analyst = st.text_input("Analyst", key="analyst")
            choice = st.radio("Decision", CHOICES, index=1, horizontal=True)
            note = st.text_input("Note")
            if st.form_submit_button("Sign decision", type="primary"):
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
            st.markdown(style.label("Decision log · what the system showed at signing"), unsafe_allow_html=True)
            st.markdown(style.html_table(signed.sort_values("at_utc", ascending=False), DECISION_LOG_COLUMNS,
                                         max_height=320), unsafe_allow_html=True)
            with st.expander("Decision log as a grid"):
                st.dataframe(signed.drop(columns=["split", "event_id"]), hide_index=True, width="stretch")

        with st.expander("Generated summary (Gemini)"):
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
            st.caption("Rephrases the fields below and nothing else; the tier comes from the rules and the forecaster.")
            st.json(sent, expanded=False)

    with main:
        risk = px.line(path, x="time_to_tca", y="risk", markers=True)
        tier_bands(risk, "y", (RISK_CRITICAL, 0), (RISK_ELEVATED, RISK_CRITICAL))
        miss = style.miss_axis(px.line(path, x="time_to_tca", y="miss_distance", markers=True), "y")
        tier_bands(miss, "y", (1, MISS_CRITICAL_M), (MISS_CRITICAL_M, MISS_ELEVATED_M))
        for fig in (risk, miss):
            fig.update_traces(line=dict(color=style.TEXT, width=1.5), marker=dict(size=5, color=style.TEXT))
        if forecast:  # the whisker is the one-sided 95% upper bound; the lower end is not shown (uninformative)
            risk.add_trace(go.Scatter(x=[0], y=[e.risk_forecast], mode="markers", marker_symbol="diamond",
                                      marker=dict(size=10, color=TIER_COLORS[e.tier_final]),
                                      name="forecast, 95% upper bound",
                                      error_y=dict(type="data", symmetric=False, array=[e.risk_hi - e.risk_forecast],
                                                   arrayminus=[0], color=style.MUTED)))
            miss.add_trace(go.Scatter(x=[0], y=[e.miss_forecast_m], mode="markers", marker_symbol="diamond",
                                      marker=dict(size=10, color=TIER_COLORS[e.tier_final]), name="forecast"))
        for fig, final in ((risk, e.risk_final), (miss, e.miss_final_m)):
            if pd.notna(final):
                fig.add_trace(go.Scatter(x=[0], y=[final], mode="markers", marker_symbol="x",
                                         marker=dict(size=10, color=style.TEXT), name="actual at TCA"))
            fig.add_vline(x=DECISION_DAYS, line_dash="dot", line_color=style.MUTED, annotation_text="decision point",
                          annotation_font_color=style.MUTED)
            fig.update_xaxes(autorange="reversed", title="days to TCA")
            fig.update_layout(height=340, legend=dict(y=-0.28, yanchor="top"), margin=dict(t=10))
        if e.pc_diluted:
            risk.add_trace(go.Scatter(x=[e.time_to_tca] * 2, y=[e.risk_latest, e.risk_potential], mode="lines+markers",
                                      line=dict(dash="dot", color=style.MUTED),
                                      name="if tracking were sharp (σ = d/√2)"))
        shown_pc = [*path.risk, e.risk_forecast, e.risk_hi, e.risk_final, e.risk_potential]
        lo = int(np.floor(np.nanmin(shown_pc))) if len(path) else int(RISK_FLOOR)
        style.pc_axis(risk, lo=lo, step=max(1, int(np.ceil(-lo / 8))))  # ~8 ticks, whatever the range
        a, b = st.columns(2)
        a.markdown(style.label("Pc per CDM update"), unsafe_allow_html=True)
        a.plotly_chart(risk, width="stretch", config=NO_TOOLBAR)
        b.markdown(style.label("Miss distance per CDM update"), unsafe_allow_html=True)
        b.plotly_chart(miss, width="stretch", config=NO_TOOLBAR)

        st.markdown(style.label("Encounter geometry · predicted miss vector per CDM, 1 km and 5 km spheres"),
                    unsafe_allow_html=True)
        geo = go.Figure([
            sphere(MISS_CRITICAL_M, CRITICAL, "1 km"), sphere(MISS_ELEVATED_M, ELEVATED, "5 km"),
            go.Scatter3d(x=path.relative_position_t, y=path.relative_position_n, z=path.relative_position_r,
                         mode="lines+markers", name="predicted miss vector", line=dict(color=style.MUTED),
                         marker=dict(size=4, color=path.time_to_tca, colorscale=GREYS,
                                     colorbar=dict(title="days to TCA", thickness=6, outlinewidth=0, len=0.5))),
            go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5, color=style.TEXT), name="satellite"),
        ])
        geo.data[0].opacity, geo.data[1].opacity = 0.18, 0.05  # the 5 km shell is context, not the subject
        axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=style.LINE, color=style.MUTED, tickformat=",")
        geo.update_layout(scene=dict(xaxis=dict(title="t (m)", **axis), yaxis=dict(title="n (m)", **axis),
                                     zaxis=dict(title="r (m)", **axis), aspectmode="data"),
                          height=440, margin=dict(l=0, r=0, t=0, b=40), legend=dict(orientation="h", y=0))
        st.plotly_chart(geo, width="stretch")
