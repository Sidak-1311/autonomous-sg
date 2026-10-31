"""Debris cascade: what each conjunction would do if it became a collision (src/cascade.py), and the events
ranked by consequence (Pc x fragments) instead of by probability alone."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)

CASCADE_COLS = ["frag_10cm", "catastrophic", "energy_j_per_g", "shell_alt_km", "shell_missions", "chain_p1y_max",
                "debris_expected"]


def pct(p: float) -> str:
    """Small probabilities as percentages that keep two significant figures (0.021%, not 0.0%)."""
    return "n/a" if pd.isna(p) else f"{p * 100:.2g}%"


def has_cascade(df) -> bool:
    return set(CASCADE_COLS) <= set(df.index if isinstance(df, pd.Series) else df.columns)


def cascade_card(e: pd.Series) -> str:
    """Event detail card: what this collision would do. Empty when the pipeline output predates the cascade."""
    if not has_cascade(e) or pd.isna(e.frag_10cm):
        return ""
    kind = "catastrophic: both bodies break up" if e.catastrophic else "non-catastrophic: only the smaller body"
    if e.shell_missions:
        chain = (f"{int(e.shell_missions)} other mission{'s' if e.shell_missions != 1 else ''} share its "
                 f"±50 km shell at {e.shell_alt_km:,.0f} km; the most exposed would gain a "
                 f"<b>{pct(e.chain_p1y_max)}</b> chance per year of being hit.")
    else:
        chain = f"No other mission in this dataset shares its ±50 km shell at {e.shell_alt_km:,.0f} km."
    return (f'<div class="bm-card" style="margin-bottom:0.8rem">{style.label("If this collision happened")}'
            f'<div style="margin-top:0.45rem;line-height:1.45"><span style="font:800 2rem {style.FONT};'
            f'letter-spacing:-0.05em">{e.frag_10cm:,.0f} fragments</span> of 10 cm or more<br>'
            f"{kind} ({e.energy_j_per_g:,.0f} J/g). {chain}</div></div>")


def render(ctx):
    shown = ctx.shown
    st.markdown(style.section("06", "Debris cascade"), unsafe_allow_html=True)
    if not has_cascade(shown) or shown.frag_10cm.isna().all():
        st.info("No cascade columns in the pipeline output yet. Re-run `python -m src.score`.")
        return
    ev = shown[shown.frag_10cm.notna()]
    worst = ev.loc[ev.chain_p1y_max.idxmax()]
    st.markdown(
        f'<p class="bm-lede">If these conjunctions became collisions, <b>{ev.catastrophic.mean():.0%}</b> would be '
        f"catastrophic, with a median of <b>{ev.frag_10cm.median():,.0f}</b> fragments each. The worst would give "
        f"another mission a <b>{pct(worst.chain_p1y_max)}</b> yearly chance of being hit (event {int(worst.event_id)}).</p>",
        unsafe_allow_html=True)

    stats = [(f"{ev.catastrophic.mean():.0%}", "Catastrophic if hit"), (f"{ev.frag_10cm.median():,.0f}", "Median fragments"),
             (pct(ev.chain_p1y_max.max()), "Worst chain risk per year"),
             (f"{ev.debris_expected.sum():,.1f}", "Expected fragments, all events")]
    for col, (value, caption) in zip(st.columns(len(stats)), stats):
        col.markdown(style.hero_stat(value, caption, size="row"), unsafe_allow_html=True)

    floor = ev.risk_forecast <= RISK_FLOOR   # Pc effectively zero: no expected debris, and a wall on the chart
    st.markdown(f'<div style="margin-top:2.2rem">{style.label("Probability vs consequence · each point is one event")}'
                "</div>", unsafe_allow_html=True)
    fig = px.scatter(ev[~floor], x="risk_forecast", y="frag_10cm", color="tier_final", color_discrete_map=TIER_COLORS,
                     category_orders={"tier_final": TIERS}, log_y=True, hover_name="event_id",
                     hover_data={"risk_forecast": False, "tier_final": True, "frag_10cm": ":,.0f",
                                 "shell_missions": True, "chain_p1y_max": ":.2%"},
                     labels={"frag_10cm": "fragments ≥ 10 cm", "tier_final": "tier", "shell_missions": "missions in shell",
                             "chain_p1y_max": "chain risk / yr"})
    fig.update_traces(marker=dict(size=6, opacity=0.7, line_width=0))
    lo = int(np.floor(ev.risk_forecast[~floor].min())) if (~floor).any() else int(RISK_FLOOR)
    style.pc_axis(fig, "x", lo=lo - lo % 5, step=5).update_layout(height=420, yaxis_title="fragments ≥ 10 cm (log)",
                                                                  legend_title_text="")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"{int(floor.sum()):,} events with Pc < 10⁻³⁰ are left out: their expected debris is effectively zero. "
               "Top right is the danger: likely and destructive.")

    st.markdown(style.label("Ranked by consequence · expected fragments = Pc at TCA × fragments if hit"),
                unsafe_allow_html=True)
    top = ev.nlargest(15, "debris_expected").assign(
        pc=lambda d: [format_pc(v) for v in d.risk_forecast], chain=lambda d: [pct(v) for v in d.chain_p1y_max],
        hit=lambda d: np.where(d.catastrophic.astype(bool), "catastrophic", "partial"))
    st.dataframe(top[["event_id", "tier_final", "pc", "frag_10cm", "hit", "shell_missions", "chain", "debris_expected"]]
                 .style.map(tier_text, subset=["tier_final"]),
                 hide_index=True, width="stretch", column_config={
                     "event_id": "event", "tier_final": "tier", "pc": "Pc at TCA", "hit": "breakup",
                     "frag_10cm": st.column_config.NumberColumn("fragments if hit", format="%,.0f"),
                     "shell_missions": st.column_config.NumberColumn("missions in shell", format="%d"),
                     "chain": "chain risk / yr",
                     "debris_expected": st.column_config.NumberColumn("expected fragments", format="%.3g")})
    with st.expander("How this is estimated, and its limits"):
        st.markdown(
            "Fragments: NASA Standard Breakup Model; a collision is catastrophic at 40 J/g or more. Masses come from "
            "size and drag area-to-mass, capped at 10 t. Chain risk: the fragments spread over a ±50 km shell and "
            "raise every other mission's collision rate by density × 10 km/s × its cross-section. First-order: real "
            "clouds spread unevenly, and only the 18 missions in this dataset are counted, so read chain risk as an "
            "order of magnitude.")
