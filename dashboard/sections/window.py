"""Next 24 hours: replay of encounters reaching the 1 km / 5 km zones."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)


def render(ctx):
    shown, history, split = ctx.shown, ctx.history, ctx.split
    queued, decisions = ctx.queued, ctx.decisions
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
