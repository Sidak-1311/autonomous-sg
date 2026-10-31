"""Next 24 hours: replay of encounters reaching the 1 km / 5 km zones."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import style
from dashboard.sections.common import DECISION_DAYS
from dashboard.sections.fleet import stat
from src.tiering import CRITICAL, ELEVATED, MISS_CRITICAL_M, MISS_ELEVATED_M, TIER_COLORS, format_ttca

HOURS = np.arange(24)
FUTURE_OPACITY = 0.25


def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def _hourly(shown, start: float):
    """Counts per hour of the window: inside 1 km, 1-5 km, and the running Critical total."""
    # ponytail: events have no shared clock, so each event's decision-point CDM counts as "now" and its TCA
    # falls time_to_tca days later. Swap in real epochs if a dataset carries them.
    real = shown.miss_final_m.notna()
    tca = shown.assign(miss_tca=shown.miss_final_m.where(real, shown.miss_forecast_m), real=real)
    win = tca[(tca.time_to_tca >= start) & (tca.time_to_tca < start + 1)].copy()
    win["hour"] = ((win.time_to_tca - start) * 24).astype(int).clip(0, 23)
    inside = win[win.miss_tca < MISS_CRITICAL_M].groupby("hour").size().reindex(HOURS, fill_value=0)
    near = win[win.miss_tca.between(MISS_CRITICAL_M, MISS_ELEVATED_M)].groupby("hour").size().reindex(HOURS,
                                                                                                     fill_value=0)
    crit = (win.tier_final == CRITICAL).groupby(win.hour).sum().reindex(HOURS, fill_value=0).cumsum()
    return win, inside, near, crit


def _chart(inside, near, crit, now: int) -> go.Figure:
    fade = np.where(HOURS < now, 1.0, FUTURE_OPACITY)
    fig = go.Figure([
        go.Bar(x=HOURS, y=inside, name="Inside 1 km", marker=dict(color=TIER_COLORS[CRITICAL], opacity=fade,
                                                                  line_width=0)),
        go.Bar(x=HOURS, y=near, name="1–5 km", marker=dict(color=TIER_COLORS[ELEVATED], opacity=fade,
                                                           line_width=0)),
        go.Scatter(x=HOURS, y=crit, name="Critical, cumulative", yaxis="y2", mode="lines",
                   line=dict(color=style.TEXT, width=2, shape="hv")),
    ])
    fig.add_vrect(x0=now - 0.5, x1=23.5, fillcolor=style.BG, opacity=0.35, line_width=0, layer="below")
    fig.add_vline(x=now - 0.5, line_color=style.ACCENT, line_width=1.5)
    fig.add_annotation(x=now - 0.5, y=1, yref="paper", yanchor="bottom", showarrow=False, text="NOW",
                       font=dict(color=style.ACCENT, size=10))
    fig.update_layout(
        barmode="stack", bargap=0.25, height=420, margin=dict(l=0, r=0, t=24, b=0),
        xaxis=dict(title="Hours into the window", dtick=2, range=[-0.5, 23.5]),
        yaxis=dict(title="Objects at closest approach"),
        yaxis2=dict(title="Critical events, cumulative", overlaying="y", side="right", rangemode="tozero",
                    showgrid=False, tickformat=",d", tickmode="auto", nticks=6),
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left", yanchor="bottom", bgcolor="rgba(0,0,0,0)"))
    return fig


def render(ctx):
    shown = ctx.shown
    _md(style.section("04", "Next 24 hours"))
    lo = float(shown.time_to_tca.min()) if len(shown) else DECISION_DAYS
    controls = st.columns(2, gap="large")
    start = controls[0].slider("Window starts (days from the decision CDM to TCA)", lo, lo + 5, lo, 0.25,
                               format="%.2f d")
    now = controls[1].slider("Now (hours into the window)", 0, 24, 24)
    win, inside, near, crit = _hourly(shown, start)
    _md(f'<p class="bm-lede"><b>{len(win):,}</b> encounters reach closest approach between '
        f'<b>{format_ttca(start)}</b> and <b>{format_ttca(start + 1)}</b> after their decision CDM; '
        f'<b>{int(inside.sum()):,}</b> come inside 1 km.</p>')

    stats = st.columns(3, gap="large")
    for col, value, caption, tier, to_come in (
            (stats[0], int(inside[:now].sum()), "Inside 1 km so far", CRITICAL, int(inside[now:].sum())),
            (stats[1], int(near[:now].sum()), "1–5 km so far", ELEVATED, int(near[now:].sum())),
            (stats[2], int(crit.iloc[now - 1]) if now else 0, "Critical events so far", CRITICAL, None)):
        with col:
            _md(stat(f"{value:,}", caption, tier if value else None))
            if to_come:
                _md(style.label(f"+{to_come:,} still to come"))

    st.write("")
    _md(style.label("Objects reaching closest approach, hour by hour"))
    st.plotly_chart(_chart(inside, near, crit, now), width="stretch", config={"displayModeBar": False})
    measured, predicted = int(win.real.sum()), int((~win.real).sum())
    source = ("predicted by the forecaster (outcome unknown)" if not measured else
              "measured" if not predicted else
              f"{measured:,} measured, {predicted:,} predicted by the forecaster (outcome unknown)")
    st.caption(f"Miss distance at TCA: {source}. Faded hours are still to come.")
