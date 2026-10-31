"""Fleet overview: where every event sits, tier mix, Pc distribution, space-weather sensitivity."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import style
from dashboard.sections.common import tier_text
from src.tiering import (CRITICAL, ELEVATED, MISS_CRITICAL_M, MISS_ELEVATED_M, RISK_CRITICAL, RISK_ELEVATED,
                         RISK_FLOOR, SAFE, TIER_COLORS, TIERS, format_miss, format_pc, format_prob)

CHART = dict(margin=dict(l=0, r=0, t=8, b=0), showlegend=False)
BAND_OPACITY = 0.07   # quieter than the shared tier_bands: the points carry the colour here
MISS_RANGE_M = (50, 100_000)


def _md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def stat(value: str, caption: str, tier: str | None = None) -> str:
    return style.hero_stat(value, caption, tier, size="row")


def _bands(fig):
    """Tier regions: Critical where Pc >= 1e-4 or miss < 1 km, Elevated where Pc >= 1e-6 or miss <= 5 km."""
    for (lo, hi), t in (((RISK_CRITICAL, 0), CRITICAL), ((RISK_ELEVATED, RISK_CRITICAL), ELEVATED)):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=TIER_COLORS[t], opacity=BAND_OPACITY, line_width=0, layer="below")
    for (lo, hi), t in (((MISS_RANGE_M[0], MISS_CRITICAL_M), CRITICAL), ((MISS_CRITICAL_M, MISS_ELEVATED_M), ELEVATED)):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=TIER_COLORS[t], opacity=BAND_OPACITY, line_width=0, layer="below")


def _headline(shown: pd.DataFrame):
    """Giant counts: all events, then one per tier (coloured only for the tiers)."""
    counts = shown.tier_final.value_counts()
    cols = st.columns(4, gap="large")
    with cols[0]:
        _md(stat(f"{len(shown):,}", "Events in view"))
    for col, t in zip(cols[1:], (CRITICAL, ELEVATED, SAFE)):
        with col:
            _md(stat(f"{counts.get(t, 0):,}", t.title(), t))


def _sky(shown: pd.DataFrame, at_tca: bool):
    x, y = ("miss_forecast_m", "risk_forecast") if at_tca else ("miss_latest_m", "risk_latest")
    sky = shown.assign(Pc=[format_pc(v) for v in shown[y]], miss=[format_miss(v) for v in shown[x]],
                       p_crit=[format_prob(v) for v in shown.p_critical])
    fig = px.scatter(sky, x=x, y=y, color="tier_final", color_discrete_map=TIER_COLORS,
                     category_orders={"tier_final": TIERS}, hover_name="event_id",
                     hover_data={x: False, y: False, "tier_final": True, "Pc": True, "miss": True,
                                 "p_crit": True, "c_object_type": True},
                     labels={"tier_final": "tier", "p_crit": "P(Critical at TCA)", "c_object_type": "object"})
    for t, size, alpha in ((SAFE, 4, 0.45), (ELEVATED, 5, 0.75), (CRITICAL, 6, 0.95)):
        fig.update_traces(selector=dict(name=t), marker=dict(size=size, opacity=alpha, line_width=0))
    _bands(fig)
    fig.add_annotation(x=1, xref="paper", xanchor="right", y=RISK_FLOOR, yanchor="bottom", showarrow=False,
                       text="FLOOR · Pc < 10⁻³⁰", font=dict(color=style.MUTED, size=10))
    style.miss_axis(style.pc_axis(fig)).update_layout(height=420, **CHART)
    lo_m = min(MISS_RANGE_M[0], float(shown[x].min())) if len(shown) else MISS_RANGE_M[0]
    hi_m = max(MISS_RANGE_M[1], float(shown[x].max())) if len(shown) else MISS_RANGE_M[1]
    fig.update_xaxes(range=[np.log10(max(lo_m, 1.0)), np.log10(hi_m)])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _mix(shown: pd.DataFrame):
    n = shown.c_object_type.value_counts()
    names = {k: f"{str(k).title()} · {v:,}" for k, v in n.items()}   # 'Debris · 1,226'
    mix = (shown.assign(c_object_type=shown.c_object_type.map(names))
           .groupby(["c_object_type", "tier_final"]).size().reset_index(name="events"))
    order = [names[k] for k in n.index[::-1]]  # biggest group on top
    fig = px.bar(mix, x="events", y="c_object_type", color="tier_final", orientation="h",
                 color_discrete_map=TIER_COLORS, category_orders={"tier_final": TIERS, "c_object_type": order},
                 labels={"c_object_type": "", "tier_final": "tier"})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(barnorm="percent", height=340, xaxis_title="Share of events (%)", bargap=0.35, **CHART)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _pc_histogram(shown: pd.DataFrame, y: str, floor: pd.Series):
    above = shown[~floor]
    fig = px.histogram(above, x=y, color="tier_final", color_discrete_map=TIER_COLORS, nbins=40,
                       category_orders={"tier_final": TIERS}, labels={"tier_final": "tier"})
    for v in (RISK_ELEVATED, RISK_CRITICAL):
        fig.add_vline(x=v, line_dash="dot", line_color=style.MUTED, line_width=1)
    lo = int(np.floor(above[y].min())) if len(above) else int(RISK_FLOOR)
    fig.update_traces(marker_line_width=0)
    style.pc_axis(fig, "x", lo=lo - lo % 4, step=4).update_layout(height=340, yaxis_title="Events", bargap=0.08,
                                                                  **CHART)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _storm(shown: pd.DataFrame):
    flips = shown[shown.storm_flips.astype(bool)]
    st.write("")
    _md(style.label("Space-weather sensitivity"))
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        _md(stat(f"{len(flips):,}", "Events that flip to Critical", CRITICAL if len(flips) else None))
    with right:
        st.caption(f"{len(flips):,} of {len(shown):,} events change tier if F10 / AP rise to the data's "
                   "99th percentile (about a minor G1 storm). A sensitivity check, not a storm forecast: "
                   "re-check these if a storm arrives.")
    if not len(flips):
        return
    ranked = (flips.assign(jump=flips.p_critical_storm - flips.p_critical,
                           miss=[format_miss(v) for v in flips.miss_latest_m])
              .sort_values("jump", ascending=False)
              [["event_id", "tier_final", "p_critical", "p_critical_storm", "jump", "miss", "c_object_type"]])
    st.write("")
    st.markdown(style.html_table(ranked, STORM_COLUMNS, max_height=480, rank=True, foot=(
        f"<span>{len(ranked):,} events · biggest jump in P(Critical) first</span>"
        "<span>F10 / AP at the data's 99th percentile</span>")), unsafe_allow_html=True)
    with st.expander("Export or sort as a grid"):
        st.dataframe(ranked.style.map(tier_text, subset=["tier_final"]), hide_index=True, width="stretch",
                     column_config={"event_id": "Event", "tier_final": "Tier", "miss": "Miss now",
                                    "c_object_type": "Object", "jump": None,
                                    "p_critical": st.column_config.NumberColumn("P(Critical) now", format="percent"),
                                    "p_critical_storm": st.column_config.NumberColumn("P(Critical) in a storm",
                                                                                      format="percent")})


def _jump(v, _) -> str:
    """Change in P(Critical) in percentage points."""
    return f'<span class="strong">+{100 * v:.1f} pts</span>' if v == v else '<span class="dim">—</span>'


STORM_COLUMNS = [
    ("event_id", "Event", "l", lambda v, r: f'<span class="strong">{v}</span>'),
    ("tier_final", "Tier now", "l", lambda v, r: style.tier_pill(v)),
    ("p_critical", "P(Critical) now", "l", lambda v, r: style.prob_bar(v)),
    ("p_critical_storm", "P(Critical) in a storm", "l", lambda v, r: style.prob_bar(v)),
    ("jump", "Change", "r", _jump),
    ("miss", "Miss now", "r", None),
    ("c_object_type", "Object", "l", lambda v, r: f'<span class="dim">{str(v).title()}</span>'),
]


def render(ctx):
    shown = ctx.shown
    _md(style.section("03", "Fleet overview"))
    at_tca = st.segmented_control("Position events by", ["Forecast at TCA", "Latest CDM"],
                                  default="Forecast at TCA") != "Latest CDM"
    y = "risk_forecast" if at_tca else "risk_latest"
    floor = shown[y] <= RISK_FLOOR
    crit_share = (shown.tier_final == CRITICAL).mean() if len(shown) else 0
    _md(f'<p class="bm-lede"><b>{crit_share:.0%}</b> of events need a maneuver; '
        f'<b>{floor.mean() if len(floor) else 0:.0%}</b> have a Pc so small it sits at the 10⁻³⁰ floor.</p>')
    _headline(shown)

    st.write("")
    _md(style.label("Where every event sits · Pc against miss distance"))
    _sky(shown, at_tca)
    st.caption(f"Shaded bands are the tier rules: Critical at Pc ≥ {format_pc(RISK_CRITICAL)} or miss < "
               f"{format_miss(MISS_CRITICAL_M)}, Elevated at Pc ≥ {format_pc(RISK_ELEVATED)} or miss ≤ "
               f"{format_miss(MISS_ELEVATED_M)}. Points are coloured by final tier, which the forecaster can raise.")

    left, right = st.columns(2, gap="large")
    with left:
        _md(style.label("Tier mix by object type"))
        _mix(shown)
    with right:
        _md(style.label("Pc above the floor"))
        _pc_histogram(shown, y, floor)
        st.caption(f"{int(floor.sum()):,} events ({floor.mean() if len(floor) else 0:.0%}) sit at the floor, "
                   f"Pc < 10⁻³⁰, and are left out so the rest stays readable. Dotted lines: the Elevated "
                   f"({format_pc(RISK_ELEVATED)}) and Critical ({format_pc(RISK_CRITICAL)}) thresholds.")
    _storm(shown)
