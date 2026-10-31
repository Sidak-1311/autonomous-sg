"""Collision-risk dashboard: operations queue, event detail, fleet overview and a 24-hour danger-zone replay.

Run from the repo root: streamlit run dashboard/app.py
Reads only the pipeline's output (DATA CONTRACT in the room): data/processed/events_scored.parquet and
data/processed/cdms.parquet, written by `python -m src.score`. Until they exist it builds a stub with the
same columns from data/test_data.csv (latest CDM as the forecast, no model). DATA_DIR overrides data/.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dashboard import hero, style  # noqa: E402
from dashboard.sections import detail, evaluation, fleet, queue, window  # noqa: E402
from dashboard.sections.common import data_dir, load, processed_dir, read_decisions, stamp  # noqa: E402
from src.tiering import CRITICAL, SAFE, TIERS, format_miss, format_pc, format_ttca  # noqa: E402

HERO_HEIGHT = 480  # the scene is a flat disc (radial offsets are tiny), so taller is just empty space

st.set_page_config(page_title="Triage-TCA", page_icon=":material/radar:", layout="wide")
style.apply()
events, cdms, source = load(str(data_dir()), stamp())

with st.sidebar:
    st.caption(f"Data: {source}")
    splits = sorted(events.split.unique())
    split = st.selectbox("Events", splits, index=splits.index("test") if "test" in splits else 0)
    tiers = st.multiselect("Tier (final)", TIERS, default=TIERS)
    types = sorted(events.c_object_type.dropna().unique())
    obj = st.multiselect("Object type", types, default=types)
shown = events[(events.split == split) & events.tier_final.isin(tiers) & events.c_object_type.isin(obj)]
history = cdms[cdms.split == split]

counts = shown.tier_final.value_counts()
review = shown.needs_review.astype(bool)
decisions = read_decisions()
latest_decision = (decisions[decisions.split == split].sort_values("at_utc")
                   .groupby("event_id").decision.last())
shown = shown.assign(decision=shown.event_id.map(latest_decision).fillna(""))
queued = (shown.tier_final != SAFE) | review  # a refusal is never auto-cleared

# ---- top bar: wordmark and the state of the data ----
scored = processed_dir() / "events_scored.parquet"
scored_at = (datetime.fromtimestamp(scored.stat().st_mtime, timezone.utc).strftime("%d %b %H:%M UTC")
             if scored.exists() else "not yet")
model = ", ".join(sorted(s for s in events.forecast_source.dropna().unique() if s != "withheld")) or "none"
st.markdown(style.top_bar([("Split", split), ("Events", f"{len(shown):,}"), ("Forecast", model),
                           ("Scored", scored_at)]), unsafe_allow_html=True)

# ---- hero: the claim on the left, every encounter around the primary on the right ----
left, right = st.columns([5, 7], gap="large", vertical_alignment="center")
with left:
    st.markdown('<div class="bm-display">Every close approach. <em>Worst first.</em></div>'
                f'<p class="bm-lede"><b>{len(shown):,} conjunctions</b> screened at the 2-day decision point. '
                'Rules set the tier, a calibrated forecaster looks ahead to closest approach, and an analyst '
                'signs every call.</p>', unsafe_allow_html=True)
with right:
    if len(shown):
        components.html(hero.hero_html(hero.hero_figure(shown, history, split=split), height=HERO_HEIGHT),
                        height=HERO_HEIGHT)
        st.markdown(style.label("Each point: one encounter relative to the primary (RTN frame), distance on a log "
                                "scale. Spheres at 1 km and 5 km. Drag to rotate."), unsafe_allow_html=True)

# ---- KPI row: flat giant numbers; order matters to the tests (Events … Undecided) ----
with st.container(key="kpis"):
    k = st.columns(6)
    with k[0]:
        st.metric("Events", f"{len(shown):,}")
    for col, t in zip(k[1:4], reversed(TIERS)):
        with col, st.container(key=f"kpi-{t.lower()}"):
            st.metric(t.title(), f"{counts.get(t, 0):,}")
    with k[4]:
        st.metric("Needs review", f"{int(review.sum()):,}",
                  help="Too little evidence to forecast: an analyst must review these whatever their tier.")
    with k[5]:
        st.metric("Undecided", f"{int((queued & (shown.decision == '')).sum()):,}",
                  help="Queued events without a signed analyst decision.")

# ---- ticker: the worst events, scrolling ----
worst = shown[shown.tier_final == CRITICAL].sort_values(["p_critical", "risk_forecast"], ascending=False).head(14)
st.markdown(style.marquee([
    f"<b>EVT {r.event_id}</b> · Pc {format_pc(r.risk_latest)} · {format_miss(r.miss_latest_m)} · "
    f"T−{format_ttca(r.time_to_tca)}" for r in worst.itertuples()]), unsafe_allow_html=True)

queue_tab, detail_tab, fleet_tab, window_tab, eval_tab = st.tabs(
    ["01  Operations queue", "02  Event detail", "03  Fleet overview", "04  Next 24 hours", "05  Evaluation"])

ctx = SimpleNamespace(events=events, shown=shown, history=history, split=split, queued=queued,
                      decisions=decisions, review=review, counts=counts)
for tab, section in ((queue_tab, queue), (detail_tab, detail), (fleet_tab, fleet), (window_tab, window),
                     (eval_tab, evaluation)):
    with tab:
        section.render(ctx)
