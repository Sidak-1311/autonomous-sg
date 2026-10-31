"""Collision-risk dashboard: operations queue, event detail, fleet overview and a 24-hour danger-zone replay.

Run from the repo root: streamlit run dashboard/app.py
Reads only the pipeline's output (DATA CONTRACT in the room): data/processed/events_scored.parquet and
data/processed/cdms.parquet, written by `python -m src.score`. Until they exist it builds a stub with the
same columns from data/test_data.csv (latest CDM as the forecast, no model). DATA_DIR overrides data/.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dashboard import style  # noqa: E402
from dashboard.sections import detail, evaluation, fleet, queue, window  # noqa: E402
from dashboard.sections.common import data_dir, load, read_decisions, stamp  # noqa: E402
from src.tiering import CRITICAL, MISS_CRITICAL_M, RISK_CRITICAL, SAFE, TIERS, format_miss, format_pc  # noqa: E402


st.set_page_config(page_title="Collision risk", page_icon="🛰️", layout="wide")
style.apply()
st.markdown('<div class="hero-eyebrow">Conjunction triage · low Earth orbit</div>', unsafe_allow_html=True)
st.title("Collision-risk evaluator")
st.markdown('<p class="hero-sub">Every close approach in one queue, worst first. Rules set the tier, a forecaster '
            'looks ahead to closest approach, and an analyst signs every decision.</p>', unsafe_allow_html=True)
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
k = st.columns(6)
k[5].metric("Undecided", f"{int((queued & (shown.decision == '')).sum()):,}", border=True,
            help="Queued events without a signed analyst decision.")
k[0].metric("Events", f"{len(shown):,}", border=True)
for col, t in zip(k[1:], reversed(TIERS)):
    col.metric(t.title(), f"{counts.get(t, 0):,}", border=True)
k[4].metric("Needs review", f"{int(review.sum()):,}", border=True,
            help="Too little evidence to forecast: an analyst must review these whatever their tier.")
st.caption(f"{counts.get(CRITICAL, 0):,} events need a maneuver (CRITICAL: Pc ≥ {format_pc(RISK_CRITICAL)} or miss "
           f"< {format_miss(MISS_CRITICAL_M)} at TCA). {int(review.sum()):,} have too little evidence to forecast; "
           "they stay in the queue whatever their tier.")

queue_tab, detail_tab, fleet_tab, window_tab, eval_tab = st.tabs(
    ["Operations queue", "Event detail", "Fleet overview", "Next 24 hours", "Evaluation"])

ctx = SimpleNamespace(events=events, shown=shown, history=history, split=split, queued=queued,
                      decisions=decisions, review=review, counts=counts)
for tab, section in ((queue_tab, queue), (detail_tab, detail), (fleet_tab, fleet), (window_tab, window),
                     (eval_tab, evaluation)):
    with tab:
        section.render(ctx)
