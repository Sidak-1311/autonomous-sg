"""Evaluation: baseline vs model, and the Pc-dilution hypothesis tested against outcomes."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)


def render(ctx):
    shown, history, split = ctx.shown, ctx.history, ctx.split
    queued, decisions = ctx.queued, ctx.decisions
    st.subheader("Baseline vs model (cross-validated on train events)")
    metrics = load_metrics()
    if metrics is None:
        st.info("No data/processed/metrics.json yet. Run `python -m src.score` to measure the baseline.")
    else:
        table = metrics_table(metrics)
        st.dataframe(table, width="stretch", column_config={
            name: st.column_config.NumberColumn(name, format="%.3f") for name in ("baseline", "model")})
        run = metrics.get("model") or metrics.get("baseline") or {}
        st.caption(f"{run.get('n_critical', '?'):,} actual Critical events among {run.get('n_events', '?'):,}. "
                   "Baseline: the latest CDM's risk carried to TCA. 95% CIs bootstrap over events. "
                   "Lower is better for MSE and loss.")
        if "model" not in metrics:
            st.info("No trained model yet, so these are baseline numbers only. "
                    "The forecaster's full training run is waiting for the team's go-ahead.")

    # Pc dilution is shown as a hypothesis tested against outcomes, not as a warning: on the training data
    # it does not predict danger, and the dashboard says so whenever the outcomes show it.
    st.subheader("Pc dilution: does a 'low because poorly tracked' Pc hide danger?")
    flagged = shown.pc_diluted.astype(bool)
    st.caption(f"{int(flagged.sum()):,} of {len(shown):,} events are flagged: their covariance is wider than the "
               "σ = d/√2 at which Pc peaks, so sharper tracking alone could raise Pc by 10× or more.")
    known = shown[shown.tier_actual.notna() & shown.risk_final.notna()]
    if known.empty:
        st.info("No known outcomes for these events, so the flag can't be checked here. "
                "Choose the train split to see whether it predicted anything.")
    else:
        check = (known.assign(flag=np.where(known.pc_diluted.astype(bool), "flagged", "not flagged"),
                              critical=known.tier_actual == CRITICAL, change=known.risk_final - known.risk_latest)
                 .groupby("flag").agg(events=("critical", "size"), ended_critical=("critical", "mean"),
                                      median_change=("change", "median")))
        st.dataframe(check, width="stretch", column_config={
            "ended_critical": st.column_config.NumberColumn("ended Critical", format="percent"),
            "median_change": st.column_config.NumberColumn("median change in log10 Pc to TCA", format="%.1f")})
        if {"flagged", "not flagged"} <= set(check.index):
            f, n = check.ended_critical["flagged"], check.ended_critical["not flagged"]
            if f > n:
                st.success(f"Flagged events ended Critical more often ({f:.1%} vs {n:.1%}).")
            else:
                st.warning(f"Measured failure: flagged events ended Critical no more often than the rest "
                           f"({f:.1%} vs {n:.1%}). As tracking improves, σ usually falls far below d/√2, so the "
                           "peak Pc is never reached. Treat the flag as a tracking-quality note, not a risk signal.")
