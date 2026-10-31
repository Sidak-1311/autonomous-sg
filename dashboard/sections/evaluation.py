"""Evaluation: baseline vs model as big numbers with their intervals, the alert-fatigue curve, and the two
visible failures (the Pc-dilution flag and the risk regressor), each measured, not asserted."""
from dashboard.sections.common import *  # noqa: F401,F403  (helpers, st, np, pd, px, go, style, brief)


def big_card(value: str, caption: str, detail: str) -> str:
    return (f'<div class="bm-card" style="min-height:13.5rem"><div class="bm-hero"><span class="num" style="font-size:clamp(2.6rem,4.5vw,4.6rem)">'
            f'{value}</span>{style.label(caption)}</div><div style="color:{style.MUTED};margin-top:0.7rem;'
            f'font-size:0.9rem">{detail}</div></div>')


def failure_card(title: str, body: str) -> str:
    return (f'<div class="bm-card" style="border-left:3px solid {style.ACCENT}">{style.label(title)}'
            f'<div style="margin-top:0.5rem;line-height:1.5">{body}</div></div>')


def ci_text(run: dict, key: str, pct: bool) -> str:
    lo_hi = run.get("ci95", {}).get(key)
    if not lo_hi:
        return "no interval"
    return f"95% CI {lo_hi[0]:.1%}–{lo_hi[1]:.1%}" if pct else f"95% CI {lo_hi[0]:.3g}–{lo_hi[1]:.3g}"


def headline(metrics: dict | None):
    if metrics is None:
        st.info("No data/processed/metrics.json yet. Run `python -m src.score` to measure the baseline.")
        return
    cards = []
    if depth := metrics.get("queue_depth"):
        lo, hi = depth["ci95"]["reduction"]
        cards.append(big_card(f"{depth['reduction']:.0%}", "Fewer reviews than today's ranking, at 95% recall",
                              f"95% CI {lo:.0%}–{hi:.0%} · top {depth['model']:,} vs {depth['baseline']:,} events, "
                              "all train events, out of fold. The interval is wide; quote it with the number."))
    base, model = metrics.get("baseline", {}), metrics.get("model")
    for key, caption in (("recall_critical", "Critical recall"), ("precision_critical", "Critical precision")):
        if key in base:
            run = model or base
            who = "model" if model else "baseline"
            cards.append(big_card(f"{run[key]:.1%}", f"{caption} · {who}",
                                  (f"baseline {base[key]:.1%} · " if model else "") + ci_text(run, key, True)))
    for col, html in zip(st.columns(len(cards)) if cards else [], cards):
        col.markdown(html, unsafe_allow_html=True)
    if "model" not in metrics:
        st.info("No trained model yet, so these are baseline numbers only. "
                "The forecaster's full training run is waiting for the team's go-ahead.")
    table = metrics_table(metrics)
    run = metrics.get("model") or base
    st.write("")
    st.markdown(style.label("All metrics with 95% intervals"), unsafe_allow_html=True)
    view = table.reset_index().rename(columns=VIEW_NAMES)   # html_table needs identifier column names
    st.markdown(style.html_table(view, metric_columns(view), foot=(
        f"<span>{run.get('n_critical', '?'):,} actual Critical events among {run.get('n_events', '?'):,} · "
        "out of fold</span><span>Baseline = latest CDM carried to TCA · lower is better for MSE and loss</span>")),
        unsafe_allow_html=True)
    with st.expander("Export or sort as a grid"):
        st.dataframe(table, width="stretch", column_config={
            name: st.column_config.NumberColumn(name, format="%.3f") for name in ("baseline", "model")})


PCT_METRICS = {"Critical recall", "Critical precision"}
VIEW_NAMES = {"baseline 95% CI": "baseline_ci", "model 95% CI": "model_ci", "model better?": "model_better"}


def _metric_value(v, r) -> str:
    if v != v:
        return '<span class="dim">—</span>'
    return f"{v:.1%}" if r.metric in PCT_METRICS else f"{v:.3g}"


def _metric_ci(v, r) -> str:
    """metrics_table writes '[lo, hi]'; percentages get the same 1-dp % as their value."""
    try:
        lo, hi = (float(x) for x in str(v).strip("[]").split(","))
    except ValueError:
        return f'<span class="dim">{v}</span>'
    text = f"{lo:.1%} – {hi:.1%}" if r.metric in PCT_METRICS else f"{lo:.3g} – {hi:.3g}"
    return f'<span class="dim">{text}</span>'


def _verdict(v, _) -> str:
    """yes / no / same as chips: 'yes' lit, the others quiet."""
    return style.chip(v, on=v == "yes") if isinstance(v, str) else '<span class="dim">—</span>'


def metric_columns(table: pd.DataFrame) -> list[tuple]:
    """Columns for whichever scorers metrics.json has (baseline only before the model is trained)."""
    cols = [("metric", "Metric", "l", lambda v, r: f'<span class="strong">{v}</span>')]
    for name in ("baseline", "model"):
        if name in table:
            cols += [(name, name.title(), "r", _metric_value),
                     (f"{name}_ci", "95% CI", "r", _metric_ci)]
    if "model_better" in table:
        cols.append(("model_better", "Model better?", "l", _verdict))
    return cols


def fatigue(shown: pd.DataFrame):
    """How many truly Critical events an analyst catches working the queue top-down (train outcomes only),
    against today's ranking by the latest CDM alone."""
    known = shown[shown.tier_actual.notna()]
    if not (known.tier_actual == CRITICAL).any():
        st.info("Alert fatigue needs known outcomes: choose the train split.")
        return
    in_sample = known.forecast_in_sample.astype(bool).any()  # the model was fit on these very events
    rankings = {"baseline: latest CDM": (known.assign(_r=known.tier_now.map(TIER_RANK))
                                         .sort_values(["_r", "risk_latest"], ascending=False), style.MUTED, "dash"),
                "this queue" + (" (in-sample)" if in_sample else ""): (queue_order(known), style.TEXT, "solid")}
    curve, k95 = go.Figure(), {}
    for name, (ranked, color, dash) in rankings.items():
        hits = (ranked.tier_actual == CRITICAL).cumsum().to_numpy()
        caught = hits / hits[-1]
        k95[name] = int(np.argmax(caught >= 0.95)) + 1
        curve.add_trace(go.Scatter(x=np.arange(1, len(hits) + 1), y=caught, name=name,
                                   line=dict(color=color, dash=dash, width=2)))
    curve.add_hline(y=0.95, line_dash="dot", line_color=style.LINE)
    curve.update_layout(xaxis_title="events reviewed, worst first", yaxis_title="actual Criticals caught",
                        yaxis_tickformat=".0%", height=380)
    st.plotly_chart(curve, width="stretch", config={"displayModeBar": False})
    n, (base, mine) = len(known), k95.values()
    st.caption(f"To catch 95% of the {int((known.tier_actual == CRITICAL).sum()):,} events that really ended "
               f"Critical: top {mine:,} of {n:,} ({mine / n:.0%}) with this queue, top {base:,} ({base / n:.0%}) "
               "with the baseline ranking.")
    if in_sample:
        st.warning("In-sample: the model was trained on these events, so the queue's curve is optimistic. "
                   "Use the out-of-fold numbers above for claims.")


def dilution_failure(shown: pd.DataFrame) -> str:
    """The Pc-dilution flag, checked against outcomes every time the page renders."""
    flagged = shown.pc_diluted.astype(bool)
    st.caption(f"{int(flagged.sum()):,} of {len(shown):,} events are flagged: their covariance is wider than the "
               "σ = d/√2 at which Pc peaks.")
    known = shown[shown.tier_actual.notna() & shown.risk_final.notna()]
    if known.empty:
        st.info("No known outcomes for these events, so the flag can't be checked here. "
                "Choose the train split to see whether it predicted anything.")
        return ""
    check = (known.assign(flag=np.where(known.pc_diluted.astype(bool), "flagged", "not flagged"),
                          critical=known.tier_actual == CRITICAL, change=known.risk_final - known.risk_latest)
             .groupby("flag").agg(events=("critical", "size"), ended_critical=("critical", "mean"),
                                  median_change=("change", "median")))
    st.markdown(style.html_table(check.reset_index(), [
        ("flag", "Pc-dilution flag", "l", lambda v, r: style.chip(v, on=v == "flagged")),
        ("events", "Events", "r", lambda v, r: f"{v:,}"),
        ("ended_critical", "Ended Critical", "r", lambda v, r: f'<span class="strong">{v:.1%}</span>'),
        ("median_change", "Median change in log10 Pc to TCA", "r", lambda v, r: f"{v:+.1f}"),
    ]), unsafe_allow_html=True)
    with st.expander("Flagged vs not flagged as a grid"):
        st.dataframe(check, width="stretch", column_config={
            "ended_critical": st.column_config.NumberColumn("ended Critical", format="percent"),
            "median_change": st.column_config.NumberColumn("median change in log10 Pc to TCA", format="%.1f")})
    if not {"flagged", "not flagged"} <= set(check.index):
        return ""
    f, n = check.ended_critical["flagged"], check.ended_critical["not flagged"]
    if f > n:
        return f"Not a failure here: flagged events ended Critical more often ({f:.1%} vs {n:.1%})."
    return (f"Measured failure: flagged events ended Critical no more often than the rest ({f:.1%} vs {n:.1%}). "
            "As tracking improves, σ usually falls far below d/√2, so the peak Pc is never reached. "
            "The flag is a tracking-quality note, not a risk signal.")


def regressor_failure(metrics: dict | None) -> str:
    base, model = (metrics or {}).get("baseline", {}), (metrics or {}).get("model", {})
    if not model or "mse_hr" not in base or "mse_hr" not in model:
        return ""
    if np.isclose(model["mse_hr"], base["mse_hr"]):
        return (f"Measured failure: the risk forecaster did not beat carrying the latest Pc forward out of fold "
                f"(MSE on high-risk events {base['mse_hr']:.2f} either way), so Pc at TCA is the latest Pc. "
                "P(Critical) and the miss-distance forecast still come from the model.")
    return ""


def lede(metrics: dict | None) -> str:
    depth = (metrics or {}).get("queue_depth")
    gain = (f"<b>{depth['reduction']:.0%} fewer reviews</b> than today's ranking for the same 95% recall, "
            if depth else "Measured against the latest-CDM baseline, out of fold; ")
    return f'<p class="bm-lede">{gain}and two ideas that failed when measured, with the reason each failed.</p>'


def render(ctx):
    st.markdown(style.section("05", "Evaluation"), unsafe_allow_html=True)
    metrics = load_metrics()
    st.markdown(lede(metrics), unsafe_allow_html=True)
    headline(metrics)

    st.markdown(style.label("Alert fatigue · actual Criticals caught, working the queue top-down"),
                unsafe_allow_html=True)
    fatigue(ctx.shown)

    st.markdown(style.label("Where it fails"), unsafe_allow_html=True)
    dilution = dilution_failure(ctx.shown)
    failures = [("Pc-dilution flag", dilution), ("Risk regressor", regressor_failure(metrics))]
    failures = [(t, b) for t, b in failures if b]
    for col, (title, body) in zip(st.columns(len(failures)) if failures else [], failures):
        col.markdown(failure_card(title, body), unsafe_allow_html=True)
