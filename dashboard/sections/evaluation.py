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
    # Never set two numbers side by side unless they were measured on the same events.
    comparable = not model or base.get("n_events") == model.get("n_events")
    for key, caption in (("recall_critical", "Critical recall"), ("precision_critical", "Critical precision")):
        if key in base:
            run = model or base
            who = "model" if model else "baseline"
            cards.append(big_card(f"{run[key]:.1%}", f"{caption} · {who}",
                                  (f"baseline {base[key]:.1%} · " if model and comparable else "")
                                  + ci_text(run, key, True)))
    for col, html in zip(st.columns(len(cards)) if cards else [], cards):
        col.markdown(html, unsafe_allow_html=True)
    if not comparable:
        st.warning(f"Not comparable yet: the baseline is scored on {base.get('n_events', 0):,} events and the model "
                   f"on {model.get('n_events', 0):,}. No 'better or worse' verdict until both use the same events.")
    if "model" not in metrics:
        st.info("No trained model yet, so these are baseline numbers only. "
                "The forecaster's full training run is waiting for the team's go-ahead.")
    table = metrics_table(metrics)
    run = metrics.get("model") or base
    st.write("")
    st.markdown(style.label("All metrics with 95% intervals"), unsafe_allow_html=True)
    if not comparable:
        table = table.drop(columns=["model better?"], errors="ignore")
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
    with st.expander("How the Pc-dilution flag was measured"):
        st.markdown(style.html_table(check.reset_index(), [
            ("flag", "Pc-dilution flag", "l", lambda v, r: style.chip(v, on=v == "flagged")),
            ("events", "Events", "r", lambda v, r: f"{v:,}"),
            ("ended_critical", "Ended Critical", "r", lambda v, r: f'<span class="strong">{v:.1%}</span>'),
            ("median_change", "Median change in log10 Pc to TCA", "r", lambda v, r: f"{v:+.1f}"),
        ]), unsafe_allow_html=True)
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
    if test := holdout(metrics):   # the organisers' own check, so it comes first
        held_out(test, load_error_analysis(HOLDOUT_ANALYSIS), "Official holdout · the headline",
                 f"The organisers' check: 20% of train events held out by event_id (<b>{test['n']:,} events</b>, "
                 f"<b>{test['model']['n_critical']}</b> truly Critical); the model, its calibration and its alert "
                 "threshold were fit on the other 80% only.")

    st.markdown(style.label("Alert fatigue · actual Criticals caught, working the queue top-down"),
                unsafe_allow_html=True)
    fatigue(ctx.shown)

    st.markdown(style.label("Where it fails"), unsafe_allow_html=True)
    dilution = dilution_failure(ctx.shown)
    failures = [("Pc-dilution flag", dilution), ("Risk regressor", regressor_failure(metrics))]
    failures = [(t, b) for t, b in failures if b]
    for col, (title, body) in zip(st.columns(len(failures)) if failures else [], failures):
        col.markdown(failure_card(title, body), unsafe_allow_html=True)
    misses(load_error_analysis())
    lockbox((metrics or {}).get("test_lockbox"), load_error_analysis(LOCKBOX_ANALYSIS))


ERROR_ANALYSIS = ROOT / "models" / "error_analysis.json"   # written by `python -m src.error_analysis`
LOCKBOX_ANALYSIS = ROOT / "models" / "error_analysis_lockbox.json"   # `python -m src.error_analysis --lockbox`


def load_error_analysis(path=None) -> dict | None:
    path = path or ERROR_ANALYSIS
    return json.loads(path.read_text()) if path.exists() else None


def confusion_grid(conf: dict) -> pd.DataFrame:
    return pd.DataFrame(conf["matrix"], index=[f"actual {t}" for t in conf["labels"]],
                        columns=[f"predicted {t}" for t in conf["labels"]])


def matrix_html(labels: list[str], counts: list[list[int]], cols_word: str = "Predicted") -> str:
    """A 3x3 confusion matrix as a styled table: tier pills down the side, the diagonal (correct) in bold."""
    df = pd.DataFrame(counts, columns=[f"c{j}" for j in range(len(labels))]).assign(actual=labels)
    cols = [("actual", f"Actual ↓ · {cols_word} →", "l", lambda v, r: style.pill(v))]
    for j, t in enumerate(labels):
        cols.append((f"c{j}", t.title(), "r",
                     lambda v, r, t=t: f'<span class="strong">{v:,}</span>' if r.actual == t else f"{v:,}"))
    return style.html_table(df, cols)


def as_grid(df: pd.DataFrame, **kw):
    """The same numbers as a sortable, exportable grid, out of the way."""
    with st.expander("Export or sort as a grid"):
        st.dataframe(df, width="stretch", **kw)


HOLDOUT_METRICS = ROOT / "models" / "holdout_metrics.json"       # 80/20 split by event_id (src.train)
HOLDOUT_ANALYSIS = ROOT / "models" / "error_analysis_holdout.json"


def holdout(metrics: dict | None) -> dict | None:
    """The organisers' 80/20 held-out evaluation: scores + CIs from holdout_metrics.json, matrices from
    metrics.json['holdout']. None unless both are there."""
    cm = (metrics or {}).get("holdout")
    if not cm or not HOLDOUT_METRICS.exists():
        return None
    hm = json.loads(HOLDOUT_METRICS.read_text())
    return {"n": cm["n"], "baseline": {**hm["baseline"], "confusion": cm["baseline"]},
            "model": {**hm["model"], "confusion": cm["model"]}}


def lockbox(test: dict | None, ea: dict | None):
    """The hidden ESA Kelvins test labels: the frozen model scored once, baseline on the same 2,167 events."""
    if test:
        held_out(test, ea, "Hidden test set · scored once",
                 f"The ESA Kelvins test outcomes, published after the competition: <b>{test['n']:,} events</b>, "
                 f"<b>{test['model']['n_critical']}</b> truly Critical. Scored once on the frozen model; never used "
                 "to train, tune or choose.")


def held_out(test: dict, ea: dict | None, title: str, lede: str):
    """One held-out set: recall / precision vs the baseline on the same events, both confusion matrices, and
    every Critical the model missed there."""
    st.markdown(f'<div style="margin-top:2.6rem">{style.label(title)}</div>', unsafe_allow_html=True)
    base, model = test["baseline"], test["model"]
    st.markdown(f'<p class="bm-lede">{lede}</p>', unsafe_allow_html=True)
    cards = [big_card(f"{model[k]:.1%}", f"{name} · model", f"baseline {base[k]:.1%} · {ci_text(model, k, True)}")
             for k, name in (("recall_critical", "Critical recall"), ("precision_critical", "Critical precision"))]
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)
    left, right = st.columns(2)
    for col, (who, run) in zip((left, right), (("Baseline: latest CDM", base), ("Model", model))):
        if "confusion" in run:
            col.markdown(style.label(who), unsafe_allow_html=True)
            col.markdown(matrix_html(run["confusion"]["labels"], run["confusion"]["matrix"]), unsafe_allow_html=True)
            with col:
                as_grid(confusion_grid(run["confusion"]))
    if ea and ea.get("missed_examples"):
        missed = pd.DataFrame(ea["missed_examples"]).assign(
            pc=lambda d: [f"{format_pc(a)} → {format_pc(b)}" for a, b in zip(d.risk_decision, d.risk_final)],
            miss=lambda d: [f"{format_miss(a)} → {format_miss(b)}" for a, b in zip(d.miss_decision_m, d.miss_final_m)],
            cause=lambda d: d.cause.str.replace("_", " "))
        st.markdown(style.label(f"Every Critical the model missed here ({len(missed)}) · at the decision point → at TCA"),
                    unsafe_allow_html=True)
        st.markdown(style.html_table(missed, [
            ("event_id", "Event", "r", lambda v, r: f'<span class="strong">{v}</span>'),
            ("tier_issued", "Tier issued", "l", lambda v, r: style.pill(v)),
            ("cause", "Cause", "l", None),
            ("pc", "Pc", "r", None),
            ("miss", "Miss distance", "r", None)]), unsafe_allow_html=True)
        as_grid(missed[["event_id", "tier_issued", "cause", "pc", "miss"]], hide_index=True)


def misses(ea: dict | None):
    """The 3x3 confusion matrix and why the missed Critical events were missed (models/error_analysis.json)."""
    if not ea:
        return
    cm = ea["confusion_matrix"]
    st.markdown(f'<div style="margin-top:2rem">{style.label("Confusion matrix · rows actual final tier, columns tier issued at the decision point")}</div>',
                unsafe_allow_html=True)
    labels = cm["rows_actual_cols_issued"]
    st.markdown(matrix_html(labels, cm["counts"], "Issued"), unsafe_allow_html=True)
    as_grid(pd.DataFrame(cm["counts"], index=[f"actual {t}" for t in labels], columns=[f"issued {t}" for t in labels]))
    mc = ea["missed_critical"]
    st.caption(f"{ea['scope']}: {ea['n_critical']:,} actual Critical among {ea['n_events']:,}; recall "
               f"{ea['recall_critical']:.1%} ({ea['recall_critical_final_cdm_under_2d']:.1%} when the final CDM comes "
               f"after the decision point). {mc['n']} missed, {ea['silent_missed_critical']} of them silently "
               "(the rest were sent to an analyst as refusals).")
    causes = pd.DataFrame([{"why": k.replace("_", " "), "missed": v["n"], "meaning": v["meaning"]}
                           for k, v in mc["by_cause"].items() if v["n"]]).sort_values("missed", ascending=False)
    st.markdown(style.label("Why the missed Critical events were missed"), unsafe_allow_html=True)
    st.markdown(style.html_table(causes, [
        ("why", "Cause", "l", lambda v, r: f'<span class="strong">{v}</span>'),
        ("missed", "Missed", "r", lambda v, r: f"{v:,}"),
        ("meaning", "Meaning", "l", None)]), unsafe_allow_html=True)
    as_grid(causes, hide_index=True)
    if trade := ea.get("miss_watch_tradeoff"):
        best = trade[0]
        st.caption(f"Catching the miss-driven ones costs alerts: a watch rule for forecast miss < "
                   f"{best['miss_below_m']:,.0f} m recovers {best['recovered_critical']} Criticals for "
                   f"{best['extra_alerts']} extra alerts ({best['alerts_per_recovered']:.0f} per catch).")
