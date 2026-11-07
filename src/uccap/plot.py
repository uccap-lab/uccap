"""Plots for capability studies, risk-calibrated decisions, UC-Cap scores, batch results and monitoring.

Requires matplotlib (``pip install "uccap[plot]"``). Every panel function draws on a
given matplotlib Axes (or creates one) and returns it, so panels can be combined freely.
The figure functions group the panels by question and return a complete Figure:

=====================  ==================================================  =====================
figure                 panels                                              question
=====================  ==================================================  =====================
outlier_figure         outlier_chart, outlier_box                          Points to investigate?
stability_figure       individuals_chart, moving_range_chart               Process stable?
capability_figure      capability_histogram, probability_plot              Capability and fit
approval_figure        decision_plot, required_plot                        Approve at this n?
uccap_figure           uccap_contributions, uccap_gauge                    UC-Cap risk score
study_overview         histogram, probability plot, I-MR (and decision)    Everything at a glance
monitor_figure         monitor_means, monitor_chart                        Still acceptable now?
monitor_overview       final pi, Ppk against final pi, first intervention  Which dimensions to act on?
=====================  ==================================================  =====================

The outlier fences (screen before the calculation) and the control limits of the I-MR
charts (stability over time) answer different questions and are drawn on separate panels.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

# Colours: one data hue, one fitted-curve hue, neutral ink for specification lines,
# and status colours that always come with a text label.
DATA = "#2a78d6"
FIT = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8980"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
WARN = "#d39b00"
BAD = "#d03b3b"
FILL = "#2a78d6"

D2 = 1.128  # d2 for moving ranges of two
D4 = 3.267  # D4 for moving ranges of two


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('plotting needs matplotlib: pip install "uccap[plot]"') from exc
    return plt


def _ax(ax, figsize=(6.4, 4.0)):
    if ax is None:
        _, ax = _plt().subplots(figsize=figsize, layout="constrained")
    _style(ax)
    return ax


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, color=AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=10, color=INK, fontweight="bold")


def _top_legend(ax, note=None):
    """Legend (and an optional note) above the plot area, right-aligned, so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    if note:
        handles = [_plt().Line2D([], [], linestyle="none")] + handles
        labels = [note] + labels
    if handles:
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1, 1), ncol=len(handles), fontsize=8,
                  labelcolor=INK_2, frameon=False, borderaxespad=0.1, handlelength=0.8, handletextpad=0.4,
                  columnspacing=1.2)


def _value_labels(ax, t, y, mask, decimals=None):
    """Write each value next to its point (the raw data): above a local peak, below a local dip."""
    ts, ys = t[mask], y[mask]
    if decimals is None:  # as many decimals as the data have, at most 3
        decimals = next((d for d in range(4) if np.allclose(ys, np.round(ys, d), rtol=0, atol=1e-9)), 3)
    for i, (ti, yi) in enumerate(zip(ts, ys)):
        nb = [ys[j] for j in (i - 1, i + 1) if 0 <= j < ys.size]
        up = not nb or yi >= np.mean(nb)
        ax.annotate(f"{yi:.{decimals}f}", (ti, yi), xytext=(0, 5 if up else -5), textcoords="offset points",
                    ha="center", va="bottom" if up else "top", fontsize=6, color=INK_2, zorder=5)


def _spec_lines(ax, spec, vertical=True):
    """Draw LSL / Target / USL as labelled lines."""
    line = ax.axvline if vertical else ax.axhline
    for value, name, style in ((spec.lsl, "LSL", "--"), (spec.target, "Target", ":"), (spec.usl, "USL", "--")):
        if value is None:
            continue
        line(value, color=INK_2, linestyle=style, linewidth=1.1, zorder=3)
        if vertical:
            ax.annotate(name, (value, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                        textcoords="offset points", ha="left", va="top", fontsize=8, color=INK_2)
        else:
            ax.annotate(name, (1, value), xycoords=("axes fraction", "data"), xytext=(2, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_2)


def _model(study):
    """The distribution the capability indices were computed from."""
    if study.method == "percentile" and study.fit is not None and study.percentile_method != "empirical":
        return study.fit.dist, study.fit.name
    if study.method == "percentile":
        return None, "empirical"
    return stats.norm(study.mean, study.sigma_overall), "normal"


def capability_histogram(study, ax=None, bins="auto"):
    """Histogram of the data with the specification limits and the fitted distribution."""
    ax = _ax(ax)
    x = study.data
    ax.hist(x, bins=bins, density=True, color=DATA, alpha=0.85, edgecolor="white", linewidth=1.0, zorder=2)
    dist, name = _model(study)
    lo = min([x.min()] + [v for v in (study.spec.lsl,) if v is not None])
    hi = max([x.max()] + [v for v in (study.spec.usl,) if v is not None])
    pad = 0.08 * (hi - lo)
    if dist is not None:
        grid = np.linspace(lo - pad, hi + pad, 400)
        ax.plot(grid, dist.pdf(grid), color=FIT, linewidth=2, zorder=4, label=f"fitted {name}")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.52), labelcolor=INK_2,
                  facecolor="white", edgecolor="none", framealpha=0.9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    _spec_lines(ax, study.spec)
    ax.set_yticks([])
    ax.set_ylabel("density")
    ax.set_xlabel("measurement")
    idx = study.indices
    if study.method == "normal":
        text = [f"Pp  {idx.get('Pp', math.nan):.2f}", f"Ppk {idx['Ppk']:.2f}",
                f"Cp  {idx.get('Cp', math.nan):.2f}", f"Cpk {idx.get('Cpk', math.nan):.2f}"]
    else:
        text = [f"CNp  {idx.get('CNp', math.nan):.2f}", f"CNpk {idx['CNpk']:.2f}"]
    text = [t for t in text if not t.endswith("nan")] + [f"n = {study.n}"]
    ax.text(0.99, 0.89, "\n".join(text), transform=ax.transAxes, ha="right", va="top", fontsize=8,
            family="monospace", color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    _title(ax, "Capability histogram")
    return ax


def probability_plot(study, ax=None):
    """Quantile plot of the data against the distribution used for the indices."""
    ax = _ax(ax)
    x = np.sort(study.data)
    n = x.size
    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)  # Blom plotting positions
    dist, name = _model(study)
    if dist is None:
        dist, name = stats.norm(study.mean, study.sigma_overall), "normal"
    q = dist.ppf(p)
    ax.plot(q, x, "o", markersize=4.5, color=DATA, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ends = np.array([min(q.min(), x.min()), max(q.max(), x.max())])
    ax.plot(ends, ends, color=FIT, linewidth=1.5, zorder=2)
    ax.set_xlabel(f"{name} quantile")
    ax.set_ylabel("observed value")
    nr = study.normality
    _title(ax, f"Probability plot ({name})")
    ax.text(0.02, 0.97, f"{nr.test} p = {nr.p_value:.3f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=INK_2)
    return ax


def _series(study):
    """Values in time order with NaN where a point is missing or excluded, plus exclusion mask."""
    raw = np.asarray(study.raw_data if study.raw_data is not None else study.data, float)
    excluded = np.zeros(raw.size, bool) if study.excluded_mask is None else np.asarray(study.excluded_mask)
    return raw, excluded


def individuals_chart(study, ax=None, show_data=False):
    """Individuals (I) chart: values in time order with mean and 3-sigma-within limits.

    ``show_data=True`` writes each value next to its point (at most 3 decimals).
    """
    ax = _ax(ax, (6.4, 3.0))
    raw, excluded = _series(study)
    t = np.arange(1, raw.size + 1)
    used = np.isfinite(raw) & ~excluded
    centre, s = study.mean, study.sigma_within
    lcl, ucl = centre - 3 * s, centre + 3 * s
    ax.plot(t, np.where(used, raw, np.nan), color=DATA, linewidth=1.2, zorder=2)
    out = used & ((raw > ucl) | (raw < lcl))
    ax.plot(t[used & ~out], raw[used & ~out], "o", markersize=4, color=DATA, markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)
    if out.any():
        ax.plot(t[out], raw[out], "s", markersize=5.5, color=BAD, markeredgecolor="white", zorder=4,
                label="outside limits")
    if excluded.any():
        ax.plot(t[excluded], raw[excluded], "o", markersize=5, markerfacecolor="none", markeredgecolor=MUTED,
                zorder=3, label="excluded")
    for y, label in ((ucl, "UCL"), (centre, "mean"), (lcl, "LCL")):
        ax.axhline(y, color=INK_2 if label == "mean" else MUTED, linewidth=1, linestyle="-" if label == "mean" else "--")
        ax.annotate(f"{label} {y:.4g}", (1, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    if show_data:
        _value_labels(ax, t, raw, np.isfinite(raw))
    _top_legend(ax)
    ax.set_xlabel("observation (time order)")
    ax.set_ylabel("value")
    _title(ax, "Individuals chart")
    return ax


def moving_range_chart(study, ax=None, show_data=False):
    """Moving range (MR) chart; ranges never bridge missing or excluded points.

    ``show_data=True`` writes each moving range next to its point (at most 3 decimals).
    """
    ax = _ax(ax, (6.4, 3.0))
    raw, excluded = _series(study)
    used = np.where(np.isfinite(raw) & ~excluded, raw, np.nan)
    mr = np.abs(np.diff(used))
    t = np.arange(2, raw.size + 1)
    mr_bar = D2 * study.sigma_within
    ucl = D4 * mr_bar
    ok = np.isfinite(mr)
    out = ok & (mr > ucl)
    ax.plot(t, mr, color=DATA, linewidth=1.2, zorder=2)
    ax.plot(t[ok & ~out], mr[ok & ~out], "o", markersize=4, color=DATA, markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)
    if out.any():
        ax.plot(t[out], mr[out], "s", markersize=5.5, color=BAD, markeredgecolor="white", zorder=4)
    for y, label in ((ucl, "UCL"), (mr_bar, "MR-bar")):
        ax.axhline(y, color=MUTED if label == "UCL" else INK_2, linewidth=1, linestyle="--" if label == "UCL" else "-")
        ax.annotate(f"{label} {y:.3g}", (1, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    if show_data:
        _value_labels(ax, t, mr, ok)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("observation (time order)")
    ax.set_ylabel("moving range")
    _title(ax, "Moving range chart")
    return ax


def decision_plot(decision, study=None, ax=None, n_boot=2000, rng=0):
    """Uncertainty in the capability estimate against the requirement C0.

    Shows the uncertainty in the true capability (normal with the analytic standard
    error, or the bootstrap distribution when the decision used the bootstrap and the
    study is given), the requirement C0, the estimate, and the failure probability as a
    shaded area. The estimate needed for approval is listed in the box (analytic
    method) and drawn against the sample size by :func:`required_plot`.
    """
    ax = _ax(ax)
    c, se, c0 = decision.cpk_hat, decision.se, decision.c0
    name = decision.index_name
    if decision.method == "bootstrap" and study is not None:
        from .risk.bootstrap import bootstrap_cpk
        draws = bootstrap_cpk(study, n_boot, rng)
        draws = draws[np.isfinite(draws)]
        sample = draws  # the failure probability is the share of resamples below C0
        lo, hi = min(sample.min(), c0) - 0.1, max(sample.max(), decision.margin) + 0.1
        bins = np.linspace(lo, hi, 50)
        w = 1.0 / (sample.size * (bins[1] - bins[0]))  # one density scale for both parts
        ax.hist(sample[sample >= c0], bins=bins, weights=np.full((sample >= c0).sum(), w), color=FILL,
                alpha=0.35, edgecolor="white", linewidth=0.6)
        ax.hist(sample[sample < c0], bins=bins, weights=np.full((sample < c0).sum(), w), color=BAD,
                alpha=0.75, edgecolor="white", linewidth=0.6)
        basis = f"bootstrap, {draws.size} resamples"
    else:
        lo, hi = min(c - 4 * se, c0 - 0.1), max(c + 4 * se, decision.margin + 0.1)
        grid = np.linspace(lo, hi, 500)
        dens = stats.norm.pdf(grid, c, se)
        ax.plot(grid, dens, color=DATA, linewidth=2, zorder=3)
        ax.fill_between(grid, dens, where=grid >= c0, color=FILL, alpha=0.15, linewidth=0)
        ax.fill_between(grid, dens, where=grid <= c0, color=BAD, alpha=0.55, linewidth=0)
        basis = f"normal, SE = {se:.3f}"
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)  # headroom for the summary box
    ax.set_yticks([])
    # Only quantities on the true-capability scale go on this axis; the estimate needed for
    # approval is a threshold for the estimate and is drawn by required_plot.
    for x, label, style, colour in ((c0, f"C0 = {c0:.2f}", "--", INK_2), (c, f"{name} = {c:.2f}", "-", DATA)):
        right = x >= max(c0, c)
        ax.axvline(x, color=colour, linestyle=style, linewidth=1.3, zorder=4)
        ax.annotate(label, (x, 0.03), xycoords=("data", "axes fraction"), xytext=(3 if right else -3, 0),
                    textcoords="offset points", ha="left" if right else "right", va="bottom", rotation=90,
                    fontsize=8, color=INK_2, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
                    zorder=5)
    verdict = "ACCEPT" if decision.accept else "REJECT"
    rule = "ACCEPT" if decision.threshold_accept else "REJECT"
    lines = [f"P(true {name} < C0) = {decision.p_fail:.3f}", f"alpha = {decision.alpha:.3f}"]
    if decision.method == "analytic" and decision.n:
        from .risk.planning import min_accepted
        lines.append(f"estimate needed = {min_accepted(decision.n, c0=c0, alpha=decision.alpha):.2f}")
    lines += [f"risk-calibrated: {verdict}", f"threshold rule: {rule}"]
    ax.text(0.99, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GOOD if decision.accept else BAD))
    ax.set_xlabel(f"true {name} ({basis})")
    _title(ax, "Approval decision")
    return ax


def _fence_label(study):
    k = study.settings.get("outlier_k", 1.5)
    name = {"iqr": "IQR", "adjusted": "adjusted boxplot"}.get(study.outlier_screen, study.outlier_screen)
    return f"{name} fences, k = {k:g}"


def outlier_chart(study, ax=None, show_data=False):
    """Values in time order against the fences of the outlier screen.

    The fences come from the screen chosen for the study (IQR fences for normal data,
    the skewness-adjusted boxplot otherwise, coefficient ``outlier_k``); flagged points
    are marked, excluded points drawn hollow. Grubbs' test has no fixed fences, so only
    the flagged points are shown. The fence note and the legend sit above the plot area;
    ``show_data=True`` writes each value next to its point (at most 3 decimals).
    """
    ax = _ax(ax, (6.4, 3.0))
    raw, excluded = _series(study)
    t = np.arange(1, raw.size + 1)
    ok = np.isfinite(raw)
    flagged = np.asarray(study.outlier_mask, bool) & ok
    ax.plot(t, np.where(ok, raw, np.nan), color=DATA, linewidth=1.0, alpha=0.6, zorder=2)
    plain = ok & ~flagged & ~excluded
    ax.plot(t[plain], raw[plain], "o", markersize=4, color=DATA, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3)
    if flagged.any():
        ax.plot(t[flagged & ~excluded], raw[flagged & ~excluded], "D", markersize=5.5, color=BAD,
                markeredgecolor="white", zorder=4, label="flagged")
    if excluded.any():
        ax.plot(t[excluded], raw[excluded], "D" if flagged[excluded].any() else "o", markersize=5.5,
                markerfacecolor="none", markeredgecolor=BAD if flagged[excluded].any() else MUTED, zorder=4,
                label="excluded")
    ax.axhline(np.median(raw[ok]), color=INK_2, linewidth=1)
    ax.annotate("median", (1, np.median(raw[ok])), xycoords=("axes fraction", "data"), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    if study.outlier_fences is not None:
        lo, hi = study.outlier_fences
        ax.axhspan(lo, hi, color=FILL, alpha=0.06, linewidth=0, zorder=1)
        for y, label in ((hi, "upper fence"), (lo, "lower fence")):
            ax.axhline(y, color=BAD, linewidth=1, linestyle=(0, (4, 2)), alpha=0.8)
            ax.annotate(f"{label} {y:.4g}", (1, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        span = max(hi, np.nanmax(raw)) - min(lo, np.nanmin(raw))
        ax.set_ylim(min(lo, np.nanmin(raw)) - 0.08 * span, max(hi, np.nanmax(raw)) + 0.08 * span)
        note = _fence_label(study)
    else:
        note = "Grubbs test (no fixed fences)"
    if show_data:
        _value_labels(ax, t, raw, ok)
    _top_legend(ax, f"{note}: {int(flagged.sum())} flagged")
    ax.set_xlabel("observation (time order)")
    ax.set_ylabel("value")
    _title(ax, "Outlier screen")
    return ax


def outlier_box(study, ax=None, show_data=False):
    """Box plot of the usable values with the fences of the outlier screen.

    ``show_data=True`` overlays every value as a jittered dot.
    """
    ax = _ax(ax, (1.8, 3.0))
    raw, _ = _series(study)
    ok = np.isfinite(raw)
    v = raw[ok]
    flagged = np.asarray(study.outlier_mask, bool)[ok]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    inside = v[~flagged] if (~flagged).any() else v
    ax.add_patch(_plt().Rectangle((-0.3, q1), 0.6, q3 - q1, facecolor=FILL, alpha=0.18, edgecolor=DATA,
                                  linewidth=1.2, zorder=2))
    ax.plot([-0.3, 0.3], [med, med], color=DATA, linewidth=2, zorder=3)
    for a, b in ((q3, inside.max()), (q1, inside.min())):
        ax.plot([0, 0], [a, b], color=DATA, linewidth=1.1, zorder=2)
        ax.plot([-0.15, 0.15], [b, b], color=DATA, linewidth=1.1, zorder=2)
    if show_data:
        jitter = np.random.default_rng(0).uniform(-0.22, 0.22, (~flagged).sum())
        ax.plot(jitter, v[~flagged], "o", markersize=3, color=DATA, alpha=0.55, markeredgecolor="none", zorder=3)
    if flagged.any():
        ax.plot(np.zeros(flagged.sum()), v[flagged], "D", markersize=5.5, color=BAD, markeredgecolor="white",
                zorder=4)
    if study.outlier_fences is not None:
        for y in study.outlier_fences:
            ax.axhline(y, color=BAD, linewidth=1, linestyle=(0, (4, 2)), alpha=0.8)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)
    _title(ax, "Box plot")
    return ax


def required_plot(decision, ax=None, n_range=None):
    """Smallest estimate the risk-calibrated rule accepts, against the number of parts.

    The curve solves C - C0 = k * SE(C) for each n (analytic standard error); the dot is
    this study, green when it lies on or above the curve. The gap between the curve and
    C0 is the evidence margin a sample of that size needs.
    """
    from .risk.planning import min_accepted
    ax = _ax(ax)
    n0 = max(decision.n, 2)
    lo_n, hi_n = n_range or (max(5, int(n0 / 3)), max(60, int(n0 * 3)))
    ns = np.arange(max(lo_n, 2), hi_n + 1)
    req = np.array([min_accepted(int(n), c0=decision.c0, alpha=decision.alpha) for n in ns])
    ok = np.isfinite(req)
    ax.plot(ns[ok], req[ok], color=INK, linewidth=1.8, zorder=3, label="required estimate")
    top = max(np.nanmax(req[ok]) if ok.any() else decision.c0, decision.cpk_hat) + 0.15
    ax.fill_between(ns[ok], req[ok], top, color=GOOD, alpha=0.07, linewidth=0)
    ax.fill_between(ns[ok], decision.c0 - 0.3, req[ok], color=BAD, alpha=0.05, linewidth=0)
    ax.axhline(decision.c0, color=INK_2, linestyle="--", linewidth=1.1)
    ax.annotate(f"C0 = {decision.c0:.2f}", (1, decision.c0), xycoords=("axes fraction", "data"), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)
    colour = GOOD if decision.accept else BAD
    if decision.n:
        need = min_accepted(decision.n, c0=decision.c0, alpha=decision.alpha)
        ax.plot(decision.n, decision.cpk_hat, "o", markersize=8, color=colour, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        ax.annotate(f"this study: n = {decision.n}, {decision.index_name} = {decision.cpk_hat:.2f}\n"
                    f"needed at n = {decision.n}: {need:.2f}",
                    (decision.n, decision.cpk_hat), xytext=(8, -4), textcoords="offset points", fontsize=8,
                    color=INK, va="top")
    ax.set_xlim(ns[0], ns[-1])
    ax.set_ylim(min(decision.c0, decision.cpk_hat) - 0.15, top)
    ax.text(0.99, 0.98, "accepted", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=GOOD)
    ax.set_xlabel("number of parts n")
    ax.set_ylabel(f"estimated {decision.index_name}")
    note = f"alpha = {decision.alpha:.3g}" + (f" (cost ratio {decision.cost_ratio:g})" if decision.cost_ratio else "")
    _title(ax, f"Estimate needed for approval, {note}")
    return ax


def _logit(p):
    return math.log(p / (1 - p))


def uccap_contributions(h, ax=None):
    """How the UC-Cap probability is built: anchor z plus learned correction on the logit scale.

    The x axis is the logit; the top axis shows the matching probability. The shaded bands
    are the risk levels Low, Medium and High.
    """
    TAU_LOW, TAU_HIGH = h.thresholds
    ax = _ax(ax, (6.4, 2.6))
    z, f = h.z, h.correction
    total = z + f
    lim = max(4.0, abs(z) + 0.5, abs(total) + 0.5, abs(f) + 0.5)
    lo_t, hi_t = _logit(TAU_LOW), _logit(TAU_HIGH)
    for a, b, colour, name in ((-lim, lo_t, GOOD, "Low"), (lo_t, hi_t, WARN, "Medium"), (hi_t, lim, BAD, "High")):
        ax.axvspan(a, b, color=colour, alpha=0.08, linewidth=0, zorder=0)
        ax.annotate(name, ((max(a, -lim) + min(b, lim)) / 2, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -3), textcoords="offset points", ha="center", va="top", fontsize=8, color=colour)
    ax.barh(2, z, color=DATA, height=0.5, zorder=2)
    ax.barh(1, f, left=z, color=FIT, height=0.5, zorder=2)
    ax.plot([z, z], [1.75, 1.25], color=INK_2, linewidth=0.8, zorder=1)
    ax.plot(total, 0, "o", markersize=8, color={"Low": GOOD, "Medium": WARN}.get(h.level, BAD),
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks([2, 1, 0], [f"anchor z = {z:+.2f}", f"correction f = {f:+.2f}", f"z + f = {total:+.2f}"],
                  fontsize=8)
    ax.set_ylim(-0.6, 2.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("logit of P(Ppk < C0)")
    top = ax.secondary_xaxis("top", functions=(lambda t: t, lambda t: t))
    probs = [p for p in sorted({0.02, 0.1, TAU_LOW, TAU_HIGH, 0.9, 0.98}) if 0 < p < 1 and abs(_logit(p)) <= lim]
    top.set_xticks([_logit(p) for p in probs], [f"{p:g}" for p in probs])
    top.tick_params(labelsize=7.5, colors=INK_2, length=2, color=AXIS)
    top.spines["top"].set_color(AXIS)
    _title(ax, "UC-Cap: anchor and learned correction")
    return ax


def uccap_gauge(h, ax=None):
    """Risk score 0-100 on the Low / Medium / High scale, with the anchor-only score for comparison."""
    TAU_LOW, TAU_HIGH = h.thresholds
    ax = _ax(ax, (6.4, 1.6))
    bands = ((0, 100 * TAU_LOW, GOOD, "Low: accept / monitor"), (100 * TAU_LOW, 100 * TAU_HIGH, WARN, "Medium: review"),
             (100 * TAU_HIGH, 100, BAD, "High: hold / correct"))
    for a, b, colour, name in bands:
        ax.barh(0, b - a, left=a, height=0.5, color=colour, alpha=0.25, linewidth=0)
        ax.annotate(name, ((a + b) / 2, -0.3), ha="center", va="top", fontsize=7.5, color=INK_2)
    colour = {"Low": GOOD, "Medium": WARN}.get(h.level, BAD)
    ax.plot(h.score, 0.36, "v", markersize=10, color=colour, markeredgecolor="white", zorder=3)
    ax.annotate(f"score {h.score:.1f} ({h.level})", (h.score, 0.48), ha="center", va="bottom", fontsize=8.5,
                color=INK, fontweight="bold")
    if math.isfinite(h.p_stat):
        ax.plot(100 * h.p_stat, 0, "|", markersize=17, markeredgewidth=2, color=DATA, zorder=3)
        ax.text(1, 1.02, f"| anchor only, sigmoid(z): {100 * h.p_stat:.1f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5, color=DATA)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, 0.85)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("risk score = 100 x P(Ppk < C0)")
    _title(ax, "UC-Cap risk score")
    return ax


def _figure(panels, title, figsize, width_ratios=None, height_ratios=None, ncols=None):
    plt = _plt()
    ncols = ncols or len(panels)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", squeeze=False,
                             gridspec_kw=dict(width_ratios=width_ratios, height_ratios=height_ratios))
    for ax, draw in zip(axes.flat, panels):
        draw(ax)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return fig


def outlier_figure(study, title=None, show_data=False):
    """Outlier screen: values in time order against the fences, and the box plot.

    ``show_data=True`` labels every point with its value and overlays the values on the box plot.
    """
    fig = _figure([lambda ax: outlier_chart(study, ax, show_data), lambda ax: outlier_box(study, ax, show_data)],
                  title, (10, 3.6),
                  width_ratios=[5, 1])
    a, b = fig.axes[:2]
    b.set_ylim(a.get_ylim())
    b.set_yticklabels([])
    return fig


def stability_figure(study, title=None, show_data=False):
    """Stability over time: individuals chart and moving range chart (control limits, not outlier fences).

    ``show_data=True`` labels every point with its value.
    """
    return _figure([lambda ax: individuals_chart(study, ax, show_data),
                    lambda ax: moving_range_chart(study, ax, show_data)], title,
                   (10, 5.6), ncols=1)


def capability_figure(study, title=None):
    """Capability: histogram with the specification and fitted distribution, and the probability plot."""
    return _figure([lambda ax: capability_histogram(study, ax), lambda ax: probability_plot(study, ax)], title,
                   (11, 4))


def approval_figure(study, decision=None, title=None, **kwargs):
    """Risk-calibrated approval: uncertainty in the estimate, and the estimate needed for this n.

    Without a decision, ``risk_decision(study, **kwargs)`` is computed, with alpha = 0.05
    unless ``alpha`` or ``cost_ratio`` is given.
    """
    if decision is None:
        from .risk import risk_decision
        if "alpha" not in kwargs and "cost_ratio" not in kwargs:
            kwargs["alpha"] = 0.05
        decision = risk_decision(study, **kwargs)
    return _figure([lambda ax: decision_plot(decision, study, ax), lambda ax: required_plot(decision, ax)], title,
                   (11, 4))


def uccap_figure(study_or_risk, title=None, **kwargs):
    """UC-Cap hybrid risk: contributions on the logit scale and the risk score.

    Accepts a :class:`~uccap.HybridRisk` or a study (the risk is then computed with
    ``hybrid_risk(study, **kwargs)``).
    """
    from .hybrid import HybridRisk, hybrid_risk
    h = study_or_risk if isinstance(study_or_risk, HybridRisk) else hybrid_risk(study_or_risk, **kwargs)
    return _figure([lambda ax: uccap_contributions(h, ax), lambda ax: uccap_gauge(h, ax)], title, (8, 4.6),
                   ncols=1, height_ratios=[1.6, 1])


def study_overview(study, decision=None, title=None):
    """One figure with the histogram, probability plot, I and MR charts (and the decision)."""
    plt = _plt()
    if decision is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        capability_histogram(study, axes[0, 0])
        probability_plot(study, axes[0, 1])
        individuals_chart(study, axes[1, 0])
        moving_range_chart(study, axes[1, 1])
    else:
        fig = plt.figure(figsize=(15, 7), layout="constrained")
        gs = fig.add_gridspec(2, 3)
        capability_histogram(study, fig.add_subplot(gs[0, 0]))
        probability_plot(study, fig.add_subplot(gs[0, 1]))
        decision_plot(decision, study, fig.add_subplot(gs[0, 2]))
        individuals_chart(study, fig.add_subplot(gs[1, :2]))
        moving_range_chart(study, fig.add_subplot(gs[1, 2]))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return fig


def batch_plot(result, ax=None, c0=None):
    """Capability estimate and its one-sided lower bound (estimate - k*SE) per dimension.

    A dimension is accepted by the analytic rule when the lower bound reaches C0; the
    decision label next to each row is the one in the results table.
    """
    rows = [(d, s, result.decisions.get(d)) for d, s in result.studies.items()]
    rows = [r for r in rows if r[2] is not None]
    if not rows:
        raise ValueError("no dimensions with a decision to plot")
    ax = _ax(ax, (7.5, 0.42 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for yi, (_, _, dec) in zip(y, rows):
        lower = dec.cpk_hat - dec.k * dec.se
        colour = GOOD if dec.accept else BAD
        ax.plot([lower, dec.cpk_hat], [yi, yi], color=colour, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.plot(dec.cpk_hat, yi, "o", markersize=7, color=colour, markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.annotate("ACCEPT" if dec.accept else "REJECT", (1, yi), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    c0 = c0 if c0 is not None else rows[0][2].c0
    ax.axvline(c0, color=INK_2, linestyle="--", linewidth=1.2)
    ax.annotate(f"C0 = {c0:.2f}", (c0, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK_2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("capability estimate (dot) and lower bound estimate - k·SE (bar)")
    _title(ax, "Batch overview")
    return ax


def planning_plot(expected=(1.5, 1.67, 2.0), *, c0=1.33, alpha=None, cost_ratio=None, power=0.8,
                  n_range=(8, 200), ax=None, n_sim=20000, rng=0):
    """Probability of acceptance against the number of parts, one curve per assumed true Cpk."""
    from .risk.planning import _alpha, acceptance_probability, sample_size
    a = _alpha(alpha, cost_ratio)
    ax = _ax(ax)
    ns = np.unique(np.linspace(n_range[0], n_range[1], 60).astype(int))
    colours = [DATA, FIT, GOOD, MUTED, BAD]
    for i, c in enumerate(np.atleast_1d(expected)):
        p = [acceptance_probability(c, n, c0=c0, alpha=a, n_sim=n_sim, rng=rng) for n in ns]
        colour = colours[i % len(colours)]
        ax.plot(ns, p, color=colour, linewidth=1.8, label=f"true Cpk {c:g}")
        plan = sample_size(c, c0=c0, alpha=a, power=power, n_max=max(n_range[1], 5000), n_sim=n_sim, rng=rng)
        if plan.n is not None and plan.n <= n_range[1]:
            ax.plot(plan.n, plan.achieved, "o", color=colour, markersize=5, markeredgecolor="white", zorder=3)
            ax.annotate(f"n = {plan.n}", (plan.n, plan.achieved), xytext=(4, -10), textcoords="offset points",
                        fontsize=8, color=colour)
    ax.axhline(power, color=INK_2, linestyle="--", linewidth=1.1)
    ax.annotate(f"{power:.0%}", (0, power), xycoords=("axes fraction", "data"), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlim(n_range)
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of parts n")
    ax.set_ylabel("probability of acceptance")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _title(ax, f"Sample size for C0 = {c0:g}, alpha = {a:.3g}")
    return ax


# ---------------------------------------------------------------------------------------------------------
# Operational acceptability monitoring (uccap.monitor)

ZONE_COLOURS = {"continue": GOOD, "monitor": WARN, "intervene": BAD}


def monitor_means(result, ax=None):
    """Subgroup means in production order and the posterior mean of the process."""
    ax = _ax(ax, (9, 3.2))
    f = result.frame
    ax.plot(f["t"], f["mean"], "o-", color=DATA, markersize=3, linewidth=0.9, label="subgroup mean")
    ax.plot(f["t"], f["mu_post"], color=FIT, linewidth=1.6, label="posterior mean")
    ax.set_ylabel("subgroup mean")
    _top_legend(ax)
    _title(ax, "Subgroup evidence")
    return ax


def monitor_chart(result, ax=None):
    """Posterior acceptability probability pi_t with the three zones, t_min and the first intervention."""
    ax = _ax(ax, (9, 3.2))
    f, s = result.frame, result.settings
    ga, gr, tmin = s["gamma_a"], s["gamma_r"], s["t_min"]
    t_end = f["t"].iloc[-1] + 0.5
    ax.axhspan(ga, 1.02, color=GOOD, alpha=0.08, linewidth=0)
    ax.axhspan(-0.02, gr, xmin=0, xmax=1, color=BAD, alpha=0.08, linewidth=0)
    for level, name in ((ga, f"gamma_A = {ga:g}"), (gr, f"gamma_R = {gr:g}")):
        ax.axhline(level, color=INK_2, linestyle="--", linewidth=1)
        ax.annotate(name, (1, level), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.axvline(tmin - 0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate(f"t_min = {tmin}", (tmin - 0.5, 0.02), xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color=MUTED)
    ax.plot(f["t"], f["pi"], color=INK_2, linewidth=1, zorder=2)
    for zone, colour in ZONE_COLOURS.items():
        m = f["zone"] == zone
        if m.any():
            ax.plot(f.loc[m, "t"], f.loc[m, "pi"], "o", markersize=3.5, color=colour, zorder=3,
                    label=f"{zone} ({int(m.sum())})")
    tau = result.intervention
    if tau is not None:
        ax.annotate(f"first intervention t = {tau}", (tau, f.loc[f["t"] == tau, "pi"].iloc[0]), xytext=(8, 14),
                    textcoords="offset points", fontsize=8, color=BAD,
                    arrowprops=dict(arrowstyle="-", color=BAD, linewidth=0.8))
    ax.set_xlim(0.5, t_end)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("subgroup t")
    ax.set_ylabel("pi_t")
    _top_legend(ax)
    _title(ax, "Acceptability probability")
    return ax


def monitor_figure(result, title=None):
    """Monitoring of one characteristic: subgroup means (top) and pi_t with its zones (bottom)."""
    fig = _figure([lambda ax: monitor_means(result, ax), lambda ax: monitor_chart(result, ax)],
                  title if title is not None else (result.name or None), (10, 5.6), ncols=1,
                  height_ratios=[1, 1.2])
    a, b = fig.axes[:2]
    a.sharex(b)
    a.tick_params(labelbottom=False)
    return fig


def monitor_overview(batch, title=None):
    """Many dimensions: final pi, empirical Ppk against final pi by category, and first interventions."""
    from matplotlib.ticker import MaxNLocator
    from .monitor.sboam import CATEGORIES
    f = batch.frame
    s = batch.settings
    ok = f[f["pi_final"].notna()] if "pi_final" in f else f.iloc[:0]
    if ok.empty:
        raise ValueError("no monitored dimensions to plot")
    ga, gr, tmin = s["gamma_a"], s["gamma_r"], s["t_min"]

    def final(ax):
        _style(ax)
        ax.hist(ok["pi_final"], bins=np.linspace(0, 1, 21), color=DATA, edgecolor="white")
        for level, name, y in ((gr, f"gamma_R = {gr:g}", 0.9), (ga, f"gamma_A = {ga:g}", 1)):
            ax.axvline(level, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(name, (level, y), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                        textcoords="offset points", ha="right", va="top", fontsize=8, color=INK_2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("final pi")
        ax.set_ylabel("dimensions")
        _title(ax, "Final acceptability")

    def scatter(ax):
        _style(ax)
        colours = dict(zip(CATEGORIES, (BAD, WARN, GOOD, MUTED)))
        for cat in CATEGORIES:
            m = ok["category"] == cat
            if m.any() and "Ppk_normal" in ok:
                ax.plot(ok.loc[m, "Ppk_normal"], ok.loc[m, "pi_final"], "o", markersize=4, alpha=0.8,
                        color=colours[cat], label=f"{cat} ({int(m.sum())})")
        ax.set_xlabel("Ppk (overall sigma)")
        ax.set_ylabel("final pi")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        _title(ax, "Capability and final acceptability")

    def first(ax):
        _style(ax)
        tau = ok["first_intervention"].dropna().astype(int)
        if len(tau):
            width = max(1, int(np.ceil((tau.max() - tau.min() + 1) / 24)))   # at most about 24 bars
            ax.hist(tau, bins=np.arange(tau.min() - 0.5, tau.max() + width + 0.5, width), color=BAD,
                    edgecolor="white")
        else:
            ax.text(0.5, 0.5, "no intervention", transform=ax.transAxes, ha="center", va="center", color=INK_2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axvline(tmin - 0.5, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate(f"t_min = {tmin}", (tmin - 0.5, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                    textcoords="offset points", va="top", fontsize=8, color=MUTED)
        ax.set_xlabel("first intervention (subgroup)")
        ax.set_ylabel("dimensions")
        _title(ax, f"First intervention ({len(tau)} of {len(ok)})")

    return _figure([final, scatter, first], title, (12, 4))
