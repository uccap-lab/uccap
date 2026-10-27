"""Plain-text reports for capability studies and approval decisions.

Layout: labels are right-aligned against a common column, values start
left-aligned at the next column, and numbers in a column share a decimal point.
"""
from __future__ import annotations

import math

import numpy as np

WIDTH = 64
LABEL = 20  # width of the right-aligned label column
HEAVY = "=" * WIDTH
LIGHT = "-" * WIDTH

TEST_NAMES = {"anderson": "Anderson-Darling", "shapiro": "Shapiro-Wilk",
              "ks": "Kolmogorov-Smirnov (Lilliefors)", "chi2": "Chi-square goodness of fit",
              "k2": "D'Agostino-Pearson K2"}
SIGMA_NAMES = {"amr": "average moving range", "mmr": "median moving range",
               "srmssd": "successive differences", "pooled": "pooled subgroups",
               "rbar": "R-bar / d2 of subgroups", "sbar": "S-bar / c4 of subgroups"}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _idx(v) -> str:
    """Index value with a sign slot, so decimal points line up in a column."""
    return "" if _missing(v) else f"{v: .3f}"


def _row(label: str, value: str) -> str:
    return f"{label:>{LABEL}} : {value}"


def _title(text: str) -> list[str]:
    return [HEAVY, f"  {text}", HEAVY]


def _section(text: str) -> list[str]:
    return [LIGHT, f"  {text}", ""]


def study_report(study) -> str:
    spec = study.spec
    limits = "   ".join(
        f"{name} = {val:g}" for name, val in (("LSL", spec.lsl), ("Target", spec.target), ("USL", spec.usl))
        if val is not None)
    if not spec.is_bilateral:
        limits += "   (one-sided)"
    elif spec.is_asymmetric:
        limits += "   (asymmetric, " + ("Pearn-Chen" if spec.asymmetric == "pearn-chen" else "starred indices") + ")"
    norm = study.normality
    flagged = [str(i + 1) for i in np.flatnonzero(study.outlier_mask)]
    excluded = [str(i + 1) for i in np.flatnonzero(study.excluded_mask)] if study.n_excluded else []
    if not flagged:
        outliers = "none flagged"
    elif excluded and set(excluded) == set(flagged):
        outliers = f"{len(flagged)} flagged (obs. {', '.join(flagged)}), excluded"
    else:
        outliers = f"{len(flagged)} flagged (obs. {', '.join(flagged)}), review before use"
    k = (getattr(study, "settings", None) or {}).get("outlier_k", 1.5)
    if k != 1.5 and (study.settings or {}).get("outlier_method") != "grubbs":
        outliers += f"   (fences k = {k:g})"
    size = str(study.n)
    notes = []
    if excluded:
        notes.append(f"{len(excluded)} excluded: obs. {', '.join(excluded)}")
    if study.n_missing:
        notes.append(f"{study.n_missing} missing")
    if notes:
        size += f"   (of {study.raw_data.size}; {'; '.join(notes)})"
    method_name = SIGMA_NAMES.get(study.sigma_method, study.sigma_method)
    span = (study.settings or {}).get("mr_span", 2)
    if study.sigma_method in ("amr", "mmr") and span != 2:
        method_name += f", span {span}"
    within = (f"{study.sigma_within:.4f}   ({method_name})"
              if not _missing(study.sigma_within) else f"- ({study.sigma_method})")

    lines = _title("PROCESS CAPABILITY STUDY") + [
        _row("Sample size", size),
        _row("Specification", limits),
        _row("Mean", f"{study.mean:.4f}"),
        _row("Sigma within", within),
    ]
    if not _missing(getattr(study, "sigma_between", float("nan"))):
        lines += [_row("Sigma between", f"{study.sigma_between:.4f}   (subgroup means)"),
                  _row("Sigma between/within", f"{study.sigma_bw:.4f}")]
    lines += [
        _row("Sigma overall", f"{study.sigma_overall:.4f}"
             + (f"   (within / overall = {study.sigma_ratio:.2f})" if not _missing(study.sigma_within) else "")),
        _row("Normality test", TEST_NAMES.get(norm.test, norm.test)),
        _row("p-value", f"{norm.p_value:.3f}   ({'normal' if norm.is_normal else 'non-normal'} at alpha = {norm.alpha:g})"),
    ]
    other = getattr(study, "normality_other", None) or {}
    if other:
        short = {"anderson": "AD", "shapiro": "SW", "chi2": "Chi2", "k2": "K2", "ks": "KS"}
        lines.append(_row("Other tests", "   ".join(f"{short.get(t, t)} p = {p:.3f}" for t, p in other.items())
                          + "   (reference)"))
    lines += [
        _row("Skewness", f"{study.skewness: .3f}   (excess kurtosis {study.excess_kurtosis:.3f})"),
        _row("Outliers", outliers),
    ]

    ind = study.indices
    if study.method == "normal":
        lines += _section("Normal-based indices")
        lines.append(f"{'Within':>{LABEL}}   {'':8}{'Overall':>10}")
        for w, o in (("Cp", "Pp"), ("Cpk", "Ppk"), ("Cpu", "Ppu"), ("Cpl", "Ppl"), ("Cpm", "Ppm"), ("Cpmk", "Ppmk")):
            wv, ov = ind.get(w), ind.get(o) if o else None
            if _missing(wv) and _missing(ov):
                continue
            right = f"{o:>10} : {_idx(ov)}" if o and not _missing(ov) else ""
            lines.append(f"{w:>{LABEL}} : {_idx(wv):<8}{right}".rstrip())
    else:
        source = ("empirical sample quantiles" if study.percentile_method == "empirical"
                  else f"best fit: {study.fit.name}")
        lines += _section(f"Percentile-based indices ({source})")
        for name in ("CNp", "CNpk", "CNpu", "CNpl", "CNpm", "CNpmk"):
            v = ind.get(name)
            if not _missing(v):
                lines.append(_row(name, _idx(v)))
    if study.candidate_fits:
        title = "Distribution ranking (sorted by AICc; lower is better)"
        if study.method == "normal":
            title += " - reference only"
        lines += _section(title)
        lines.append(f"{'Distribution':>{LABEL}}   {'k':<4}{'AIC':<11}{'BIC':<11}{'AICc':<11}{'dAICc'}")
        best = study.candidate_fits[0].aicc
        for f in study.candidate_fits:
            lines.append(f"{f.name:>{LABEL}} : {f.k:<4}{f.aic:<11.1f}{f.bic:<11.1f}{f.aicc:<11.1f}{f.aicc - best:.1f}")
        for name, reason in study.failed_fits:
            lines.append(f"{name:>{LABEL}} : not fitted ({reason})")
        lines += ["", "  k = parameters; dAICc = AICc - best AICc (< 2: about as good as the best)"]
    ppm = getattr(study, "ppm", None) or {}
    if ppm:
        lines += _section("Parts per million outside the specification")
        lines.append(f"{'':>{LABEL}}   {'< LSL':>12}{'> USL':>12}{'Total':>12}")
        names = {"observed": "Observed", "within": "Expected within", "overall": "Expected overall"}
        for key in ("observed", "within", "overall"):
            if key in ppm:
                v = ppm[key]
                lines.append(f"{names[key]:>{LABEL}} : {v['below']:>12.1f}{v['above']:>12.1f}{v['total']:>12.1f}")
    if study.method == "normal" and study.n >= 3:
        lo, hi = study.prediction_interval(0.95)
        tlo, thi = study.tolerance_interval(0.99, 0.95)
        lines += _section("Intervals (normal, overall sigma)")
        lines.append(_row("Prediction, 95%", f"[{lo:.4f}, {hi:.4f}]   (next single part)"))
        lines.append(_row("Tolerance, 99%/95%", f"[{tlo:.4f}, {thi:.4f}]   (99% of parts, 95% confidence)"))
    lb = study.lower_bound(0.95)
    lines += [LIGHT, _row("Capability used", f"{study.cpk_name} = {study.cpk:.3f}   "
                          f"(95% lower bound {lb:.3f})")]
    if study.warnings:
        lines += [LIGHT, "  Warnings", ""] + [f"  ! {w}" for w in study.warnings]
    lines.append(HEAVY)
    return "\n".join(lines)


def decision_report(d) -> str:
    if d.cost_ratio is not None:
        basis = f"cost ratio c_FA/c_FR = {d.cost_ratio:g}  ->  alpha = {d.alpha:.3f}"
    else:
        basis = f"tolerable failure probability, alpha = {d.alpha:.3f}"
    verdict = lambda b: "ACCEPT" if b else "REJECT"
    rel = "<=" if d.p_fail <= d.alpha else ">"
    lines = _title("RISK-CALIBRATED APPROVAL") + [
        _row("Requirement", f"{d.index_name} >= C0 = {d.c0:g}"),
        _row("Risk basis", basis),
        LIGHT,
        _row(f"{d.index_name} estimate", f"{d.cpk_hat:.3f}"),
        _row("Standard error", f"{d.se:.3f}" + ("" if d.method == "analytic" else "   (analytic, for reference)")),
        _row("P(true < C0)", f"{d.p_fail:.3f}   ({d.method})"),
        _row("Required estimate", f"{d.margin:.3f}   " + (f"(C0 + {d.k:.3f} x SE at this estimate)" if d.method == "analytic"
                                                           else f"(analytic: C0 + {d.k:.3f} x SE; for reference)")),
    ]
    if d.method == "analytic" and d.n >= 2:
        from .risk.planning import min_accepted
        lines.append(_row("Smallest accepted", f"{min_accepted(d.n, c0=d.c0, alpha=d.alpha):.3f}   (estimate that just passes with n = {d.n})"))
    lines += [
        LIGHT,
        _row("Threshold rule", f"{verdict(d.threshold_accept):<8}(estimate {'>=' if d.threshold_accept else '<'} C0)"),
        _row("Risk-calibrated", f"{verdict(d.accept):<8}(P = {d.p_fail:.3f} {rel} alpha)"),
    ]
    if d.n < 20:
        lines += ["", f"  ! small sample (n = {d.n}): the analytic standard error is a",
                  "    large-sample approximation and tends to be optimistic here."]
    if d.accept != d.threshold_accept:
        lines += ["", "  Note: the two rules differ because the estimate is close to C0 for this",
                  "  sample size. C0 is unchanged; the calibrated rule shows how much evidence",
                  "  the sample gives. More parts or the actual cost ratio settle the case."]
    lines.append(HEAVY)
    return "\n".join(lines)
