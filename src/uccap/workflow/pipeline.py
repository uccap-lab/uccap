"""End-to-end capability study following the practical PCI workflow.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026), Figure 1:
tolerance type -> outlier screening -> normality test -> normal indices, or
best-fit distribution and percentile indices for non-normal data.
"""
from __future__ import annotations

import math

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import outliers as _outliers
from .fitting import DEFAULT_CANDIDATES, FitResult, best_fit
from .indices import Spec, normal_indices
from .normality import NormalityResult, check_normality
from .percentile import EmpiricalQuantiles, percentile_indices
from scipy import stats

from .sigma import SUBGROUP_METHODS, WITHIN_METHODS, sigma_between, sigma_overall

MIN_N = 8          # Anderson-Darling and the moving-range estimators need at least this many
SMALL_N = 20       # below this, warn that indices and their standard errors are unreliable
RATIO_RANGE = (0.75, 1.25)  # sigma_within / sigma_overall outside this range is flagged
LARGE_DELTA = 10.0  # normal passes the test but trails the best fit by more than this AICc
LARGE_N = 500      # above this, normality tests flag deviations too small to matter

_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


class DataError(ValueError):
    """The data cannot support a capability study."""


def _coerce(x) -> tuple[np.ndarray, list[tuple[int, object]]]:
    """Convert input to floats. Missing or unreadable entries become NaN.

    Returns the array and a list of (observation number, original value) for
    entries that were present but not numeric, infinite, or ambiguous (a comma
    without a decimal point, such as "4,618", which may be a decimal or a thousands
    separator). Units after a number ("4.618 mm") are dropped.
    """
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    arr = np.asarray(x, dtype=object)
    if arr.ndim > 1:
        if 1 in arr.shape:
            arr = arr.ravel()
        else:
            raise DataError(
                f"x has shape {arr.shape}; pass one dimension at a time, use subgroups= "
                "for rational subgroups, or uccap.batch_study for several dimensions")
    out = np.full(arr.size, np.nan)
    bad = []
    for i, v in enumerate(arr):
        if v is None:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            f = float(v)
            if np.isinf(f):
                bad.append((i + 1, v))
            elif not np.isnan(f):
                out[i] = f
            continue
        text = str(v).strip()
        if text == "" or text.lower() in {"nan", "na", "n/a", "null", "none", "-"}:
            continue
        # "1,234.5" has a thousands comma; "4,618" alone could be either, so it is not guessed
        if "," in text and "." not in text:
            bad.append((i + 1, v))
            continue
        m = _NUMBER.match(text.replace(",", ""))
        if m:
            out[i] = float(m.group(1))
        else:
            bad.append((i + 1, v))
    return out, bad



@dataclass
class CapabilityStudy:
    data: np.ndarray  # observations used in the calculations
    spec: Spec
    mean: float
    sigma_within: float
    sigma_overall: float
    normality: NormalityResult
    outlier_mask: np.ndarray
    indices: dict[str, float]
    method: str  # "normal" or "percentile"
    sigma_method: str = "amr"
    fit: Optional[FitResult] = None
    candidate_fits: list[FitResult] = field(default_factory=list)
    failed_fits: list = field(default_factory=list)  # (name, reason)
    raw_data: Optional[np.ndarray] = None  # all observations, in time order
    excluded_mask: Optional[np.ndarray] = None  # over raw_data
    settings: dict = field(default_factory=dict, repr=False)
    missing_mask: Optional[np.ndarray] = None  # over raw_data: missing or unreadable
    warnings: list = field(default_factory=list)
    skewness: float = float("nan")
    excess_kurtosis: float = float("nan")
    time_ordered: bool = True
    percentile_method: str = "fit"
    outlier_screen: str = "iqr"  # the screen actually used ("auto" resolved)
    outlier_fences: Optional[tuple] = None  # (lower, upper) of the IQR or adjusted screen
    sigma_between: float = float("nan")  # between subgroup means (subgroups only)
    normality_other: dict = field(default_factory=dict)  # p-values of the other tests, for reference
    ppm: dict = field(default_factory=dict)  # observed and expected parts per million out of specification

    @property
    def sigma_bw(self) -> float:
        """Between/within sigma sqrt(sigma_between^2 + sigma_within^2) (subgroups only)."""
        return math.sqrt(self.sigma_between ** 2 + self.sigma_within ** 2)

    def lower_bound(self, confidence: float = 0.95) -> float:
        """One-sided lower confidence bound of the capability used for approval.

        estimate - z * SE with SE = sqrt((1/9 + estimate^2 / 2) / n), the standard error of
        the risk-calibration paper; an approximation for percentile indices.
        """
        from scipy.stats import norm
        c = self.cpk
        return c - norm.ppf(confidence) * math.sqrt((1 / 9 + c * c / 2) / self.n)

    def _normal_interval(self, half_width_factor: float) -> tuple[float, float]:
        if self.method != "normal":
            raise ValueError("these intervals assume normal data; the study used the percentile method")
        h = half_width_factor * self.sigma_overall
        return self.mean - h, self.mean + h

    def prediction_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Two-sided interval expected to contain the next single part, for normal data.

        mean +- t(1 - a/2, n - 1) s sqrt(1 + 1/n), with s the overall sigma. The workflow
        article lists prediction intervals as a supplementary tool; this is the standard form.
        """
        from scipy.stats import t
        return self._normal_interval(t.ppf((1 + confidence) / 2, self.n - 1) * math.sqrt(1 + 1 / self.n))

    def tolerance_interval(self, coverage: float = 0.99, confidence: float = 0.95) -> tuple[float, float]:
        """Two-sided interval containing at least ``coverage`` of the parts with ``confidence``, for normal data.

        mean +- k s with Howe's factor k = z((1 + p)/2) sqrt((n - 1)(1 + 1/n) / chi2(1 - confidence, n - 1)).
        The workflow article lists tolerance intervals as a supplementary tool; this is the standard form.
        """
        from scipy.stats import chi2, norm
        n = self.n
        k = norm.ppf((1 + coverage) / 2) * math.sqrt((n - 1) * (1 + 1 / n) / chi2.ppf(1 - confidence, n - 1))
        return self._normal_interval(k)

    @property
    def n_missing(self) -> int:
        return 0 if self.missing_mask is None else int(self.missing_mask.sum())

    @property
    def n_excluded(self) -> int:
        return 0 if self.excluded_mask is None else int(self.excluded_mask.sum())

    def without_outliers(self) -> "CapabilityStudy":
        """Recompute the study with the flagged outliers excluded.

        Only do this after checking that the flagged points are special causes
        (measurement error, setup, damaged part), not genuine process variation.
        """
        return capability_study(self.raw_data, **{**self.settings, "exclude": "flagged"})

    @property
    def n(self) -> int:
        return int(self.data.size)

    @property
    def cpk(self) -> float:
        """The capability estimate used for approval: Ppk (normal) or CNpk (non-normal).

        Approval uses overall sigma, as in the risk-calibration paper, whose
        Cpk estimator and standard error are based on the sample standard
        deviation. The within-sigma Cpk is reported as short-term potential, so
        the choice of within-sigma estimator never changes the decision.
        """
        return self.indices["CNpk"] if self.method == "percentile" else self.indices["Ppk"]

    @property
    def cpk_name(self) -> str:
        return "CNpk" if self.method == "percentile" else "Ppk"

    @property
    def sigma_ratio(self) -> float:
        """sigma_within / sigma_overall; well below 1 suggests drift between subgroups."""
        return self.sigma_within / self.sigma_overall

    def report(self) -> str:
        """Formatted text report of the study."""
        from ..report import study_report
        return study_report(self)

    summary = report

    def plot(self, decision=None, title=None, kind="overview", show_data=False, decimals=None):
        """Figure of the study (needs matplotlib).

        ``kind`` picks the group of panels: "overview" (histogram, probability plot,
        I-MR charts and, with a decision, the approval panel), "outliers", "stability",
        "capability", "approval" (computes the decision at alpha = 0.05 if none is given)
        or "uccap". ``show_data=True`` writes the raw values on the time-order charts
        ("outliers" and "stability") and overlays them on the box plot; ``decimals`` sets the
        decimals of these labels (default: as many as the data have, at most 3).
        """
        from .. import plot as _p
        if kind == "overview":
            return _p.study_overview(self, decision, title)
        if kind == "approval":
            return _p.approval_figure(self, decision, title)
        figures = {"outliers": _p.outlier_figure, "stability": _p.stability_figure,
                   "capability": _p.capability_figure, "uccap": _p.uccap_figure}
        if kind not in figures:
            raise ValueError('kind must be "overview", "outliers", "stability", "capability", "approval" '
                             'or "uccap"')
        if kind in ("outliers", "stability"):
            return figures[kind](self, title=title, show_data=show_data, decimals=decimals)
        return figures[kind](self, title=title)

    def __str__(self) -> str:
        return self.report()

    def to_dict(self) -> dict:
        """Flat dictionary of the main results, e.g. for a pandas DataFrame row."""
        out = {"n": self.n, "n_excluded": self.n_excluded, "n_missing": self.n_missing,
               "skewness": self.skewness, "excess_kurtosis": self.excess_kurtosis, "mean": self.mean, "sigma_within": self.sigma_within,
               "sigma_overall": self.sigma_overall, "normality_p": self.normality.p_value,
               "method": self.method, "best_fit": self.fit.name if self.fit else None,
               "n_outliers": int(self.outlier_mask.sum()), "cpk_used": self.cpk,
               "lower_bound_95": self.lower_bound(0.95),
               "ppm_observed": self.ppm.get("observed", {}).get("total", float("nan")),
               "ppm_expected_overall": self.ppm.get("overall", {}).get("total", float("nan")),
               "warnings": "; ".join(self.warnings)}
        out.update(self.indices)
        return out

def capability_study(
    x,
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    target: Optional[float] = None,
    *,
    subgroups=None,
    within: str = "amr",
    mr_span: int = 2,
    time_ordered: bool = True,
    normality_test: str = "anderson",
    alpha: float = 0.05,
    outlier_method: str = "auto",
    outlier_k: float = 1.5,
    candidates=DEFAULT_CANDIDATES,
    force_method: Optional[str] = None,
    percentile_method: str = "fit",
    exclude=None,
    asymmetric: str = "paper",
) -> CapabilityStudy:
    """Run the capability workflow on time-ordered data ``x``.

    Missing values (NaN, None, empty strings) and unreadable entries are dropped
    and listed in the report; moving ranges never bridge them.

    ``within`` selects the within sigma for individual values in time order: "amr"
    (average moving range, default), "mmr" (median moving range) or "srmssd"; ``mr_span``
    is the number of consecutive points per moving range (2 to 10, default 2). Pass
    ``subgroups`` (a list of arrays) to estimate within sigma from subgroups instead:
    ``within`` is then "pooled" (default), "rbar" (R-bar/d2) or "sbar" (S-bar/c4, both for
    equal sizes), and the between-subgroup sigma is reported as well. Set ``time_ordered=False`` when the time order is
    unknown: within sigma is then not estimated and approval uses Ppk.
    ``force_method`` ("normal" or "percentile") skips the normality branch.
    For non-normal data, ``percentile_method="fit"`` (default, parametric) takes the
    percentiles from the best-fitting distribution; ``"empirical"`` uses sample
    quantiles (non-parametric), which needs on the order of 1000 observations.

    ``exclude`` removes observations before the indices are computed: ``"flagged"``
    drops the points flagged by the outlier screen, or pass observation numbers
    (1-based, as printed in the report). Outlier screening always runs on all
    observations, and the report lists what was excluded. ``outlier_method="auto"``
    uses IQR fences for normal data and the skewness-adjusted boxplot for
    non-normal data. ``outlier_k`` is the fence coefficient of both (Q1 - k IQR,
    Q3 + k IQR; 1.5 by default, 3 flags only far-out points); Grubbs' test ignores it.

    When the target lies off the midpoint of two limits, ``asymmetric`` picks the
    starred indices: "paper" (default; workflow paper Eqs. (3), (5), (9), (13)) or
    "pearn-chen" (d* and A* form).
    """
    settings = dict(lsl=lsl, usl=usl, target=target, within=within, mr_span=mr_span, time_ordered=time_ordered,
                    normality_test=normality_test, alpha=alpha, outlier_method=outlier_method,
                    outlier_k=outlier_k, candidates=candidates, force_method=force_method,
                    percentile_method=percentile_method, asymmetric=asymmetric)
    spec = Spec(lsl, usl, target, asymmetric)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    raw, unreadable = _coerce(x)
    missing = np.isnan(raw)
    warn: list[str] = []
    if unreadable:
        shown = ", ".join(f"obs. {i} ({v!r})" for i, v in unreadable[:5])
        more = f" and {len(unreadable) - 5} more" if len(unreadable) > 5 else ""
        warn.append(f"{len(unreadable)} non-numeric entries ignored: {shown}{more}")
        if any(isinstance(v, str) and "," in v for _, v in unreadable):
            warn.append("a comma without a decimal point may be a decimal or a thousands separator, "
                        "so it is not guessed; write 4.618 or 4618")
    n_blank = int(missing.sum()) - len(unreadable)
    if n_blank:
        warn.append(f"{n_blank} missing values ignored")

    valid = raw[~missing]
    if valid.size < MIN_N:
        raise DataError(f"{valid.size} usable observations; at least {MIN_N} are needed")
    if np.ptp(valid) == 0:
        raise DataError("all observations are identical; check the measurement resolution "
                        "or the data entry")

    # Outlier screening on all usable observations; the normality test picks the screen.
    screen = outlier_method
    if screen == "auto":
        screen = "iqr" if check_normality(valid, normality_test, alpha).is_normal else "adjusted"
    if not outlier_k > 0:
        raise ValueError("outlier_k must be positive")
    mask = np.zeros(raw.size, bool)
    mask[~missing] = (_outliers.METHODS[screen](valid) if screen == "grubbs"
                      else _outliers.METHODS[screen](valid, k=outlier_k))
    fences = _outliers.FENCES[screen](valid, outlier_k) if screen in _outliers.FENCES else None

    excluded = np.zeros(raw.size, bool)
    if exclude is not None:
        if subgroups is not None:
            raise ValueError("exclude is not supported together with subgroups")
        if isinstance(exclude, str):
            if exclude != "flagged":
                raise ValueError('exclude must be "flagged" or a list of observation numbers')
            excluded = mask.copy()
        else:
            obs = np.asarray(list(exclude), int)
            if obs.size and (obs.min() < 1 or obs.max() > raw.size):
                raise ValueError(f"observation numbers must be between 1 and {raw.size}")
            excluded[obs - 1] = True
    gaps = missing | excluded
    data = raw[~gaps]
    if data.size < MIN_N:
        raise DataError(f"{data.size} observations left after exclusion; at least {MIN_N} are needed")

    s_between = float("nan")
    if subgroups is not None:
        sigma_method = within if within in SUBGROUP_METHODS else "pooled"
        if within not in SUBGROUP_METHODS and within != "amr":
            raise ValueError('with subgroups, within must be "pooled", "rbar" or "sbar"')
        s_within = SUBGROUP_METHODS[sigma_method](subgroups)
        if len(subgroups) >= 2:
            s_between = sigma_between(subgroups, s_within)
    elif not time_ordered:
        s_within, sigma_method = float("nan"), "not estimated (time order unknown)"
    else:
        if within not in WITHIN_METHODS:
            raise ValueError(f"within must be one of {', '.join(WITHIN_METHODS)} (or pass subgroups)")
        if not (isinstance(mr_span, (int, np.integer)) and 2 <= mr_span <= 10):
            raise ValueError("mr_span must be an integer from 2 to 10")
        # Missing and excluded points become gaps so moving ranges never bridge them.
        series = np.where(gaps, np.nan, raw)
        if within == "srmssd":
            s_within = WITHIN_METHODS[within](series)
        else:
            s_within = WITHIN_METHODS[within](series, int(mr_span))
        sigma_method = within
    s_overall = sigma_overall(data)
    mu = float(data.mean())
    normality = check_normality(data, normality_test, alpha)
    other = {}
    for test in ("anderson", "shapiro", "k2"):  # fast tests, reported for reference only
        if test != normality_test and not (test == "k2" and data.size < 8):
            try:
                other[test] = check_normality(data, test, alpha).p_value
            except Exception:  # pragma: no cover - a reference value must never stop the study
                pass
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))

    # Data-quality warnings.
    n = data.size
    if n < SMALL_N:
        warn.append(f"small sample (n = {n}): indices and their standard errors are unreliable")
    if time_ordered and subgroups is None and (np.all(np.diff(data) >= 0) or np.all(np.diff(data) <= 0)):
        warn.append("data are sorted; time order looks lost, so within sigma is too small "
                    "(use time_ordered=False)")
    uniq = np.unique(data)
    step = np.min(np.diff(uniq)) if uniq.size > 1 else 0.0
    if uniq.size < 10 or (step > 0 and step > s_overall / 2):
        warn.append(f"coarse measurement resolution ({uniq.size} distinct values, step {step:.3g}, "
                    f"sigma {s_overall:.3g}); normality tests and indices may be distorted, check the MSA")
    if n >= LARGE_N and not normality.is_normal and abs(skew) < 0.5 and abs(kurt) < 1:
        warn.append(f"large sample (n = {n}): normality rejected although skewness {skew:.2f} and "
                    f"excess kurtosis {kurt:.2f} are small; consider force_method='normal'")
    ratio = s_within / s_overall
    if ratio == ratio and not RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]:
        why = ("variation between subgroups or drift over time; the process may not be stable"
               if ratio < RATIO_RANGE[0] else "check the time order or the subgroup definition")
        warn.append(f"sigma within / overall = {ratio:.2f}: {why}")
    if (spec.usl is not None and mu > spec.usl) or (spec.lsl is not None and mu < spec.lsl):
        warn.append("the mean is outside the specification limits")

    method = force_method or ("normal" if normality.is_normal else "percentile")
    fit, fits, failed = None, [], []
    if method == "normal":
        indices = normal_indices(mu, s_within, s_overall, spec)
        if force_method is None:
            # Reference ranking only: the workflow keeps the normal method when the test passes.
            fits = best_fit(data, candidates, failures=failed)
            normal_fit = next((f for f in fits if f.name == "normal"), None)
            if normal_fit is not None and normal_fit.aicc - fits[0].aicc > LARGE_DELTA:
                warn.append(f"normality test passed, but {fits[0].name} fits much better "
                            f"(dAICc of normal = {normal_fit.aicc - fits[0].aicc:.1f}); "
                            "consider force_method='percentile'")
    elif method == "percentile":
        fits = best_fit(data, candidates, failures=failed)
        fit = fits[0]
        if percentile_method == "fit":
            indices = percentile_indices(fit.dist, spec)
        elif percentile_method == "empirical":
            indices = percentile_indices(EmpiricalQuantiles(data), spec)
            if n < 1000:
                warn.append(f"empirical percentiles with n = {n}: the 0.135% / 99.865% quantiles "
                            "are close to the sample min / max, so CNpk is biased")
        else:
            raise ValueError("percentile_method must be 'fit' or 'empirical'")
    else:
        raise ValueError("force_method must be 'normal' or 'percentile'")
    ppm = _ppm(data, spec, mu, s_within, s_overall, fit.dist if (method == "percentile" and fit is not None
                                                                  and percentile_method == "fit") else None)
    return CapabilityStudy(data, spec, mu, s_within, s_overall, normality, mask,
                           indices, method, sigma_method, fit, fits, failed, raw, excluded,
                           settings, missing, warn, skew, kurt, time_ordered, percentile_method,
                           screen, fences, s_between, other, ppm)


def _ppm(data, spec, mu, s_within, s_overall, dist=None) -> dict:
    """Parts per million below LSL, above USL and in total: observed, and expected from the model.

    Expected values use the normal model with within and overall sigma, or the fitted
    distribution (overall only) for non-normal data.
    """
    out = {}
    lo, hi = spec.lsl, spec.usl

    def tails(below, above):
        b = below if lo is not None else 0.0
        a = above if hi is not None else 0.0
        return {"below": 1e6 * b, "above": 1e6 * a, "total": 1e6 * (b + a)}
    out["observed"] = tails(np.mean(data < lo) if lo is not None else 0.0,
                            np.mean(data > hi) if hi is not None else 0.0)
    if dist is None:
        for name, s in (("within", s_within), ("overall", s_overall)):
            if s == s and s > 0:
                out[name] = tails(stats.norm.cdf(lo, mu, s) if lo is not None else 0.0,
                                  stats.norm.sf(hi, mu, s) if hi is not None else 0.0)
    else:
        out["overall"] = tails(float(dist.cdf(lo)) if lo is not None else 0.0,
                               float(dist.sf(hi)) if hi is not None else 0.0)
    return out
