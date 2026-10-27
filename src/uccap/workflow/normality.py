"""Normality tests.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026), Section 2.
Anderson–Darling is the default because it weights the tails, where capability is decided.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class NormalityResult:
    test: str
    statistic: float
    p_value: float
    alpha: float

    @property
    def is_normal(self) -> bool:
        return self.p_value >= self.alpha


def _ad_pvalue(a2: float, n: int) -> float:
    """D'Agostino & Stephens (1986) p-value for A^2 with estimated mean and variance."""
    a = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if a >= 0.6:
        p = math.exp(1.2937 - 5.709 * a + 0.0186 * a ** 2)
    elif a >= 0.34:
        p = math.exp(0.9177 - 4.279 * a - 1.38 * a ** 2)
    elif a >= 0.2:
        p = 1 - math.exp(-8.318 + 42.796 * a - 59.938 * a ** 2)
    else:
        p = 1 - math.exp(-13.436 + 101.14 * a - 223.73 * a ** 2)
    return min(max(p, 0.0), 1.0)


def anderson_darling(x) -> tuple[float, float]:
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n < 8:
        raise ValueError("Anderson-Darling needs at least 8 observations")
    z = (x - x.mean()) / x.std(ddof=1)
    logcdf = stats.norm.logcdf(z)
    logsf = stats.norm.logsf(z)
    i = np.arange(1, n + 1)
    a2 = -n - np.sum((2 * i - 1) * (logcdf + logsf[::-1])) / n
    return float(a2), _ad_pvalue(float(a2), n)


def chi_square_gof(x) -> tuple[float, float]:
    """Pearson chi-square goodness-of-fit test against a normal with estimated mean and sd.

    The workflow article lists the chi-square test as suited to large samples and sensitive
    to the binning. Here the bins are equiprobable under the fitted normal and their number is
    ceil(2 n^0.4) (Moore's rule, an implementation choice); two degrees of freedom are
    subtracted for the estimated parameters.
    """
    x = np.asarray(x, float)
    n = x.size
    k = max(int(math.ceil(2 * n ** 0.4)), 4)
    if n < 5 * k // 2:
        raise ValueError("the chi-square test needs more observations")
    edges = stats.norm.ppf(np.linspace(0, 1, k + 1), x.mean(), x.std(ddof=1))
    observed = np.histogram(x, bins=np.r_[-np.inf, edges[1:-1], np.inf])[0]
    expected = n / k
    stat = float(np.sum((observed - expected) ** 2) / expected)
    return stat, float(stats.chi2.sf(stat, k - 3))


def check_normality(x, test: str = "anderson", alpha: float = 0.05) -> NormalityResult:
    """Test for normality.

    ``test`` is "anderson" (Anderson-Darling, default), "shapiro" (Shapiro-Wilk), "ks"
    (Kolmogorov-Smirnov with estimated parameters, Lilliefors), "chi2" (chi-square
    goodness of fit) - the four tests of the workflow article - or "k2" (D'Agostino-Pearson
    K^2 omnibus test, an addition of uccap).
    """
    x = np.asarray(x, float)
    if test == "anderson":
        stat, p = anderson_darling(x)
    elif test == "shapiro":
        stat, p = stats.shapiro(x)
    elif test == "ks":
        # Parameters are estimated, so use the Monte Carlo goodness-of-fit (Lilliefors) p-value.
        kw = dict(statistic="ks", n_mc_samples=999)
        try:
            res = stats.goodness_of_fit(stats.norm, x, rng=np.random.default_rng(0), **kw)
        except TypeError:  # SciPy < 1.15 names the seed argument random_state
            res = stats.goodness_of_fit(stats.norm, x, random_state=np.random.default_rng(0), **kw)
        stat, p = res.statistic, res.pvalue
    elif test == "chi2":
        stat, p = chi_square_gof(x)
    elif test == "k2":
        with warnings.catch_warnings():  # older SciPy warns below 20 observations; the manual says so
            warnings.simplefilter("ignore", UserWarning)
            stat, p = stats.normaltest(x)
    else:
        raise ValueError(f"unknown test {test!r}")
    return NormalityResult(test, float(stat), float(p), alpha)
