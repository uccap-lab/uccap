"""Outlier screening before capability calculation.

The workflow paper begins with outlier detection before computing indices; its case
study (Section 6.1.1) uses the IQR rule, 1.5 x IQR beyond Q1/Q3. Tukey's IQR fences
with k = 1.5 are therefore the default here; Grubbs' test
is available for approximately normal data, and the adjusted boxplot
(Hubert & Vandervieren, 2008) for skewed data, where plain IQR fences would
flag the long tail itself. Flagged points are reported, never
silently removed: investigate them before deciding to exclude.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def iqr_fences(x, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences Q1 - k*IQR and Q3 + k*IQR."""
    q1, q3 = np.percentile(np.asarray(x, float), [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_outliers(x, k: float = 1.5) -> np.ndarray:
    """Boolean mask of points outside [Q1 - k*IQR, Q3 + k*IQR]."""
    x = np.asarray(x, float)
    lo, hi = iqr_fences(x, k)
    return (x < lo) | (x > hi)


def grubbs_outliers(x, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask from iterated two-sided Grubbs tests (assumes normality)."""
    x = np.asarray(x, float)
    mask = np.zeros(x.size, bool)
    while (~mask).sum() > 2:
        idx = np.flatnonzero(~mask)
        v = x[idx]
        n = v.size
        dev = np.abs(v - v.mean())
        j = int(np.argmax(dev))
        g = dev[j] / v.std(ddof=1)
        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
        if g <= g_crit:
            break
        mask[idx[j]] = True
    return mask


def medcouple(x) -> float:
    """Robust skewness measure (Brys, Hubert & Struyf, 2004); O(n^2), fine for n in the hundreds."""
    z = np.asarray(x, float)
    z = z - np.median(z)
    zp = np.sort(z[z >= 0])[::-1]  # descending
    zm = np.sort(z[z <= 0])[::-1]  # descending
    a, b = zp[:, None], zm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (a + b) / (a - b)
    n_zero = int(np.sum(z == 0))
    if n_zero:
        offset = zp.size - n_zero  # zeros sit at the end of zp and the start of zm
        for i in range(n_zero):
            for j in range(n_zero):
                h[offset + i, j] = np.sign(n_zero - 1 - i - j)
    return float(np.median(h))


def adjusted_fences(x, k: float = 1.5) -> tuple[float, float]:
    """Skewness-adjusted fences (Hubert & Vandervieren, 2008), using the medcouple."""
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mc = medcouple(x)
    if mc >= 0:
        return float(q1 - k * np.exp(-4 * mc) * iqr), float(q3 + k * np.exp(3 * mc) * iqr)
    return float(q1 - k * np.exp(-3 * mc) * iqr), float(q3 + k * np.exp(4 * mc) * iqr)


def adjusted_boxplot_outliers(x, k: float = 1.5) -> np.ndarray:
    """Skewness-adjusted IQR fences (Hubert & Vandervieren, 2008)."""
    x = np.asarray(x, float)
    lo, hi = adjusted_fences(x, k)
    return (x < lo) | (x > hi)


METHODS = {"iqr": iqr_outliers, "grubbs": grubbs_outliers, "adjusted": adjusted_boxplot_outliers}
FENCES = {"iqr": iqr_fences, "adjusted": adjusted_fences}
