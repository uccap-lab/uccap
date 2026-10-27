"""Within (short-term) and overall (long-term) standard deviation estimators.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026), Section 5.
Moving-range estimators require the data in time order. Excluded observations can
be passed as NaN: moving ranges and successive differences that touch a NaN are
dropped, so the remaining ranges still compare true neighbours.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .constants import c4, d2, d4


def _moving_ranges(x: np.ndarray, w: int) -> np.ndarray:
    if w < 2:
        raise ValueError("window must be >= 2")
    if x.size < w:
        raise ValueError("need at least `window` observations")
    windows = np.lib.stride_tricks.sliding_window_view(x, w)
    windows = windows[~np.isnan(windows).any(axis=1)]
    if windows.shape[0] == 0:
        raise ValueError("no complete moving-range windows")
    return windows.max(axis=1) - windows.min(axis=1)


def sigma_overall(x) -> float:
    """Sample standard deviation with the N-1 denominator (used for Pp, Ppk)."""
    x = np.asarray(x, float)
    return float(np.std(x[~np.isnan(x)], ddof=1))


def sigma_amr(x, window: int = 2) -> float:
    """Average moving range: MR-bar / d2(w). Subgroup size 1."""
    return float(_moving_ranges(np.asarray(x, float), window).mean() / d2(window))


def sigma_mmr(x, window: int = 2) -> float:
    """Median moving range: MR-tilde / d4(w). Subgroup size 1, robust to isolated spikes."""
    return float(np.median(_moving_ranges(np.asarray(x, float), window)) / d4(window))


def sigma_srmssd(x) -> float:
    """Square root of the mean of squared successive differences, unbiased with c4(N).

    Use when consecutive points cannot be treated as a rational subgroup.
    The choice of c4(N) as the unbiasing constant is an implementation choice.
    """
    d = np.diff(np.asarray(x, float))
    d = d[~np.isnan(d)]
    m = d.size  # N - 1 when nothing is excluded
    if m < 1:
        raise ValueError("need at least 2 consecutive observations")
    return float(np.sqrt(np.sum(d ** 2) / (2 * m)) / c4(m + 1))


def _as_subgroups(subgroups: Sequence[Sequence[float]]) -> list[np.ndarray]:
    groups = [np.asarray(g, float) for g in subgroups]
    if any(g.size < 2 for g in groups):
        raise ValueError("every subgroup needs at least 2 observations")
    return groups


def sigma_rbar(subgroups) -> float:
    """R-bar / d2(n) for equal subgroup sizes n >= 2."""
    groups = _as_subgroups(subgroups)
    sizes = {g.size for g in groups}
    if len(sizes) != 1:
        raise ValueError("R-bar method needs equal subgroup sizes")
    return float(np.mean([np.ptp(g) for g in groups]) / d2(sizes.pop()))


def sigma_sbar(subgroups) -> float:
    """S-bar / c4(n) for equal subgroup sizes n >= 2."""
    groups = _as_subgroups(subgroups)
    sizes = {g.size for g in groups}
    if len(sizes) != 1:
        raise ValueError("S-bar method needs equal subgroup sizes")
    return float(np.mean([np.std(g, ddof=1) for g in groups]) / c4(sizes.pop()))


def sigma_pooled(subgroups, unbiased: bool = False) -> float:
    """Pooled standard deviation across subgroups (equal-variance assumption).

    With ``unbiased=True`` the estimate is divided by c4(df + 1).
    """
    groups = _as_subgroups(subgroups)
    df = sum(g.size - 1 for g in groups)
    sp = np.sqrt(sum((g.size - 1) * np.var(g, ddof=1) for g in groups) / df)
    return float(sp / c4(df + 1) if unbiased else sp)


def sigma_between(subgroups, sigma_within: float) -> float:
    """Standard deviation between subgroup means, net of within-subgroup noise.

    sqrt(max(0, s_xbar^2 - sigma_within^2 / n_bar)), with s_xbar the standard deviation of
    the subgroup means and n_bar the mean subgroup size (the usual between/within model).
    """
    groups = _as_subgroups(subgroups)
    if len(groups) < 2:
        raise ValueError("need at least 2 subgroups")
    means = np.array([g.mean() for g in groups])
    n_bar = np.mean([g.size for g in groups])
    return float(np.sqrt(max(0.0, np.var(means, ddof=1) - sigma_within ** 2 / n_bar)))


WITHIN_METHODS = {"amr": sigma_amr, "mmr": sigma_mmr, "srmssd": sigma_srmssd}
SUBGROUP_METHODS = {"pooled": sigma_pooled, "rbar": sigma_rbar, "sbar": sigma_sbar}
