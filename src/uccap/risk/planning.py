"""Sample-size planning for the risk-calibrated approval rule.

The rule accepts when Cpk_hat >= C0 + k * SE(Cpk_hat) with SE = sqrt((1/9 + Cpk_hat^2 / 2) / n)
(see :mod:`uccap.risk.decision`). Before a study this answers two questions:

* how large must the estimate be for n parts to be accepted (:func:`min_accepted`), and
* how many parts are needed so that a process with a given true Cpk is accepted with a
  given probability (:func:`sample_size`, :func:`acceptance_probability`).

The acceptance probability is simulated by default: for a centred normal process with
Cpk = C, Cpk_hat = (3C - |mean|) / (3 s) with mean ~ N(0, 1/n) and (n-1) s^2 ~ chi2(n-1), so
no full samples are drawn. A centred process is the conservative case (its estimate has the
largest downward bias). ``method="approx"`` uses Cpk_hat ~ N(Cpk, SE(Cpk, n)^2) instead,
which overstates the probability by a few points for small n.

All results assume normal data; for non-normal characteristics treat them as a lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from .decision import k_from_alpha, se_cpk

MIN_N = 8  # smallest sample uccap analyses


def _alpha(alpha: Optional[float], cost_ratio: Optional[float]) -> float:
    if alpha is not None and cost_ratio is not None:
        raise ValueError("give at most one of alpha or cost_ratio")
    if cost_ratio is not None:
        if cost_ratio <= 0:
            raise ValueError("cost_ratio must be positive")
        return 1 / (1 + cost_ratio)
    return 0.05 if alpha is None else alpha


def min_accepted(n: int, *, c0: float = 1.33, alpha: Optional[float] = None,
                 cost_ratio: Optional[float] = None) -> float:
    """Smallest estimate accepted with n parts: the root of (C - C0)^2 = k^2 (1/9 + C^2/2) / n.

    Returns ``inf`` when no estimate can be accepted (n <= k^2 / 2).
    """
    k = k_from_alpha(_alpha(alpha, cost_ratio))
    if n < 2:
        raise ValueError("n must be at least 2")
    if k == 0:
        return float(c0)
    a = 1 - k * k / (2 * n)
    b = c0 * c0 - k * k / (9 * n)
    if a <= 0:
        return math.inf if k > 0 else -math.inf
    disc = c0 * c0 - a * b
    root = math.sqrt(max(disc, 0.0))
    # k > 0: the root above C0; k < 0 (alpha > 0.5): the root below C0
    return (c0 + root) / a if k > 0 else (c0 - root) / a


def acceptance_probability(true_cpk: float, n: int, *, c0: float = 1.33, alpha: Optional[float] = None,
                           cost_ratio: Optional[float] = None, method: str = "simulate",
                           n_sim: int = 20000, rng=0) -> float:
    """Probability that a study of n parts is accepted when the true Cpk is ``true_cpk``."""
    t = min_accepted(n, c0=c0, alpha=alpha, cost_ratio=cost_ratio)
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    if method == "approx":
        return float(stats.norm.sf((t - true_cpk) / se_cpk(true_cpk, n)))
    if method != "simulate":
        raise ValueError(f"unknown method {method!r}")
    g = np.random.default_rng(rng)
    mean = np.abs(g.standard_normal(n_sim)) / math.sqrt(n)
    s = np.sqrt(g.chisquare(n - 1, n_sim) / (n - 1))
    return float(np.mean((3 * true_cpk - mean) / (3 * s) >= t))


@dataclass(frozen=True)
class SamplePlan:
    n: Optional[int]          # parts needed (None: not reachable within n_max)
    expected: float           # assumed true Cpk
    c0: float
    alpha: float
    power: float              # requested probability of acceptance
    achieved: float           # probability of acceptance at n
    threshold: float          # smallest estimate accepted at n
    cost_ratio: Optional[float] = None

    def report(self) -> str:
        rule = (f"cost ratio {self.cost_ratio:g} (alpha = {self.alpha:.3f})" if self.cost_ratio is not None
                else f"alpha = {self.alpha:g}")
        lines = [
            "Sample-size plan (risk-calibrated approval)",
            f"{'Requirement C0':>26}  {self.c0:g}",
            f"{'Risk rule':>26}  {rule}",
            f"{'Assumed true Cpk':>26}  {self.expected:g}",
            f"{'Wanted P(accept)':>26}  {self.power:.0%}",
        ]
        if self.n is None:
            lines.append(f"{'Parts needed':>26}  not reachable: the assumed Cpk is too close to C0")
        else:
            lines += [
                f"{'Parts needed':>26}  {self.n}",
                f"{'P(accept) with n parts':>26}  {self.achieved:.1%}",
                f"{'Smallest accepted Cpk':>26}  {self.threshold:.3f}",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


def sample_size(expected: float, *, c0: float = 1.33, alpha: Optional[float] = None,
                cost_ratio: Optional[float] = None, power: float = 0.8, n_max: int = 5000,
                method: str = "simulate", n_sim: int = 20000, rng=0) -> SamplePlan:
    """Parts needed so that a process with true Cpk ``expected`` is accepted with probability ``power``.

    Without ``alpha`` or ``cost_ratio``, alpha = 0.05. The answer is at least 8, the
    smallest sample uccap analyses.
    """
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    a = _alpha(alpha, cost_ratio)
    kw = dict(c0=c0, alpha=a, method=method, n_sim=n_sim, rng=rng)

    def ok(n):
        return acceptance_probability(expected, n, **kw) >= power

    found = None
    if ok(n_max):
        lo, hi = MIN_N, n_max  # acceptance probability increases with n when expected > c0
        if ok(lo):
            found = lo
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            found = hi
    n = found if found is not None else n_max
    return SamplePlan(found, float(expected), float(c0), a, float(power),
                      acceptance_probability(expected, n, **kw), min_accepted(n, c0=c0, alpha=a), cost_ratio)
