from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


def se_cpk(cpk_hat: float, n: int) -> float:
    """Delta-method standard error: sqrt((1/9 + Cpk^2 / 2) / n) (paper, Appendix A)."""
    return math.sqrt((1 / 9 + cpk_hat ** 2 / 2) / n)


def p_fail_normal(cpk_hat: float, se: float, c0: float) -> float:
    """Approximate P(Cpk_true < C0 | data) = Phi((C0 - Cpk_hat) / SE)."""
    return float(stats.norm.cdf((c0 - cpk_hat) / se))


def k_from_alpha(alpha: float) -> float:
    """Margin multiplier for a tolerable failure probability alpha: k = z_{1-alpha}."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return float(stats.norm.ppf(1 - alpha))


def k_from_cost_ratio(cost_ratio: float) -> float:
    """Bayes margin for lambda = c_FA / c_FR: alpha = 1/(1+lambda), k = z_{1-alpha}."""
    if cost_ratio <= 0:
        raise ValueError("cost_ratio must be positive")
    return k_from_alpha(1 / (1 + cost_ratio))


def expected_loss(p_fail, accept, cost_ratio: float) -> float:
    """Empirical risk proxy: sum of lambda * p * delta + (1 - p) * (1 - delta).

    False acceptance costs ``cost_ratio`` and false rejection costs 1.
    """
    p = np.asarray(p_fail, float)
    d = np.asarray(accept, float)
    return float(np.sum(cost_ratio * p * d + (1 - p) * (1 - d)))


@dataclass(frozen=True)
class RiskDecision:
    accept: bool
    cpk_hat: float
    se: float
    c0: float
    k: float
    p_fail: float
    alpha: float
    method: str  # "analytic" or "bootstrap"
    cost_ratio: Optional[float] = None
    n: int = 0
    index_name: str = "Cpk"

    @property
    def margin(self) -> float:
        """Required capability: C0 + k * SE."""
        return self.c0 + self.k * self.se

    @property
    def threshold_accept(self) -> bool:
        """What the deterministic Cpk_hat >= C0 rule would decide."""
        return self.cpk_hat >= self.c0

    def report(self) -> str:
        """Formatted text report of the decision."""
        from ..report import decision_report
        return decision_report(self)

    def __str__(self) -> str:
        return self.report()

    def plot(self, study=None, ax=None):
        """Uncertainty in the estimate against C0 (needs matplotlib); pass the study for bootstrap decisions."""
        from ..plot import decision_plot
        return decision_plot(self, study, ax)


def risk_decision(
    study_or_cpk,
    n: Optional[int] = None,
    *,
    c0: float = 1.33,
    alpha: Optional[float] = None,
    cost_ratio: Optional[float] = None,
    method: str = "auto",
    n_boot: int = 1000,
    rng=0,
) -> RiskDecision:
    """Risk-calibrated approval decision.

    ``study_or_cpk`` is a :class:`uccap.workflow.CapabilityStudy` or a Cpk estimate
    (then pass ``n``). Give exactly one of ``alpha`` or ``cost_ratio``.
    ``method="auto"`` uses the analytic rule for normal studies and the bootstrap
    failure probability for non-normal studies. ``rng`` seeds the bootstrap (0 by default,
    as in ``batch_study``, so the same data give the same decision); pass None for a fresh seed.
    """
    if (alpha is None) == (cost_ratio is None):
        raise ValueError("give exactly one of alpha or cost_ratio")
    if alpha is None:
        if not cost_ratio > 0:
            raise ValueError("cost_ratio must be positive")
        alpha = 1 / (1 + cost_ratio)
    k = k_from_alpha(alpha)
    if method == "bootstrap" or method == "auto":
        if int(n_boot) != n_boot or n_boot < 100:
            raise ValueError("n_boot must be an integer of at least 100")

    study = None
    if isinstance(study_or_cpk, numbers.Real) and not isinstance(study_or_cpk, bool):
        if n is None:
            raise ValueError("n is required when passing a Cpk value")
        if int(n) != n or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if not math.isfinite(study_or_cpk):
            raise ValueError("the Cpk estimate must be a finite number")
        cpk_hat, index_name = float(study_or_cpk), "Cpk"
        if method == "auto":
            method = "analytic"
    else:
        study = study_or_cpk
        cpk_hat, n, index_name = study.cpk, study.n, study.cpk_name
        if method == "auto":
            method = "analytic" if study.method == "normal" else "bootstrap"

    se = se_cpk(cpk_hat, n)
    if method == "analytic":
        p = p_fail_normal(cpk_hat, se, c0)
    elif method == "bootstrap":
        if study is None:
            raise ValueError("bootstrap needs a CapabilityStudy")
        from .bootstrap import bootstrap_p_fail
        p = bootstrap_p_fail(study, c0, n_boot=n_boot, rng=rng)
    else:
        raise ValueError(f"unknown method {method!r}")
    return RiskDecision(bool(p <= alpha), cpk_hat, se, c0, k, p, alpha, method, cost_ratio, int(n), index_name)
