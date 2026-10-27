"""Risk-calibrated capability approval with finite samples.

Source: Risk-calibrated process capability approval with finite samples
(Jiang & Yang, IJAMT 144:7189-7206, 2026, doi:10.1007/s00170-026-18284-2; arXiv:2603.14479).

Approval rule:  accept  iff  Cpk_hat >= C0 + k * SE(Cpk_hat).

* k = 0                       deterministic threshold
* k = z_{1-gamma}             lower confidence bound LCB_{1-gamma} >= C0
* k = z_{1-alpha}             accept iff P(Cpk_true < C0 | data) <= alpha
* alpha = 1 / (1 + lambda)    cost-sensitive (Bayes) rule, lambda = c_FA / c_FR
"""
from .decision import (RiskDecision, expected_loss, k_from_alpha, k_from_cost_ratio,
                       p_fail_normal, risk_decision, se_cpk)
from .bootstrap import bootstrap_p_fail
from .planning import SamplePlan, acceptance_probability, min_accepted, sample_size

__all__ = ["RiskDecision", "risk_decision", "se_cpk", "p_fail_normal", "k_from_alpha",
           "k_from_cost_ratio", "expected_loss", "bootstrap_p_fail",
           "sample_size", "SamplePlan", "min_accepted", "acceptance_probability"]
