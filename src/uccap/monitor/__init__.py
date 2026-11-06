"""Operational acceptability monitoring after approval.

Source: Bayesian operational acceptability monitoring for manufacturing process control
(Jiang & Yang, IJAMT, accepted 2026).

Sequential Bayesian operational acceptability monitoring (SBOAM): every completed subgroup updates an
exponentially discounted Normal-Inverse-Gamma posterior of the process mean and standard deviation, and
the posterior acceptability probability

    pi_t = P(q(theta_t) >= q0 | data up to subgroup t)

places the process in one of three zones: continue (pi_t >= gamma_A), monitor, or intervene
(pi_t <= gamma_R, from subgroup t_min on). q(theta) is the specification-conformance probability.
"""
from .batch import MonitorBatchResult, batch_monitor, monitor_notes
from .sboam import (CATEGORIES, ZONES, MonitorPrior, MonitorResult, conformance, cpk_for_q0, monitor_study,
                    subgroups)

__all__ = ["monitor_study", "MonitorResult", "MonitorPrior", "batch_monitor", "MonitorBatchResult", "conformance",
           "cpk_for_q0", "subgroups", "monitor_notes", "ZONES", "CATEGORIES"]
