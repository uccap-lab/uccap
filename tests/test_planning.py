import math

import numpy as np
import pytest

import uccap
from uccap.risk import acceptance_probability, min_accepted, risk_decision, sample_size


def test_min_accepted_matches_rule():
    # n = 32, lambda = 19: the smallest accepted estimate quoted in the manual
    t = min_accepted(32, cost_ratio=19)
    assert t == pytest.approx(1.691, abs=5e-4)
    assert risk_decision(t + 1e-6, n=32, cost_ratio=19).accept
    assert not risk_decision(t - 1e-3, n=32, cost_ratio=19).accept
    assert min_accepted(32, alpha=0.5) == 1.33
    assert math.isinf(min_accepted(2, alpha=0.001))  # n <= k^2/2: nothing passes


def test_acceptance_probability_against_full_simulation():
    rng = np.random.default_rng(3)
    c, n = 1.67, 40
    x = rng.normal(0, 1, (20000, n))
    m, s = x.mean(1), x.std(1, ddof=1)
    est = np.minimum(3 * c - m, m + 3 * c) / (3 * s)
    mc = np.mean(est >= min_accepted(n))
    assert acceptance_probability(c, n) == pytest.approx(mc, abs=0.015)
    assert acceptance_probability(c, n, method="approx") > acceptance_probability(c, n)


def test_sample_size_plan():
    plan = uccap.sample_size(1.67)
    assert 70 <= plan.n <= 90 and plan.achieved >= 0.8
    assert acceptance_probability(1.67, plan.n - 1) < 0.8
    assert sample_size(2.0).n < plan.n < sample_size(1.5).n
    assert sample_size(1.67, power=0.9).n > plan.n
    assert sample_size(4.0).n == 8  # never below the smallest analysable sample
    assert sample_size(1.30).n is None and "not reachable" in sample_size(1.30).report()
    assert "Parts needed" in plan.report()
    with pytest.raises(ValueError):
        sample_size(1.67, alpha=0.05, cost_ratio=19)


def test_planning_plot():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from uccap.plot import planning_plot
    ax = planning_plot((1.67,), n_range=(8, 60))
    assert len(ax.lines) >= 2
