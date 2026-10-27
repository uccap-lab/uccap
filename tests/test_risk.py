import numpy as np
import pytest

from uccap import capability_study, risk_decision
from uccap.risk import expected_loss, k_from_alpha, k_from_cost_ratio, se_cpk


def test_k_values_from_paper():
    assert k_from_cost_ratio(1) == pytest.approx(0.0, abs=1e-12)
    assert k_from_cost_ratio(19) == pytest.approx(1.645, abs=1e-3)
    assert k_from_cost_ratio(99) == pytest.approx(2.326, abs=1e-3)
    assert k_from_alpha(0.05) == pytest.approx(1.645, abs=1e-3)


def test_se_formula():
    assert se_cpk(1.33, 32) == pytest.approx(((1 / 9 + 1.33 ** 2 / 2) / 32) ** 0.5)


def test_margin_equivalent_to_probability_rule():
    for cpk_hat in np.linspace(1.2, 2.0, 41):
        d = risk_decision(float(cpk_hat), 32, c0=1.33, cost_ratio=19)
        assert d.accept == (cpk_hat >= d.margin - 1e-12)


def test_calibration_only_flips_near_threshold_accepts():
    d = risk_decision(1.40, 32, c0=1.33, cost_ratio=10)
    assert d.threshold_accept and not d.accept
    assert risk_decision(2.2, 32, c0=1.33, cost_ratio=10).accept


def test_cost_ratio_one_is_threshold_rule():
    assert risk_decision(1.34, 32, c0=1.33, cost_ratio=1).accept
    assert not risk_decision(1.32, 32, c0=1.33, cost_ratio=1).accept


def test_bootstrap_path_for_nonnormal():
    x = np.random.default_rng(5).lognormal(0, 0.4, 60)
    study = capability_study(x, usl=8, force_method="percentile")
    d = risk_decision(study, c0=1.33, alpha=0.05, n_boot=200, rng=0)
    assert d.method == "bootstrap" and 0 <= d.p_fail <= 1


def test_expected_loss():
    assert expected_loss([0.1, 0.9], [1, 0], 10) == pytest.approx(10 * 0.1 + 0.1)


def test_false_accept_rate_controlled_in_simulation():
    # Simulation design of Section 5: symmetric limits T = 4, mu = 0.
    rng = np.random.default_rng(6)
    true_cpk, n, c0 = 1.25, 32, 1.33
    sigma = 4 / (3 * true_cpk)
    fa_threshold = fa_risk = 0
    for _ in range(2000):
        x = rng.normal(0, sigma, n)
        s = x.std(ddof=1)
        c = min((4 - x.mean()) / (3 * s), (x.mean() + 4) / (3 * s))
        fa_threshold += c >= c0
        fa_risk += risk_decision(c, n, c0=c0, alpha=0.05).accept
    assert fa_risk < fa_threshold
    assert fa_risk / 2000 < 0.08


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="cost_ratio"):
        risk_decision(1.5, n=30, cost_ratio=-1)
    with pytest.raises(ValueError, match="n must"):
        risk_decision(1.5, n=1, alpha=0.05)
    with pytest.raises(ValueError, match="finite"):
        risk_decision(float("nan"), n=30, alpha=0.05)
    with pytest.raises(ValueError, match="n_boot"):
        risk_decision(1.5, n=30, alpha=0.05, n_boot=10)
    # NumPy scalars are numbers too
    assert risk_decision(np.float32(1.5), n=30, alpha=0.05).cpk_hat == pytest.approx(1.5)
