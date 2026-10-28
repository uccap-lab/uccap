import json
import math

import numpy as np
import pytest

from uccap import capability_study, hybrid_risk
from uccap.hybrid import (ModelNotAvailable, compute_features, model_available, model_path, predict, residual,
                          risk_level, set_model_path)
from uccap.hybrid.features import CATEGORICAL, NUMERIC

# The trained parameters are not part of this release; tests that need them run only when they are
# found (uccap.hybrid.set_model_path or $UCCAP_MODEL).
needs_model = pytest.mark.skipif(not model_available(), reason="trained UC-Cap parameters not available")


def test_missing_model_gives_clear_error(tmp_path, monkeypatch):
    import uccap
    from uccap.hybrid import model as M
    monkeypatch.delenv("UCCAP_MODEL", raising=False)
    monkeypatch.setattr(M, "MODEL_FILE", tmp_path / "absent.json")
    set_model_path(None)
    try:
        assert not model_available() and model_path() is None
        x = np.random.default_rng(0).normal(10, 0.02, 30)
        with pytest.raises(ModelNotAvailable, match="available from the authors on request"):
            hybrid_risk(x, 9.9, 10.1)
        assert compute_features(x, 9.9, 10.1)["A_Ppk"] > 0   # features need no model
        with pytest.warns(UserWarning, match="UC-Cap risk columns left out"):
            frame = uccap.batch_study({"A": x}, {"A": {"lsl": 9.9, "usl": 10.1}}).frame
        assert "risk_score" not in frame and frame.loc[0, "decision"] in ("ACCEPT", "REJECT")
        set_model_path(tmp_path / "absent.json")
        assert not model_available()
    finally:
        set_model_path(None)


@needs_model
def test_bundle_holds_parameters_only():
    b = json.loads(model_path().read_text())
    assert set(b) >= {"numeric_features", "numeric_median", "numeric_mean", "numeric_scale",
                      "categorical_features", "categorical_levels", "layers"}
    assert tuple(b["numeric_features"]) == NUMERIC and tuple(b["categorical_features"]) == CATEGORICAL
    assert [np.shape(L["weight"]) for L in b["layers"]] == [(64, 24), (64, 64), (1, 64)]
    text = model_path().read_text()
    assert "dim_id" not in text and "split_id" not in text


def test_normal_sample_features():
    x = np.random.default_rng(1).normal(0.01, 0.01, 20)
    f = compute_features(x, -0.05, 0.05)
    mu, s = x.mean(), x.std(ddof=1)
    assert f["A_normality"] == "Pass" and f["A_Best_Fit"] == "normal"
    assert f["A_Ppk"] == pytest.approx(min(0.05 - mu, mu + 0.05) / (3 * s))
    assert f["A_Pp"] == pytest.approx(0.1 / (6 * s))
    assert f["A_Dist_to_nearest_spec"] == pytest.approx(0.05 - mu)
    assert f["A_margin_to_threshold"] == pytest.approx(f["A_Ppk"] - 1.33)
    assert f["A_z_stat_133"] == pytest.approx((1.33 - f["A_Ppk"]) / f["A_SE_Ppk"])
    assert f["nominal"] == 0.0 and f["Spec_Width"] == pytest.approx(0.1)


def test_non_normal_sample_uses_empirical_percentiles():
    x = np.random.default_rng(0).lognormal(-4, 0.9, 20) - 0.02
    f = compute_features(x, -0.05, 0.1)
    assert f["A_normality"] == "Fail" and f["A_Best_Fit"] == "empirical"
    lo, mid, hi = np.quantile(x, [0.00135, 0.5, 0.99865])
    assert f["A_Ppk"] == pytest.approx(min((0.1 - mid) / (hi - mid), (mid + 0.05) / (mid - lo)))


def test_non_normal_sample_named_best_fit():
    from uccap.hybrid.features import NAMED_FITS
    from uccap.workflow.fitting import fit_distribution

    x = np.random.default_rng(1).weibull(1.5, 40) * 0.02 + 0.001
    f = compute_features(x, 0.0, 0.1)
    assert f["A_normality"] == "Fail" and f["A_Best_Fit"] in NAMED_FITS
    # Model inputs stay the empirical percentile indices, as in the training data.
    lo, mid, hi = np.quantile(x, [0.00135, 0.5, 0.99865])
    assert f["A_Ppk"] == pytest.approx(min((0.1 - mid) / (hi - mid), mid / (mid - lo)))
    # Percentile indices from the fitted distribution are added for reference.
    q = fit_distribution(x, f["A_Best_Fit"]).dist.ppf([0.00135, 0.5, 0.99865])
    assert f["Fit_Ppk"] == pytest.approx(min((0.1 - q[1]) / (q[2] - q[1]), q[1] / (q[1] - q[0])))
    assert f["Fit_Pp"] == pytest.approx(0.1 / (q[2] - q[0]))
    if model_available():
        assert "fit" in hybrid_risk(x, 0.0, 0.1).report()


def test_best_fit_normal_or_empirical_has_no_fit_indices():
    f = compute_features(np.random.default_rng(1).normal(0.01, 0.01, 20), -0.05, 0.05)
    assert math.isnan(f["Fit_Ppk"]) and math.isnan(f["Fit_Pp"])


def test_goodness_of_fit_size_is_about_five_percent():
    from uccap.hybrid.features import fit_adequate
    from uccap.workflow.fitting import fit_distribution

    rng = np.random.default_rng(5)
    gens = {"gamma": lambda: rng.gamma(3, 1, 30), "weibull": lambda: rng.weibull(1.5, 30),
            "logistic": lambda: rng.logistic(0, 1, 30), "lognormal": lambda: rng.lognormal(0, 0.5, 30)}
    for name, gen in gens.items():
        rej = 0
        for _ in range(200):
            x = gen()
            rej += not fit_adequate(x, fit_distribution(x, name))
        assert rej / 200 < 0.11, name


@needs_model
def test_prediction_is_anchor_plus_correction():
    x = np.random.default_rng(2).normal(0.0, 0.012, 20)
    r = hybrid_risk(x, -0.05, 0.05)
    assert r.correction == pytest.approx(residual([r.features])[0])
    assert r.p == pytest.approx(1 / (1 + math.exp(-(r.z + r.correction))))
    assert r.level == risk_level(r.p)
    assert r.score == pytest.approx(100 * r.p)
    assert "risk level" in r.report() and "risk score" in r.report()


@needs_model
def test_risk_falls_as_capability_improves():
    z = np.random.default_rng(0).standard_normal(20)
    ps = [hybrid_risk(0.004 + s * z, -0.05, 0.05).p for s in (0.025, 0.012, 0.006)]
    assert ps[0] > 0.7 > ps[1] > ps[2] and ps[2] < 0.1


@needs_model
def test_reproducible_and_accepts_study():
    x = np.random.default_rng(4).normal(10.0, 0.05, 20)
    a = hybrid_risk(x, 9.8, 10.2, rng=7)
    assert hybrid_risk(x, 9.8, 10.2, rng=7).p == a.p
    st = capability_study(x, lsl=9.8, usl=10.2, target=10.0)
    assert hybrid_risk(st, rng=7).p == pytest.approx(a.p)


@needs_model
def test_missing_values_use_training_medians():
    p = predict({"A_z_stat_133": 0.0})
    assert p.shape == (1,) and 0 < p[0] < 1


def test_upper_only_coded_like_training_data_and_lower_only_rejected():
    x = np.abs(np.random.default_rng(5).normal(0.02, 0.01, 20))
    f = compute_features(x, None, 0.2)
    mu, s = x.mean(), x.std(ddof=1)
    assert f["LSL"] == 0 and f["nominal"] == 0 and f["Upper_only"]
    if f["A_normality"] == "Pass":
        assert f["A_Ppk"] == pytest.approx((0.2 - mu) / (3 * s))
    assert f["A_Dist_to_nearest_spec"] == pytest.approx(min(mu, 0.2 - mu))  # to LSL = 0 or to USL
    # closer to 0 is better for an upper-only characteristic: low risk here
    if model_available():
        assert hybrid_risk(x, None, 0.2).level == "Low"
    with pytest.raises(ValueError, match="upper limit only"):
        compute_features(x, 0.0, None)
    with pytest.raises(ValueError):
        compute_features(x[:5], -5, 5)


def test_risk_levels():
    assert [risk_level(p) for p in (0.1, 0.4, 0.69, 0.7)] == ["Low", "Medium", "Medium", "High"]


def test_review_thresholds_and_custom_bands():
    from uccap.hybrid import review_thresholds, risk_level
    lo, hi = review_thresholds(10, 2, 0.5)          # Eq. 24: c_R/c_FA, 1 - c_R/c_FR
    assert (lo, hi) == pytest.approx((0.05, 0.75))
    a, b = review_thresholds(10, 2, 5)              # Eq. 25 fails: binary rule c_FR/(c_FA+c_FR)
    assert a == b == pytest.approx(2 / 12)
    assert risk_level(0.5) == "Medium" and risk_level(0.5, (0.6, 0.8)) == "Low"


def test_decision_layer_labels_and_details():
    from uccap.hybrid import mechanism

    base = {"LSL": 9.95, "USL": 10.05, "A_mean": 10.0, "A_normality": "Pass", "A_P_value": 0.5,
            "A_Skewness": 0.0, "A_Kurtosis": 3.0}
    ok = mechanism.signals({**base, "A_Pp": 2.0, "A_Ppk": 1.95, "A_SE_Ppk": 0.3})
    assert ok.active == () and mechanism.primary_driver("Low", ok) == "Acceptable"
    assert mechanism.action("Acceptable", "Low", ok) == "Accept"
    assert mechanism.primary_driver("Medium", ok) == "Latent model risk"
    assert mechanism.action("Latent model risk", "Medium", ok) == "Investigate latent risk"

    shift = mechanism.signals({**base, "A_mean": 10.02, "A_Pp": 2.0, "A_Ppk": 1.2, "A_SE_Ppk": 0.2})
    assert shift.active == ("shift",) and mechanism.primary_driver("High", shift) == "Mean shift to USL"
    assert mechanism.action("Mean shift to USL", "High", shift) == "Re-center"
    assert mechanism.displayed_level("Low", shift) == "Low"
    assert mechanism.primary_driver("Low", shift) == "Acceptable" and mechanism.action("Acceptable", "Low", shift) == "Accept"
    assert "Noted: the mean is shifted toward USL" in mechanism.reason_detail("Acceptable", "Low", shift, 0.1)

    marginal = mechanism.signals({**base, "A_Pp": 1.2, "A_Ppk": 1.19, "A_SE_Ppk": 0.2})
    assert marginal.active == () and mechanism.primary_driver("Medium", marginal) == "Latent model risk"

    mixed = mechanism.signals({**base, "A_mean": 10.02, "A_Pp": 0.9, "A_Ppk": 0.5, "A_SE_Ppk": 0.2})
    assert mixed.compound and mechanism.displayed_level("Low", mixed) == "Medium"
    driver = mechanism.primary_driver("High", mixed)
    assert driver == "Mixed mechanism"
    assert mechanism.action(driver, "High", mixed) == "Reduce sd + re-center"
    assert "combined response" in mechanism.action_detail(driver, "High", mixed)
    assert "Pp = 0.900" in mechanism.reason_detail(driver, "High", mixed, 0.9)

    skew = mechanism.signals({**base, "A_Pp": 2.0, "A_Ppk": 1.9, "A_SE_Ppk": 0.3, "A_normality": "Fail"})
    assert mechanism.primary_driver("Medium", skew) == "Skewed"
    assert mechanism.action("Skewed", "Medium", skew) == "Review distribution"


@needs_model
def test_decision_layer_on_hybrid_risk():
    x = np.random.default_rng(11).normal(10.012, 0.011, 20)
    r = hybrid_risk(x, lsl=9.95, usl=10.05, nominal=10.0)
    assert r.level == r.model_level == "Low"
    assert r.driver == "Acceptable" and r.action == "Accept"
    assert r.reason and r.action_detail.startswith("Accept or continue routine monitoring")
    text = r.report()
    assert "reason, in full" in text and "action, in full" in text


def test_decision_layer_reproduces_table_1_labels():
    """Consistent rows of Table 1 of the hybrid article (D019 and D021 are not: their Ppk
    does not follow from the listed mean, SD and limits). SE(Ppk) for the 20 A-side parts."""
    from uccap.hybrid import mechanism
    from uccap.risk.decision import se_cpk

    rows = [  # dim, LSL, USL, mean, SD, Ppk, normality, level, reason, action
        ("D003", 1.55, 1.75, 1.646, 0.0116, 2.774, "Pass", "Low", "Acceptable", "Accept"),
        ("D018", 9.77, 9.97, 9.845, 0.0132, 1.883, "Pass", "Low", "Acceptable", "Accept"),
        ("D002", 1.07, 1.27, 1.178, 0.0230, 1.389, "Fail", "Medium", "Skewed", "Review distribution"),
        ("D004", 2.17, 2.37, 2.278, 0.0231, 1.334, "Pass", "Medium", "Latent model risk", "Investigate latent risk"),
        ("D020", 0.05, 0.15, 0.100, 0.0132, 1.255, "Pass", "Medium", "Latent model risk", None),
        ("D010", 5.42, 5.62, 5.578, 0.0468, 0.302, "Pass", "High", "Mixed mechanism", "Reduce sd + re-center"),
        ("D015", 8.10, 8.30, 8.237, 0.0487, 0.431, "Pass", "High", "Mixed mechanism", "Reduce sd + re-center"),
    ]
    for dim, lsl, usl, mean, sd, ppk, norm, level, reason, act in rows:
        s = mechanism.signals({"LSL": lsl, "USL": usl, "A_mean": mean, "A_Pp": (usl - lsl) / (6 * sd),
                               "A_Ppk": ppk, "A_SE_Ppk": se_cpk(ppk, 20), "A_normality": norm,
                               "A_P_value": 0.01 if norm == "Fail" else 0.5, "A_Skewness": 0.0, "A_Kurtosis": 3.0})
        assert mechanism.displayed_level(level, s) == level, dim
        driver = mechanism.primary_driver(level, s)
        assert driver == reason, dim
        if act is not None:  # D020: the article gives Review distribution for a Latent model risk
            assert mechanism.action(driver, level, s) == act, dim


def test_unusable_inputs_give_clear_errors():
    with pytest.raises(ValueError, match="identical"):
        compute_features([1.0] * 20, lsl=0, usl=2)
    with pytest.raises(ValueError, match="not numbers"):
        compute_features(list(np.linspace(0.9, 1.1, 20)) + ["a"], lsl=0, usl=2)
    with pytest.raises(ValueError, match="USL > 0"):
        compute_features(np.linspace(-1.1, -0.9, 20), None, -0.5)


def test_upper_only_has_no_pp_and_no_pp_signals():
    y = np.random.default_rng(7).lognormal(np.log(0.02), 0.5, 30)
    f = compute_features(y, None, 0.12)
    assert math.isnan(f["A_Pp"]) and math.isnan(f["Fit_Pp"])
    if not model_available():
        return
    r = hybrid_risk(y, usl=0.12)
    assert r.driver in ("Acceptable", "Skewed", "Latent model risk")
    assert "Pp is not computed" in r.reason or "no mean-shift or spread signal" in r.reason or r.driver == "Skewed"
