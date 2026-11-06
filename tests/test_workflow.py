import math

import numpy as np
import pytest

from uccap.workflow import (Spec, capability_study, cpk, cpm, cpmk, sigma_amr, sigma_mmr,
                            sigma_overall, sigma_pooled, sigma_rbar, sigma_sbar, check_normality)
from uccap.workflow.constants import c4
from uccap.workflow.percentile import percentile_indices
from uccap.workflow.outliers import grubbs_outliers, iqr_outliers
from scipy import stats


def test_c4_matches_table():
    assert c4(2) == pytest.approx(0.7979, abs=1e-4)
    assert c4(5) == pytest.approx(0.9400, abs=1e-4)


def test_symmetric_indices():
    spec = Spec(lsl=-4, usl=4, target=0)
    assert cpk(1.0, 1.0, spec) == pytest.approx(1.0)
    assert cpm(0.0, 1.0, spec) == pytest.approx(8 / 6)
    assert cpmk(1.0, 1.0, spec) == pytest.approx(3 / (3 * math.sqrt(2)))


def test_unilateral_indices():
    spec = Spec(usl=10)
    assert cpk(4.0, 1.0, spec) == pytest.approx(2.0)


def test_asymmetric_reduces_when_on_target():
    spec = Spec(lsl=0, usl=10, target=4)
    # on target: Cpk* = d*/(3 sigma) with d* = 4
    assert cpk(4.0, 1.0, spec) == pytest.approx(4 / 3)


def test_asymmetric_paper_and_pearn_chen_forms():
    # du = 2, dl = 1, mean 0.5 above target, sigma 1
    paper, pc = Spec(lsl=-1, usl=2, target=0), Spec(lsl=-1, usl=2, target=0, asymmetric="pearn-chen")
    assert cpk(0.5, 1.0, paper) == pytest.approx(0.5 / 3)          # Eq. (5): min(dl, du) - |T - mu|
    assert cpk(0.5, 1.0, pc) == pytest.approx(0.75 / 3)            # A* = 0.25
    assert cpm(0.5, 1.0, paper) == pytest.approx(1 / (3 * np.sqrt(1.25)))
    assert cpmk(0.5, 1.0, paper) == pytest.approx((0.5 / 3) / np.sqrt(1.25))
    assert cpk(-1.5, 1.0, paper) == 0.0                            # beyond the lower tolerance
    for spec in (paper, pc):                                       # both reduce to d*/(3 sigma) on target
        assert cpk(0.0, 1.0, spec) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        Spec(lsl=0, usl=1, target=0.3, asymmetric="other")


def test_sigma_estimators_agree_on_normal_data():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, 5000)
    for f in (sigma_amr, sigma_mmr, sigma_overall):
        assert f(x) == pytest.approx(2, rel=0.05)
    groups = x.reshape(-1, 5)
    for f in (sigma_rbar, sigma_sbar, sigma_pooled):
        assert f(groups) == pytest.approx(2, rel=0.05)


def test_ad_test_matches_scipy_critical_values():
    rng = np.random.default_rng(2)
    x = rng.normal(size=50)
    ours = check_normality(x)
    try:  # SciPy >= 1.17 asks for a p-value method; older versions reject the argument
        ref = stats.anderson(x, method="interpolate")
    except TypeError:
        ref = stats.anderson(x)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-6)
    assert check_normality(rng.exponential(size=200)).p_value < 0.01


def test_percentile_indices_equal_normal_indices_for_normal_dist():
    spec = Spec(lsl=-4, usl=4, target=0)
    out = percentile_indices(stats.norm(0.5, 1), spec)
    assert out["CNpk"] == pytest.approx(cpk(0.5, 1, spec), rel=1e-4)


def test_outliers():
    x = np.r_[np.random.default_rng(3).normal(size=40), 12.0]
    assert iqr_outliers(x)[-1] and grubbs_outliers(x)[-1]
    assert grubbs_outliers(x).sum() == 1


def test_pipeline_branches():
    rng = np.random.default_rng(4)
    normal = capability_study(rng.normal(4.62, 0.02, 32), lsl=4.52, usl=4.72, target=4.62)
    assert normal.method == "normal" and normal.cpk == normal.indices["Ppk"]
    skewed = capability_study(rng.lognormal(0, 0.6, 200), usl=12)
    assert skewed.method == "percentile" and skewed.fit is not None
    assert "CNpk" in skewed.indices


from uccap.workflow.fitting import ALL_CANDIDATES, DEFAULT_CANDIDATES, best_fit, fit_distribution


def test_every_candidate_fits_positive_skewed_data():
    x = np.random.default_rng(7).lognormal(1, 0.4, 150)
    assert {f.name for f in best_fit(x)} == set(DEFAULT_CANDIDATES)
    assert {f.name for f in best_fit(x, "all")} == set(ALL_CANDIDATES)


@pytest.mark.parametrize("name", ALL_CANDIDATES)
def test_ppf_inverts_cdf(name):
    x = np.random.default_rng(8).gamma(4, 1, 200)
    fit = fit_distribution(x, name)
    q = np.array([0.00135, 0.5, 0.99865])
    assert np.allclose(fit.dist.cdf(fit.dist.ppf(q)), q, atol=1e-5)


def test_aicc_recovers_true_family():
    rng = np.random.default_rng(9)
    assert best_fit(rng.gumbel(10, 2, 400))[0].name == "lev"
    assert best_fit(rng.weibull(1.5, 400) * 3)[0].name in {"weibull", "weibull3", "gamma", "gamma3"}


def test_positive_only_families_skip_negative_data():
    x = np.random.default_rng(10).normal(0, 1, 60)
    names = {f.name for f in best_fit(x)}
    assert "weibull" not in names and "boxcox" not in names and "normal" in names


from uccap.workflow.outliers import adjusted_boxplot_outliers, medcouple


def test_medcouple_sign_and_symmetry():
    rng = np.random.default_rng(11)
    assert abs(medcouple(rng.normal(size=401))) < 0.2
    assert medcouple(rng.lognormal(0, 0.8, 400)) > 0.2
    assert medcouple([1, 2, 3, 3, 3, 4, 5]) == pytest.approx(0.0)


def test_adjusted_boxplot_flags_fewer_points_on_skewed_data():
    x = np.random.default_rng(12).lognormal(0, 0.6, 300)
    assert adjusted_boxplot_outliers(x).sum() < iqr_outliers(x).sum()


def test_exclude_flagged_outliers():
    rng = np.random.default_rng(13)
    x = rng.normal(10, 0.1, 40)
    x[[7, 20]] = [11.0, 9.0]
    full = capability_study(x, lsl=9.5, usl=10.5)
    assert full.outlier_mask[[7, 20]].all()
    clean = full.without_outliers()
    assert clean.n == 40 - full.outlier_mask.sum() and clean.n_excluded == full.outlier_mask.sum()
    assert clean.cpk > full.cpk
    same = capability_study(x, lsl=9.5, usl=10.5, exclude=[8, 21])
    assert same.n == 38 and "excluded" in same.report()


def test_moving_ranges_do_not_bridge_excluded_points():
    x = np.array([1.0, 2.0, 100.0, 3.0, 4.0])
    gapped = np.where([False, False, True, False, False], np.nan, x)
    # Remaining neighbour pairs: (1,2) and (3,4); the pair (2,3) is not adjacent in time.
    assert sigma_amr(gapped) == pytest.approx(1.0 / 1.128379)


def test_exclude_validates_numbers():
    with pytest.raises(ValueError):
        capability_study(np.arange(20.0), usl=30, exclude=[0])


from uccap.workflow.pipeline import DataError


def test_data_errors_are_friendly():
    with pytest.raises(DataError, match="at least 8"):
        capability_study([1.0, 2.0, 3.0], usl=5)
    with pytest.raises(DataError, match="identical"):
        capability_study(np.ones(20), usl=5)
    with pytest.raises(DataError, match="shape"):
        capability_study(np.zeros((10, 3)), usl=5)


def test_missing_and_text_entries_are_dropped_and_reported():
    x = list(np.random.default_rng(14).normal(0, 1, 30)) + [None, "", "bad", "1.5 mm", np.nan]
    s = capability_study(x, lsl=-5, usl=5)
    assert s.n == 31 and s.n_missing == 4
    assert any("non-numeric" in w for w in s.warnings)
    assert "missing" in s.report()


def test_missing_values_break_moving_ranges():
    x = np.array([1.0, 2.0, np.nan, 3.0, 4.0] * 4)
    s = capability_study(x, lsl=-10, usl=20)
    assert s.sigma_within == pytest.approx(sigma_amr(x))


def test_sorted_data_and_unknown_time_order():
    x = np.sort(np.random.default_rng(15).normal(0, 1, 30))
    assert any("sorted" in w for w in capability_study(x, lsl=-4, usl=4).warnings)
    s = capability_study(x, lsl=-4, usl=4, time_ordered=False)
    assert s.cpk_name == "Ppk" and math.isnan(s.sigma_within)


def test_coarse_resolution_warning():
    x = np.round(np.random.default_rng(16).normal(1, 0.05, 40), 1)
    assert any("resolution" in w for w in capability_study(x, lsl=0.8, usl=1.2).warnings)


def test_empirical_percentiles():
    x = np.random.default_rng(17).lognormal(0, 0.5, 2000)
    fit = capability_study(x, usl=8)
    emp = capability_study(x, usl=8, percentile_method="empirical")
    assert emp.percentile_method == "empirical" and emp.cpk == pytest.approx(fit.cpk, rel=0.15)
    small = capability_study(x[:40], usl=8, force_method="percentile", percentile_method="empirical")
    assert any("empirical" in w for w in small.warnings)


def test_approval_uses_overall_sigma_and_ratio_warning():
    rng = np.random.default_rng(18)
    drift = np.linspace(-0.3, 0.3, 40) + rng.normal(0, 0.05, 40)  # slow drift: within << overall
    s = capability_study(drift, lsl=-1, usl=1)
    assert s.cpk_name == "Ppk" and s.cpk == s.indices["Ppk"]
    assert s.sigma_ratio < 0.75 and any("within / overall" in w for w in s.warnings)
    # The within-sigma estimator does not change the approval index.
    assert capability_study(drift, lsl=-1, usl=1, within="mmr").cpk == pytest.approx(s.cpk)


def test_outlier_fence_coefficient():
    x = np.r_[np.random.default_rng(4).normal(10, 0.01, 40), 10.035, 10.06]
    default = capability_study(x, 9.9, 10.1)
    wide = capability_study(x, 9.9, 10.1, outlier_k=3.0)
    narrow = capability_study(x, 9.9, 10.1, outlier_k=1.0)
    assert wide.outlier_mask.sum() < default.outlier_mask.sum() <= narrow.outlier_mask.sum()
    assert wide.outlier_mask[-1]                       # the far-out point is still flagged
    assert "fences k = 3" in wide.report() and "fences k" not in default.report()
    assert wide.without_outliers().settings["outlier_k"] == 3.0
    with pytest.raises(ValueError):
        capability_study(x, 9.9, 10.1, outlier_k=0)


def test_moving_range_span_and_constants():
    from uccap.workflow.constants import d2, d4
    x = np.random.default_rng(0).normal(10, 0.01, 40)
    for w in (2, 3, 5, 10):
        mr = np.lib.stride_tricks.sliding_window_view(x, w)
        r = mr.max(1) - mr.min(1)
        s = capability_study(x, lsl=9.95, usl=10.05, mr_span=w)
        assert abs(s.sigma_within - r.mean() / d2(w)) < 1e-12
        s = capability_study(x, lsl=9.95, usl=10.05, within="mmr", mr_span=w)
        assert abs(s.sigma_within - np.median(r) / d4(w)) < 1e-12
    assert "span 3" in capability_study(x, lsl=9.95, usl=10.05, mr_span=3).report()
    with pytest.raises(ValueError):
        capability_study(x, lsl=9.95, usl=10.05, mr_span=11)


def test_subgroup_methods_and_between_sigma():
    rng = np.random.default_rng(1)
    g = rng.normal(10, 0.01, (10, 5)) + rng.normal(0, 0.02, (10, 1))  # drift between subgroups
    x = g.ravel()
    pooled = capability_study(x, lsl=9.9, usl=10.1, subgroups=list(g))
    rbar = capability_study(x, lsl=9.9, usl=10.1, subgroups=list(g), within="rbar")
    sbar = capability_study(x, lsl=9.9, usl=10.1, subgroups=list(g), within="sbar")
    assert pooled.sigma_method == "pooled" and rbar.sigma_method == "rbar" and sbar.sigma_method == "sbar"
    for s in (pooled, rbar, sbar):
        assert 0.006 < s.sigma_within < 0.014
    means = g.mean(1)
    expect = np.sqrt(max(0, means.var(ddof=1) - pooled.sigma_within ** 2 / 5))
    assert abs(pooled.sigma_between - expect) < 1e-12 and pooled.sigma_between > 0.01
    assert abs(pooled.sigma_bw - np.hypot(pooled.sigma_between, pooled.sigma_within)) < 1e-12
    assert "Sigma between" in pooled.report()
    with pytest.raises(ValueError):
        capability_study(x, lsl=9.9, usl=10.1, subgroups=list(g), within="mmr")


def test_ppm_lower_bound_other_tests_and_ppm_index():
    from scipy import stats
    x = np.random.default_rng(42).normal(4.625, 0.023, 32)
    s = capability_study(x, lsl=4.52, usl=4.72, target=4.62)
    exp = 1e6 * stats.norm.sf(4.72, s.mean, s.sigma_overall)
    assert abs(s.ppm["overall"]["above"] - exp) < 1e-6 and s.ppm["observed"]["total"] == 0
    assert set(s.normality_other) == {"shapiro", "k2"}
    se = np.sqrt((1 / 9 + s.cpk ** 2 / 2) / s.n)
    assert abs(s.lower_bound() - (s.cpk - 1.6448536269514722 * se)) < 1e-9
    assert s.indices["Ppm"] < s.indices["Pp"] and "Ppm" in s.report()
    one = capability_study(x, usl=4.72)
    assert one.ppm["observed"]["below"] == 0 and "Parts per million" in one.report()


def test_chi_square_goodness_of_fit():
    from uccap.workflow.normality import check_normality
    rng = np.random.default_rng(3)
    assert check_normality(rng.normal(0, 1, 400), "chi2").p_value > 0.01
    assert check_normality(rng.exponential(1, 400), "chi2").p_value < 0.001
    assert check_normality(rng.normal(0, 1, 50), "k2").test == "k2"


def test_prediction_and_tolerance_intervals():
    rng = np.random.default_rng(5)
    s = capability_study(rng.normal(10, 1, 50), 6, 14)
    lo, hi = s.prediction_interval(0.95)
    assert hi - lo == pytest.approx(2 * stats.t.ppf(0.975, 49) * s.sigma_overall * np.sqrt(1 + 1 / 50))
    tlo, thi = s.tolerance_interval(0.99, 0.95)
    assert thi - tlo == pytest.approx(2 * 3.13 * s.sigma_overall, rel=0.02)   # tabulated k for n = 50
    assert tlo < lo and thi > hi


def test_infinite_and_ambiguous_comma_entries_are_not_read():
    x = list(np.random.default_rng(0).normal(10, 0.1, 20)) + [np.inf, "4,618", "1,234.5"]
    s = capability_study(x, lsl=9, usl=11)
    assert s.n == 21  # "1,234.5" is read with a thousands comma
    assert "2 non-numeric entries ignored" in s.warnings[0]
    assert "thousands separator" in s.warnings[1]


def test_normality_alpha_is_checked():
    with pytest.raises(ValueError, match="alpha"):
        capability_study(np.random.default_rng(0).normal(10, 0.1, 20), lsl=9, usl=11, alpha=0)
