import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import uccap  # noqa: E402
from uccap import plot  # noqa: E402


def test_study_and_decision_plots():
    x = np.random.default_rng(1).normal(10, 0.4, 40)
    study = uccap.capability_study(x, lsl=8.5, usl=11.5, target=10)
    dec = uccap.risk_decision(study, c0=1.33, cost_ratio=19)
    assert len(study.plot().axes) == 4
    assert len(study.plot(dec, title="t").axes) == 5
    assert dec.plot().get_xlim()[0] < dec.c0


def test_nonnormal_bootstrap_and_batch_plots():
    y = np.random.default_rng(7).lognormal(np.log(0.02), 0.5, 60)
    study = uccap.capability_study(y, usl=0.12)
    dec = uccap.risk_decision(study, c0=1.33, alpha=0.05, rng=0)
    plot.decision_plot(dec, study, n_boot=200)
    res = uccap.batch_study({"A": y, "B": y * 2}, {"A": {"usl": 0.12}, "B": {"usl": 0.12}}, c0=1.33,
                            cost_ratio=19, n_boot=200)
    ax = res.plot()
    assert {t.get_text() for t in ax.get_yticklabels()} == {"A", "B"}


def test_figures_by_category():
    x = np.random.default_rng(42).normal(4.625, 0.023, 32)
    x[20] = 4.70
    study = uccap.capability_study(x, lsl=4.52, usl=4.72, target=4.62)
    lo, hi = study.outlier_fences
    assert study.outlier_screen == "iqr" and lo < hi < 4.70 and study.outlier_mask[20]
    dec = uccap.risk_decision(study, cost_ratio=19)
    for kind, n_axes in (("outliers", 2), ("stability", 2), ("capability", 2), ("approval", 2)):
        assert len(study.plot(dec, kind=kind).axes) == n_axes
    assert len(study.plot(kind="approval").axes) == 2  # decision computed on the fly
    if uccap.hybrid.model_available():
        assert len(uccap.hybrid_risk(study).plot().axes) >= 2
    with pytest.raises(ValueError):
        study.plot(kind="pie")


def test_required_plot_matches_min_accepted():
    from uccap.risk.planning import min_accepted
    dec = uccap.risk_decision(1.603, 32, cost_ratio=19)
    ax = plot.required_plot(dec)
    xs, ys = ax.lines[0].get_data()
    i = list(xs).index(32)
    assert abs(ys[i] - min_accepted(32, cost_ratio=19)) < 1e-12


def test_grubbs_screen_has_no_fences():
    x = np.random.default_rng(3).normal(10, 0.01, 30)
    study = uccap.capability_study(x, lsl=9.95, usl=10.05, outlier_method="grubbs")
    assert study.outlier_fences is None
    plot.outlier_figure(study)
