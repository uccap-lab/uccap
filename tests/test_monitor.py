"""Operational acceptability monitoring (uccap.monitor)."""
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

import uccap
from uccap.monitor import (CATEGORIES, MonitorPrior, batch_monitor, conformance, cpk_for_q0, monitor_study,
                           subgroups)

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "templates" / "multiple_dimensions.csv"
SIM = dict(prior=(0.0, 1e-3, 2.0, 1.0), n_draws=500)


def series(mu_path, rng=1, n=5):
    gen = np.random.default_rng(rng)
    mu_path = np.asarray(mu_path, float)
    return (mu_path[:, None] + gen.standard_normal((len(mu_path), n))).ravel()


def test_conformance_and_q0():
    assert conformance(0.0, 1.0, -3, 3) == pytest.approx(0.9973, abs=1e-4)
    assert conformance(0.0, 1.0, -math.inf, 3) == pytest.approx(0.99865, abs=1e-5)
    assert cpk_for_q0(0.997) == pytest.approx(0.9892, abs=1e-4)          # Eq. (17)
    assert cpk_for_q0(2 * 0.99865010 - 1) == pytest.approx(1.0, abs=1e-6)


def test_tolerance_aware_prior():
    p = MonitorPrior.tolerance_aware(9.9, 10.1, 10.0)
    assert (p.mu0, p.kappa0, p.alpha0) == (10.0, 1e-3, 2.0)
    assert p.beta0 == pytest.approx((0.2 / 8) ** 2)
    assert MonitorPrior.tolerance_aware(0, 1).mu0 == 0.5


def test_update_matches_closed_form():
    """Without forgetting the posterior is the conjugate one of all data (Eqs. 22-26)."""
    x = series(np.zeros(12), rng=3)
    r = monitor_study(x, -4, 4, forgetting=1.0, prior=(0.0, 1e-3, 2.0, 1.0))
    last = r.frame.iloc[-1]
    n, m = x.size, x.mean()
    ss = ((x - m) ** 2).sum()
    assert last["weight"] == n
    assert last["mu_post"] == pytest.approx(n * m / (1e-3 + n))
    beta = 1.0 + ss / 2 + 1e-3 * n * m * m / (2 * (1e-3 + n))
    assert last["sigma_post"] == pytest.approx(math.sqrt(beta / (2.0 + n / 2 - 1)))


def test_forgetting_discounts_old_subgroups():
    r = monitor_study(series(np.zeros(30)), -4, 4, forgetting=0.9, **SIM)
    w = r.frame["weight"].to_numpy()
    assert w[0] == 5 and w[1] == pytest.approx(5 + 0.9 * 5)
    assert w[-1] == pytest.approx(5 * (1 - 0.9 ** 30) / 0.1)


def test_zones_and_t_min():
    stable = monitor_study(series(np.zeros(60)), -4, 4, **SIM)
    assert stable.zone == "continue" and stable.intervention is None
    assert stable.category == CATEGORIES[2]
    shifted = monitor_study(series(np.r_[np.zeros(40), np.full(40, 2.0)]), -4, 4, **SIM)
    assert shifted.zone == "intervene" and shifted.intervention > 40
    assert shifted.category == CATEGORIES[0]
    bad = monitor_study(series(np.full(30, 2.5)), -4, 4, **SIM)
    f = bad.frame
    assert not (f.loc[f["t"] < 10, "zone"] == "intervene").any()       # nothing before t_min
    assert bad.intervention == 10
    ga, gr = 0.95, 0.80
    for _, row in f.iterrows():
        expect = ("intervene" if row["t"] >= 10 and row["pi"] <= gr else "continue" if row["pi"] >= ga
                  else "monitor")
        assert row["zone"] == expect


def test_recovered_transient_category():
    r = monitor_study(series(np.r_[np.zeros(20), np.full(12, 2.5), np.zeros(80)], rng=5), -4, 4, **SIM)
    assert r.intervention is not None and r.zone == "continue"
    assert r.category == CATEGORIES[1]


def test_monte_carlo_is_reproducible():
    x = series(np.zeros(20))
    a, b = monitor_study(x, -4, 4, rng=7), monitor_study(x, -4, 4, rng=7)
    assert np.array_equal(a.pi, b.pi)


def test_gauge_correction_raises_pi():
    x = series(np.full(40, 1.3), rng=2)
    plain = monitor_study(x, -4, 4, **SIM)
    corrected = monitor_study(x, -4, 4, gauge_sd=0.5, **SIM)
    assert corrected.pi[-1] > plain.pi[-1]
    assert any("gauge" in n for n in corrected.notes)


def test_one_sided_limits():
    x = np.abs(np.random.default_rng(0).normal(0, 0.02, 200))
    gdt = monitor_study(x, 0, 0.12, 0)
    assert gdt.lsl == -math.inf and any("GD&T" in n for n in gdt.notes)
    assert gdt.prior.beta0 == pytest.approx((0.12 / 8) ** 2)
    upper = monitor_study(x, None, 0.12)
    assert upper.prior.beta0 == pytest.approx((0.12 / 8) ** 2) and upper.zone == "continue"
    two_sided = monitor_study(x, 0, 0.12, 0, zero_lsl_gdt=False)
    assert two_sided.lsl == 0
    with pytest.raises(ValueError, match="both limits"):
        monitor_study(x, 0.0, None)
    assert monitor_study(x, -0.1, None, prior=(0, 1e-3, 2, 1e-4)).zone in ("continue", "monitor", "intervene")


def test_input_checks_and_subgroups():
    assert [len(g) for g in subgroups(np.arange(12.0), 5)] == [5, 5]
    assert [len(g) for g in subgroups([[1.0, 2.0, np.nan], [3.0, 4.0]])] == [2, 2]
    with pytest.raises(ValueError):
        monitor_study([1.0, 2.0], -4, 4)
    with pytest.raises(ValueError):
        monitor_study(series(np.zeros(10)), -4, 4, gamma_a=0.8, gamma_r=0.9)
    with pytest.raises(ValueError):
        monitor_study(series(np.zeros(10)), 4, -4)
    with pytest.raises(ValueError):
        monitor_study(series(np.zeros(10)))
    short = monitor_study(series(np.zeros(4)), -4, 4)
    assert any("t_min" in n for n in short.notes)


def test_normality_note():
    x = np.random.default_rng(0).lognormal(0, 0.8, 600)
    r = monitor_study(x, 0, 30, 3)
    assert any("Anderson-Darling" in n for n in r.notes)


def test_report():
    text = monitor_study(series(np.zeros(20)), -4, 4, name="D1").report()
    for key in ("D1", "q0", "Final pi", "Category", "half-life 13.5"):
        assert key in text


def test_batch_monitor_and_exports(tmp_path):
    rng = np.random.default_rng(4)
    data = {"A": rng.normal(10, 0.02, 300), "B": rng.normal(10.07, 0.02, 300), "C": rng.normal(0, 1, 300)}
    specs = {"A": {"lsl": 9.9, "usl": 10.1, "target": 10}, "B": {"lsl": 9.9, "usl": 10.1, "target": 10},
             "C": {"lsl": -4}}
    b = batch_monitor(data, specs, n_draws=500)
    f = b.frame.set_index("dimension")
    assert f.loc["A", "zone"] == "continue" and f.loc["B", "zone"] == "intervene"
    assert f.loc["C", "zone"] == "n/a" and "C" in b.errors
    assert f.loc["A", "Ppk_normal"] == pytest.approx(min(10.1 - data["A"].mean(), data["A"].mean() - 9.9)
                                                       / (3 * data["A"].std(ddof=1)))
    assert list(b.table().columns)[:4] == ["dimension", "target", "LSL", "USL"]
    tr = b.trajectories()
    assert len(tr) == 120 and set(tr["dimension"]) == {"A", "B"}
    html = b.to_html(str(tmp_path / "m.html"))
    assert "pi_final" in html and "How the decisions are made" in html and "intervene" in html
    assert "B" in b.report("B") and "lower limit only" in b.report("C")
    pytest.importorskip("openpyxl")
    from openpyxl import load_workbook
    b.to_excel(tmp_path / "m.xlsx")
    wb = load_workbook(tmp_path / "m.xlsx")
    assert wb.sheetnames == ["Results", "How decided", "Reports", "Trajectories"]
    assert wb["Trajectories"].max_row == 121


def test_batch_monitor_reads_the_standard_layout():
    b = batch_monitor(str(TEMPLATE))
    assert list(b.frame["dimension"]) == ["D101", "D102", "D103", "D104"]
    assert (b.frame["subgroups"] == 6).all()
    assert b.frame.loc[3, "warnings"].startswith("upper limit only")


def test_top_level_exports():
    assert uccap.monitor_study is monitor_study and uccap.batch_monitor is batch_monitor
    assert "jiang2026bayesian" in uccap.cite("monitor") and "jiang2026bayesian" in uccap.cite()


def test_existing_batch_unchanged():
    """The monitoring module leaves batch_study and its exports as they were."""
    r = uccap.batch_study(str(TEMPLATE))
    assert "pi_final" not in r.frame.columns
    assert "How the decisions are made" in r.to_html()


def test_monitor_plots():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from uccap.plot import monitor_figure, monitor_overview
    r = monitor_study(series(np.r_[np.zeros(30), np.full(30, 2.0)]), -4, 4, **SIM)
    fig = r.plot()
    assert len(fig.axes) == 2
    b = batch_monitor({"A": series(np.zeros(30)), "B": series(np.full(30, 2.0))},
                      {"A": {"lsl": -4, "usl": 4}, "B": {"lsl": -4, "usl": 4}}, n_draws=500)
    assert len(monitor_overview(b).axes) == 3 and len(monitor_figure(r).axes) == 2


def test_cli_monitor(tmp_path, capsys):
    from uccap.cli import main
    assert main(["monitor", str(TEMPLATE), "--out", str(tmp_path) + "/", "--csv", "--html", "--draws", "500"]) == 0
    assert (tmp_path / "multiple_dimensions_monitor.html").exists()
    traj = pd.read_csv(tmp_path / "multiple_dimensions_monitor_trajectories.csv")
    assert set(traj["dimension"]) == {"D101", "D102", "D103", "D104"}
    assert "pi_final" in capsys.readouterr().out
    pytest.importorskip("matplotlib")
    assert main(["monitor", str(TEMPLATE), "--out", str(tmp_path) + "/", "--plots", "--quiet"]) == 0
    folder = tmp_path / "multiple_dimensions_monitor_plots"
    assert (folder / "all_dimensions.png").exists() and (folder / "D101_monitor.png").exists()
