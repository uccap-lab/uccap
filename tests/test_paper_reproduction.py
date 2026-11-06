"""Published results of the workflow, risk and monitoring articles, reproduced (see validation/)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "validation"))
import workflow_case_study as workflow_cs  # noqa: E402
import risk_simulation as risk_sim  # noqa: E402
import monitor_simulation as monitor_sim  # noqa: E402


def test_workflow_constants_table1():
    for w, u2, a2, uc, ac, u4, a4 in workflow_cs.constants():
        assert (round(u2, 4), round(uc, 4), round(u4, 4)) == (a2, ac, a4), w


def test_workflow_case_study_dimension_105():
    rows = workflow_cs.rows()
    for name, s, s_a, cp, cp_a, cpk, cpk_a in rows:
        assert round(s, 4) == s_a, name
        assert cp == pytest.approx(cp_a, rel=0.007) and cpk == pytest.approx(cpk_a, rel=0.007), name
    _, _, _, pp, pp_a, ppk, ppk_a = rows[0]
    assert (round(pp, 3), round(ppk, 3)) == (pp_a, ppk_a)


def test_risk_table1():
    for lam, k, u in risk_sim.table1():
        assert u == k, lam


def test_risk_table2_expected_loss():
    risk_sim.check_rule_equivalence(n_samples=50)
    for r in risk_sim.table2(b=200_000):
        # the article used 12,000 replications: compare in units of its Monte Carlo error
        assert abs(r["det_z"]) < 3.5 and abs(r["cal_z"]) < 3.5, r["lambda"]
        assert r["reduction"] == pytest.approx(r["reduction_article"], abs=2.0)


def test_monitor_operating_characteristics():
    """Monitoring article, Table 11 with B = 100 (the article: B = 1000; validation/monitor_simulation.py)."""
    stable = monitor_sim.replicate("stable", 100)
    assert stable["any"] <= 0.06                                      # article 0.015
    shift = monitor_sim.replicate("shift 1.5", 100)
    assert shift["first_bad"] == 61 and shift["detect"] == 1.0        # article 1.00
    assert 5 <= shift["delay"] <= 7                                   # article median delay 6
