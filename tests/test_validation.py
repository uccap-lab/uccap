"""Agreement with independent reference values (R) and published worked examples.

validation/reference.json is produced by validation/reference.R; see docs/validation.md.
"""
import pathlib
import sys

import pytest

from uccap.workflow.indices import Spec, normal_indices

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "validation"))
import compare  # noqa: E402


@pytest.mark.parametrize("row", list(compare.rows()), ids=lambda r: f"{r[0]}-{r[1]}")
def test_against_r(row):
    group, item, u, r, diff, tol, kind = row
    assert diff <= tol, f"{group} {item}: uccap {u} vs R {r}"


def test_nist_handbook_6_1_6():
    # NIST/SEMATECH e-Handbook of Statistical Methods, section 6.1.6:
    # USL = 20, LSL = 8, mean 16, sigma 2 -> Cp = 1, Cpk = 0.6667.
    idx = normal_indices(16.0, 2.0, 2.0, Spec(lsl=8.0, usl=20.0))
    assert idx["Cp"] == pytest.approx(1.0)
    assert idx["Cpk"] == pytest.approx(2 / 3, abs=1e-4)
    assert idx["Cpl"] == pytest.approx(4 / 3)
