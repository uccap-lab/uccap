import numpy as np
import pandas as pd
import pytest

import uccap
from uccap.batch import parse_specs, split_dimensions


@pytest.fixture
def wide():
    rng = np.random.default_rng(20)
    return pd.DataFrame({"A": rng.normal(10, 0.1, 30), "B": rng.normal(5, 0.2, 30),
                         "C": list(rng.normal(1, 0.01, 20)) + [np.nan] * 10})


SPECS = pd.DataFrame({"Dimension": ["A", "B", "C"], "Lower": [9.5, None, 0.95],
                      "Upper": [10.5, 6.0, 1.05], "Nominal": [10.0, None, 1.0]})


def test_spec_aliases():
    spec = parse_specs(SPECS)
    assert spec["A"] == {"lsl": 9.5, "usl": 10.5, "target": 10.0}
    assert spec["B"]["lsl"] is None and spec["B"]["usl"] == 6.0


def test_wide_columns_of_different_length_are_trimmed(wide):
    parts = split_dimensions(wide)
    assert len(parts["C"]) == 20 and len(parts["A"]) == 30


def test_long_layout_matches_wide(wide):
    long = wide.melt(var_name="dimension", value_name="value").dropna()
    long["order"] = long.groupby("dimension").cumcount()
    long = long.sample(frac=1, random_state=0)  # shuffled rows, order column restores time order
    a = uccap.batch_study(wide, SPECS).frame.set_index("dimension")
    b = uccap.batch_study(long, SPECS).frame.set_index("dimension")
    assert np.allclose(a.loc[["A", "B", "C"], "capability"], b.loc[["A", "B", "C"], "capability"])


def test_missing_spec_gives_error_row(wide):
    res = uccap.batch_study(wide, SPECS[SPECS.Dimension != "B"])
    assert res.frame.set_index("dimension").loc["B", "decision"] == "ERROR"
    assert "B" in res.errors and "A" in res.studies


def test_table_selection_and_exports(wide, tmp_path):
    res = uccap.batch_study(wide, SPECS, cost_ratio=19)
    t = res.table(["dimension", "Ppk", "decision"], rename={"Ppk": "Ppk (overall)"})
    assert list(t.columns) == ["dimension", "Ppk (overall)", "decision"]
    html = res.to_html(tmp_path / "r.html")
    assert "<table" in html and (tmp_path / "r.html").exists()
    res.to_excel(tmp_path / "r.xlsx")
    from openpyxl import load_workbook
    wb = load_workbook(tmp_path / "r.xlsx")
    assert wb.sheetnames == ["Results", "How decided", "Reports"]


def test_read_text_variants():
    excel_paste = "Dim1\tDim2\n4.61\t50.78\n4.62\t50.79\n"
    assert list(uccap.read_text(excel_paste).columns) == ["Dim1", "Dim2"]
    euro = "A;B\n4,61;50,78\n4,62;50,79\n"
    df = uccap.read_text(euro)
    assert df["A"].iloc[0] == pytest.approx(4.61)
    bare = "4.61, 50.78\n4.62, 50.79\n"
    df = uccap.read_text(bare)
    assert list(df.columns) == ["Dim1", "Dim2"] and df.shape == (2, 2)
    column = "4.61\n4.62\n4.63\n"
    assert uccap.read_text(column).shape == (3, 1)


@pytest.mark.skipif(not uccap.hybrid.model_available(), reason="trained UC-Cap parameters not available")
def test_uccap_risk_columns(wide):
    r = uccap.batch_study(wide, SPECS)
    t = r.frame.set_index("dimension")
    # bilateral dimensions get the UC-Cap score, identical to hybrid_risk on the same data
    for d in ("A", "C"):
        h = uccap.hybrid_risk(split_dimensions(wide)[d].to_numpy(float), *[SPECS.set_index("Dimension").loc[d, k]
                              for k in ("Lower", "Upper", "Nominal")], rng=0)
        assert t.loc[d, "risk_p"] == pytest.approx(h.p)
        assert t.loc[d, "risk_score"] == pytest.approx(100 * h.p, abs=0.05)
        assert t.loc[d, "risk_level"] == h.level
        assert "UC-Cap hybrid risk" in r.report(d)
    assert t.loc["B", "risk_score"] >= 0  # upper limit only: coded LSL = 0, Ppk = Ppu
    assert uccap.hybrid_risk(split_dimensions(wide)["B"].to_numpy(float), None, 6.0).features["Upper_only"]
    low = uccap.batch_study(wide, SPECS.assign(Lower=[9.5, 4.0, 0.95], Upper=[10.5, None, 1.05]))
    assert low.frame.set_index("dimension").loc["B", "risk_level"] == "n/a (lower limit only)"
    assert "risk_level" in r.table().columns
    assert "risk_score" not in uccap.batch_study(wide, SPECS, hybrid=False).frame


def test_subset_keeps_selected_dimensions_and_within_method():
    import numpy as np
    rng = np.random.default_rng(3)
    data = {f"D{i}": rng.normal(10, 0.01, 30) for i in range(4)}
    specs = {k: {"lsl": 9.95, "usl": 10.05} for k in data}
    res = uccap.batch_study(data, specs, cost_ratio=19, within="mmr", mr_span=3)
    sub = res.subset(["D3", "D1"])
    assert list(sub.frame["dimension"]) == ["D1", "D3"]
    assert set(sub.studies) == {"D1", "D3"} and set(sub.risks) <= {"D1", "D3"}
    normal = res.frame["method"] == "Normal"
    assert set(res.frame.loc[normal, "sd_within_method"]) == {"MMR, span 3"}
    assert set(res.frame.loc[~normal, "sd_within_method"]) <= {"not used"}


def test_column_help_and_decision_notes(wide, tmp_path):
    from uccap.export import COLUMN_HELP, decision_notes
    r = uccap.batch_study(wide, SPECS, cost_ratio=19)
    assert all(c in COLUMN_HELP for c in r.table().columns)
    notes = dict(decision_notes(r))
    assert "C0 = 1.33" in notes["Threshold rule (threshold_decision)"]
    assert "cost ratio 19" in notes["Risk-calibrated rule (decision)"] and "1.6449" in notes["Risk-calibrated rule (decision)"]
    if uccap.hybrid.model_available():
        assert "risk_p < 0.4" in notes["Risk level (risk_level)"]
    html = r.to_html()
    assert "How the decisions are made" in html and 'title="Plain threshold rule' in html
    assert "UC-Cap" not in dict(decision_notes(uccap.batch_study(wide, SPECS, hybrid=False)))
    pytest.importorskip("openpyxl")
    import openpyxl
    r.to_excel(tmp_path / "r.xlsx")
    wb = openpyxl.load_workbook(tmp_path / "r.xlsx")
    assert "How decided" in wb.sheetnames and wb["Results"]["A1"].comment is not None


def test_zero_nominal_and_lsl_read_as_gdt_unless_nominal_empty():
    rng = np.random.default_rng(7)
    x = np.abs(rng.normal(0.02, 0.01, 30))
    mu, s = x.mean(), x.std(ddof=1)
    gdt = uccap.batch_study({"F": x}, {"F": {"lsl": 0.0, "usl": 0.2, "target": 0.0}}).frame.iloc[0]
    assert gdt["Ppk"] == pytest.approx((0.2 - mu) / (3 * s), rel=1e-6)
    assert "GD&T" in gdt["warnings"]
    if uccap.hybrid.model_available():
        assert gdt["risk_level"] == "Low"
    step = uccap.batch_study({"F": x}, {"F": {"lsl": 0.0, "usl": 0.2}}).frame.iloc[0]  # real limit at 0
    assert step["Ppk"] == pytest.approx(min(0.2 - mu, mu) / (3 * s), rel=1e-6)
    off = uccap.batch_study({"F": x}, {"F": {"lsl": 0.0, "usl": 0.2, "target": 0.0}}, zero_lsl_gdt=False)
    assert off.frame.iloc[0]["decision"] == "ERROR"


def test_specification_for_unknown_dimension_warns():
    data = pd.DataFrame({"A": np.random.default_rng(0).normal(10, 0.1, 20)})
    with pytest.warns(UserWarning, match="not in the data: Z"):
        uccap.batch_study(data, {"A": dict(lsl=9, usl=11), "Z": dict(lsl=9, usl=11)})
