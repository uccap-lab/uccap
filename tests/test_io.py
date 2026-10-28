import numpy as np
import pandas as pd
import pytest

import uccap
from uccap import read_table

rng = np.random.default_rng(1)
A = rng.normal(4.62, 0.02, 30)
B = rng.normal(50.78, 0.02, 30)


def standard_csv(sep=",", dec=".", title=False, summary=False, labels=("Nominal", "USL", "LSL")):
    f = (lambda v: f"{v:.4f}".replace(".", dec))
    lines = []
    if title:
        lines += [f"Part 7781 measurement report{sep}{sep}", f"Date{sep}2026-10-27{sep}"]
    lines += [sep.join(["Dimension", "D101", "D102"]),
              sep.join([labels[0], f(4.62), f(50.78)]),
              sep.join([labels[1], f(4.72), f(50.88)]),
              sep.join([labels[2], f(4.52), ""])]
    lines += [sep.join([str(i + 1), f(a), f(b)]) for i, (a, b) in enumerate(zip(A, B))]
    if summary:
        lines += [sep.join(["Mean", f(A.mean()), f(B.mean())]), sep.join(["Cpk", "1.5", "1.2"])]
    return "\n".join(lines)


def check(t, n=30):
    assert list(t.data.columns) == ["D101", "D102"]
    assert t.specs["D101"] == {"lsl": 4.52, "usl": 4.72, "target": 4.62}
    assert t.specs["D102"]["lsl"] is None and t.specs["D102"]["usl"] == 50.88
    assert t.data.shape[0] == n


def test_standard_text():
    t = read_table(standard_csv())
    assert t.layout == "specs on top"
    check(t)
    assert np.allclose(t.data["D101"].astype(float), np.round(A, 4))


def test_title_and_summary_rows():
    t = read_table(standard_csv(title=True, summary=True))
    check(t)
    assert any("title" in n for n in t.notes) and any("summary" in n for n in t.notes)


def test_semicolon_decimal_comma():
    t = read_table(standard_csv(sep=";", dec=","))
    check(t)
    assert abs(float(t.data["D102"][0]) - round(B[0], 4)) < 1e-9


def test_chinese_labels_gbk_file(tmp_path):
    p = tmp_path / "zh.csv"
    p.write_bytes(standard_csv(labels=("标称", "上限", "下限")).replace("Dimension", "尺寸").encode("gbk"))
    check(read_table(p))


def test_excel_standard(tmp_path):
    pytest.importorskip("openpyxl")
    rows = [["Dimension", "D101", "D102"], ["Nominal", 4.62, 50.78], ["USL", 4.72, 50.88],
            ["LSL", 4.52, None]] + [[i + 1, a, b] for i, (a, b) in enumerate(zip(A, B))]
    p = tmp_path / "d.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    check(read_table(p))
    res = uccap.batch_study(str(p), c0=1.33)
    assert set(res.frame["dimension"]) == {"D101", "D102"}
    assert res.frame.set_index("dimension").loc["D101", "LSL"] == 4.52


def test_dataframe_from_read_data(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(standard_csv())
    df = uccap.read_data(p)                 # plain read: spec rows end up as data rows
    res = uccap.batch_study(df)             # ...but batch_study recognises them
    assert res.frame.set_index("dimension").loc["D101", "USL"] == 4.72
    assert res.frame.set_index("dimension").loc["D101", "n"] == 30


def test_transposed():
    lines = ["Dimension,Nominal,USL,LSL," + ",".join(f"P{i}" for i in range(1, 31)),
             "D101,4.62,4.72,4.52," + ",".join(f"{v:.4f}" for v in A),
             "D102,50.78,50.88,," + ",".join(f"{v:.4f}" for v in B)]
    t = read_table("\n".join(lines))
    assert t.layout.startswith("transposed")
    check(t)


def test_spec_table_only():
    t = read_table("Dimension,LSL,USL,Nominal\nD101,4.52,4.72,4.62\nD102,,50.88,50.78")
    assert t.layout == "specification table" and t.data.empty
    assert t.specs["D101"]["usl"] == 4.72


def test_plain_wide_with_title_and_no_header():
    t = read_table("Report\nD101,D102\n1.0,2.0\n1.1,2.1")
    assert t.layout == "wide" and list(t.data.columns) == ["D101", "D102"] and t.notes
    t = read_table("1.0,2.0\n1.1,2.1")
    assert list(t.data.columns) == ["Dim1", "Dim2"]


def test_single_dimension_file():
    t = read_table("Dimension,D101\nUSL,4.72\nLSL,4.52\n" + "\n".join(f"{i},{v:.4f}" for i, v in enumerate(A, 1)))
    s = uccap.capability_study(t.data["D101"], **{k: v for k, v in t.specs["D101"].items() if k != "target"})
    assert s.n == 30


def test_short_labels_and_unknown_rows():
    body = "\n".join(f"{i},{a:.4f},{b:.4f}" for i, (a, b) in enumerate(zip(A, B), 1))
    t = read_table("NO.,D101,D102\nT,4.62,50.78\nUSL,4.72,50.88\nLSL,4.52,\n" + body)
    check(t)
    t = read_table("NO.,D101,D102\nRef,4.62,50.78\nUSL,4.72,50.88\nLSL,4.52,\n" + body)
    assert list(t.data.columns) == ["D101", "D102"] and t.specs["D101"]["usl"] == 4.72
    assert any("Ref" in n for n in t.notes)


def test_tolerance_rows_become_limits():
    text = ("Item\tD1\tD2\nNominal\t0.55\t10\nUpperTol\t0.1\t0.05\nLowerTol\t0.1\t-0.02\n"
            + "".join(f"{i}\t{0.5 + 0.01 * (i % 5)}\t{10 + 0.001 * (i % 7)}\n" for i in range(1, 21)))
    t = read_table(text)
    assert t.data.shape == (20, 2)
    assert t.specs["D1"]["usl"] == pytest.approx(0.65) and t.specs["D1"]["lsl"] == pytest.approx(0.45)
    assert t.specs["D2"]["usl"] == pytest.approx(10.05) and t.specs["D2"]["lsl"] == pytest.approx(9.98)
    assert any("tolerances" in n for n in t.notes)


def test_symmetric_tolerance_unit_suffix_and_unknown_rows():
    text = "Dimension\tA\nNominal (mm)\t5\n+/-Tol\t0.1\nGauge\t3\n1\t5.01\n2\t5.02\n"
    t = read_table(text)
    assert t.specs["A"] == {"lsl": pytest.approx(4.9), "usl": pytest.approx(5.1), "target": 5.0}
    assert t.data.shape[0] == 2
    assert any("unknown label: Gauge" in n for n in t.notes)


def test_explicit_limits_win_over_tolerances():
    t = read_table("Dimension\tA\nNominal\t5\nUSL\t5.2\nTol\t0.1\n1\t5.01\n2\t5.02\n")
    assert t.specs["A"]["usl"] == 5.2 and t.specs["A"]["lsl"] == pytest.approx(4.9)


def test_transposed_tolerance_columns():
    t = read_table("Dimension,Nominal,Upper Tol,Lower Tol,1,2\nD1,10,0.1,0.1,10.01,10.02\n")
    assert t.specs["D1"]["usl"] == pytest.approx(10.1) and t.specs["D1"]["lsl"] == pytest.approx(9.9)
    assert list(t.data["D1"].dropna()) == ["10.01", "10.02"]
