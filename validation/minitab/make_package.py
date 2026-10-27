"""Build the Minitab / JMP cross-check workbook (validation/minitab/uccap_minitab_check.xlsx).

Each data set is analysed by uccap with the family it was drawn from, so that the
numbers can be compared with Minitab's Capability Analysis (Normal / Nonnormal)
or JMP's Process Capability platform using the same distribution.
"""
import pathlib

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from uccap.workflow.fitting import fit_distribution
from uccap.workflow.indices import Spec, normal_indices
from uccap.workflow.normality import anderson_darling
from uccap.workflow.percentile import percentile_indices
from uccap.workflow.sigma import sigma_amr, sigma_overall

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SPECS = {  # (LSL, USL); None = one-sided
    "normal": (8.5, 11.5), "lognormal": (None, 3.0), "weibull": (None, 3.0), "gamma": (None, 5.0),
    "exponential": (None, 4.0), "lev": (-4.0, 8.0), "sev": (-8.0, 4.0), "logistic": (3.5, 6.5),
}
MINITAB_NAMES = {"normal": "Normal", "lognormal": "Lognormal", "weibull": "Weibull", "gamma": "Gamma",
                 "exponential": "Exponential", "lev": "Largest extreme value",
                 "sev": "Smallest extreme value", "logistic": "Logistic"}

wb = Workbook()
ws = wb.active; ws.title = "Data"
bold = Font(bold=True); head = PatternFill("solid", fgColor="E6EFF3")
data = {k: np.loadtxt(DATA / f"{k}.csv") for k in SPECS}
for j, (k, x) in enumerate(data.items(), 1):
    ws.cell(1, j, k).font = bold
    for i, v in enumerate(x, 2):
        ws.cell(i, j, float(v))

rs = wb.create_sheet("uccap results")
cols = ["Data", "Distribution (use in Minitab)", "LSL", "USL", "N", "Mean", "StDev(Overall)",
        "StDev(Within) AMR", "AD statistic", "AD p-value", "Cp", "Cpk", "Pp", "Ppk",
        "X0.135", "X50", "X99.865", "Ppk (non-normal)", "Pp (non-normal)"]
for j, c in enumerate(cols, 1):
    cell = rs.cell(1, j, c); cell.font = bold; cell.fill = head
for i, (k, x) in enumerate(data.items(), 2):
    lsl, usl = SPECS[k]
    spec = Spec(lsl=lsl, usl=usl)
    a, p = anderson_darling(x)
    row = [k, MINITAB_NAMES[k], lsl, usl, len(x), x.mean(), sigma_overall(x), sigma_amr(x), a, p]
    if k == "normal":
        idx = normal_indices(x.mean(), sigma_amr(x), sigma_overall(x), spec)
        row += [idx.get("Cp"), idx["Cpk"], idx.get("Pp"), idx["Ppk"], None, None, None, None, None]
    else:
        fit = fit_distribution(x, k)
        q = fit.dist.ppf([0.00135, 0.5, 0.99865])
        pi = percentile_indices(fit.dist, spec)
        row += [None, None, None, None, *q, pi["CNpk"], pi.get("CNp")]
    for j, v in enumerate(row, 1):
        rs.cell(i, j, None if v is None or (isinstance(v, float) and v != v) else (round(float(v), 6) if isinstance(v, (float, np.floating)) else v))

ins = wb.create_sheet("How to check")
lines = [
    "Cross-check of uccap against Minitab (or JMP)",
    "",
    "1. Sheet Data has 8 columns of 50 values. Copy them into a Minitab worksheet.",
    "2. normal column: Stat > Quality Tools > Capability Analysis > Normal; subgroup size 1;",
    "   LSL/USL from sheet 'uccap results'. Default options (Rbar/average moving range, length 2, unbiasing on).",
    "   Compare StDev(Within), StDev(Overall), Cp, Cpk, Pp, Ppk.",
    "3. Other columns: Stat > Quality Tools > Capability Analysis > Nonnormal; Fit distribution =",
    "   the 'Distribution' named in 'uccap results'; Options: ISO method (percentiles).",
    "   Compare Ppk and Pp with 'Ppk (non-normal)' and 'Pp (non-normal)'.",
    "4. Stat > Basic Statistics > Normality Test (Anderson-Darling) on each column: compare AD and p-value.",
    "5. Send screenshots of the Minitab output; differences are documented in docs/validation.md.",
    "",
    "Notes: Minitab's Ppk uses the overall StDev without c4 unbiasing, like uccap.",
    "Minitab may estimate 2-parameter Weibull/gamma by maximum likelihood with its own optimiser;",
    "differences below 0.001 in the indices are expected.",
]
for i, t in enumerate(lines, 1):
    c = ins.cell(i, 1, t)
    if i == 1:
        c.font = Font(bold=True, size=13)
ins.column_dimensions["A"].width = 110
for col in "ABCDEFGHIJKLMNOPQRS":
    rs.column_dimensions[col].width = 14
rs.column_dimensions["B"].width = 24
wb.move_sheet("How to check", offset=-2)
wb.save(HERE / "uccap_minitab_check.xlsx")
print("written", HERE / "uccap_minitab_check.xlsx")
