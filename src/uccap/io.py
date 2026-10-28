"""Read measurement tables from files, pasted text or the clipboard.

Standard layout (one table, specifications above the data)::

    Dimension   D101    D102    D103
    Nominal     4.62    50.78
    USL         4.72    50.88   0.12
    LSL         4.52    50.70
    1           4.618   50.781  0.021
    2           4.627   50.779  0.018

The first column holds row labels: specification names (Nominal, USL, LSL; also
Unit and Description) and then part numbers or time stamps. ``read_table`` also
recognises title rows above the table, summary rows below it (Mean, Max, Cpk...),
a transposed table (one row per dimension), a separate specification table, and
plain wide or long data without specifications.

Separators (comma, semicolon, tab, spaces) are detected automatically. With a
semicolon or tab separator, a decimal comma ("4,618") is read as a decimal point,
as in European Excel exports. Text files are read as UTF-8, falling back to GBK
(GB18030) and Latin-1.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

_NUM = re.compile(r"\s*[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?\s*")


def _is_number(s: str) -> bool:
    return bool(_NUM.fullmatch(s))


def _detect(lines: list[str]) -> tuple[str, str]:
    sample = "\n".join(lines[:20])
    if "\t" in sample:
        sep = "\t"
    elif ";" in sample:
        sep = ";"
    elif "," in sample:
        sep = ","
    else:
        sep = r"\s+"
    decimal = "," if sep != "," and re.search(r"\d,\d", sample) and "." not in sample else "."
    return sep, decimal


def read_text(text: str) -> pd.DataFrame:
    """Parse a pasted table (e.g. cells copied from Excel) into a plain DataFrame.

    For tables that carry specification rows, use :func:`read_table` instead.
    """
    text = text.strip("\n")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("no data")
    sep, decimal = _detect(lines)
    first = re.split(sep, lines[0].strip()) if sep == r"\s+" else lines[0].split(sep)
    header = 0 if any(c.strip() and not _is_number(c) for c in first) else None
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, decimal=decimal, header=header,
                     engine="python", skipinitialspace=True)
    if header is None:
        df.columns = [f"Dim{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_file_text(path) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def read_data(path, sheet=0) -> pd.DataFrame:
    """Read a CSV / TXT or Excel (.xlsx, .xls) file into a plain DataFrame.

    For files in the standard layout (specification rows above the data), use
    :func:`read_table`, or pass the path straight to :func:`uccap.batch_study`.
    """
    p = str(path).lower()
    if p.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return read_text(_read_file_text(path))


def read_clipboard() -> pd.DataFrame:
    """Read cells copied from Excel or another spreadsheet."""
    return read_text(pd.io.clipboard.clipboard_get())


# ---------------------------------------------------------------- standard layout

ROW_ALIASES = {
    "target": ["nominal", "nom", "nominal value", "target", "target value", "t", "标称", "名义值", "目标",
               "目标值", "标准值", "公称值"],
    "usl": ["usl", "upper", "upper limit", "upper spec", "upper spec limit", "max limit", "上限", "规格上限"],
    "lsl": ["lsl", "lower", "lower limit", "lower spec", "lower spec limit", "min limit", "下限", "规格下限"],
    # Tolerances relative to the nominal: USL = Nominal + upper, LSL = Nominal - |lower|.
    "utol": ["upper tol", "uppertol", "upper tolerance", "+tol", "tol+", "plus tol", "upper dev",
             "upper deviation", "上公差", "上偏差"],
    "ltol": ["lower tol", "lowertol", "lower tolerance", "minus tol", "lower dev", "lower deviation",
             "下公差", "下偏差"],
    "tol": ["tol", "tolerance", "±tol", "+/ tol", "公差"],
    "unit": ["unit", "units", "单位"],
    "description": ["description", "desc", "说明", "描述", "名称"],
}
DIMENSION_LABELS = ["dimension", "dim", "feature", "characteristic", "name", "id", "尺寸", "特征"]
SUMMARY_LABELS = ["mean", "average", "avg", "max", "maximum", "min", "minimum", "range", "std", "stdev",
                  "std dev", "sd", "sigma", "cp", "cpk", "pp", "ppk", "count", "n", "均值", "平均",
                  "平均值", "最大", "最大值", "最小", "最小值", "极差", "标准差"]


def _norm(s) -> str:
    s = re.sub(r"[(（\[][^)）\]]*[)）\]]\s*$", "", str(s))   # drop a trailing unit such as "(mm)"
    return re.sub(r"[\s_\-:.()（）：]+", " ", s).strip().lower()


TOLERANCE_KEYS = ("utol", "ltol", "tol")


def apply_tolerances(spec: dict, tols: dict, name, notes: list) -> None:
    """Turn tolerances relative to the nominal into absolute limits, in place.

    ``tols`` may hold "utol", "ltol" and "tol" (symmetric). Explicit USL / LSL win.
    The upper tolerance is added with its sign; the lower tolerance is taken as a distance
    below the nominal whatever its sign (0.1 and -0.1 both give Nominal - 0.1).
    """
    up = tols.get("utol", tols.get("tol"))
    lo = tols.get("ltol", tols.get("tol"))
    if up is None and lo is None:
        return
    if spec.get("target") is None:
        notes.append(f"{name}: tolerances need a Nominal row; limits not set")
        return
    if spec.get("usl") is None and up is not None:
        spec["usl"] = spec["target"] + up
    if spec.get("lsl") is None and lo is not None:
        spec["lsl"] = spec["target"] - abs(lo)


_LABEL = {_norm(a): k for k, v in ROW_ALIASES.items() for a in v}
_DIMS = {_norm(a) for a in DIMENSION_LABELS}
_SUMMARY = {_norm(a) for a in SUMMARY_LABELS}


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _to_float(s: str, decimal: str = ".") -> Optional[float]:
    s = s.strip()
    if decimal == ",":
        s = s.replace(",", ".")
    m = re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s.replace(" ", ""))
    return float(m.group(0)) if m else None


def _grid_from_text(text: str) -> tuple[list[list[str]], str]:
    lines = [ln for ln in text.strip("\n").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("no data")
    sep, decimal = _detect(lines)
    if sep == r"\s+":
        rows = [re.split(r"\s+", ln.strip()) for ln in lines]
    else:
        import csv
        rows = list(csv.reader(lines, delimiter=sep))
    return [[c.strip() for c in r] for r in rows], decimal


def _grid(source, sheet) -> tuple[list[list[str]], str]:
    if isinstance(source, pd.DataFrame):
        df = source
        rows = []
        if not isinstance(df.columns, pd.RangeIndex):
            rows.append([_cell(c) for c in df.columns])
        rows += [[_cell(v) for v in r] for r in df.itertuples(index=False)]
        return rows, "."
    if isinstance(source, (str, os.PathLike)) and (isinstance(source, os.PathLike) or "\n" not in source):
        p = str(source).lower()
        if p.endswith((".xlsx", ".xlsm", ".xls")):
            df = pd.read_excel(source, sheet_name=sheet, header=None, dtype=object)
            return [[_cell(v) for v in r] for r in df.itertuples(index=False)], "."
        return _grid_from_text(_read_file_text(source))
    return _grid_from_text(str(source))


def _trim(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = [r for r in rows if any(r)]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows]


@dataclass
class TableInput:
    """Measurements and specifications read from one table.

    ``data`` has one column per dimension, rows in production order. ``specs``
    maps each dimension to ``{"lsl", "usl", "target"}`` (empty when the table had
    no specifications). ``layout`` names what was recognised and ``notes`` lists
    rows that were skipped, so that the reading can be checked.
    """
    data: pd.DataFrame
    specs: dict = field(default_factory=dict)
    layout: str = "wide"
    units: dict = field(default_factory=dict)
    descriptions: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        n_spec = sum(1 for s in self.specs.values() if s.get("lsl") is not None or s.get("usl") is not None)
        lines = [f"layout     : {self.layout}",
                 f"dimensions : {self.data.shape[1]}  ({', '.join(map(str, self.data.columns[:8]))}"
                 f"{', ...' if self.data.shape[1] > 8 else ''})",
                 f"parts      : {self.data.shape[0]}",
                 f"with specs : {n_spec}"]
        lines += [f"note       : {n}" for n in self.notes]
        return "\n".join(lines)


def read_table(source, sheet=0) -> TableInput:
    """Read measurements and, when present, specifications from one table.

    ``source`` is a file path (CSV, TXT, XLSX, XLS), pasted text, or a DataFrame.
    See the module docstring for the standard layout.
    """
    rows, decimal = _grid(source, sheet)
    rows = _trim(rows)
    if not rows:
        raise ValueError("the table is empty")
    labels = [_LABEL.get(_norm(r[0])) for r in rows]
    if any(lab in ("usl", "lsl", "target", *TOLERANCE_KEYS) for lab in labels):
        return _specs_on_top(rows, labels, decimal)
    for h, r in enumerate(rows[:30]):
        keys = [_LABEL.get(_norm(c)) or ("dimension" if _norm(c) in _DIMS else None) for c in r]
        if "dimension" in keys and any(k in keys for k in ("usl", "lsl", *TOLERANCE_KEYS)):
            return _transposed(rows, h, keys, decimal)
    return _plain(rows, decimal)


def _specs_on_top(rows, labels, decimal) -> TableInput:
    notes = []
    first = next(i for i, lab in enumerate(labels) if lab is not None)
    # The header is the nearest row above the specification rows that holds names
    # (text). Rows in between with numbers only have a label we do not know.
    header, unknown = None, []
    for i in range(first - 1, -1, -1):
        values = [c for c in rows[i][1:] if c]
        if values and all(_is_number(c) for c in values):
            unknown.append(rows[i][0] or f"row {i + 1}")
            continue
        header = i
        break
    if unknown:
        notes.append("ignored row(s) with an unknown label: " + ", ".join(reversed(unknown))
                     + " (use Nominal, USL, LSL)")
    if header:
        notes.append(f"skipped {header} title row(s) above the header")
    end = first
    while end < len(rows) and labels[end] is not None:
        end += 1
    width = len(rows[0])
    names = []
    for j in range(1, width):
        name = rows[header][j] if header is not None else ""
        names.append(name or f"Dim{j}")
    specs = {n: {"lsl": None, "usl": None, "target": None} for n in names}
    tols = {n: {} for n in names}
    units, desc = {}, {}
    for i in range(first, end):
        lab = labels[i]
        for j, n in enumerate(names, 1):
            v = rows[i][j]
            if not v:
                continue
            if lab in ("lsl", "usl", "target"):
                specs[n][lab] = _to_float(v, decimal)
            elif lab in TOLERANCE_KEYS:
                tols[n][lab] = _to_float(v, decimal)
            elif lab == "unit":
                units[n] = v
            else:
                desc[n] = v
    for n in names:
        apply_tolerances(specs[n], tols[n], n, notes)
    if any(tols.values()):
        notes.append("limits computed from tolerances: USL = Nominal + upper, LSL = Nominal - lower")
    # Text-labelled rows right after the specification rows, followed by numbered parts, are
    # specification rows with a label we do not know; they must not be read as measurements.
    rest = rows[end:]
    lead = 0
    while lead < len(rest) and rest[lead][0] and not _is_number(rest[lead][0]) \
            and _norm(rest[lead][0]) not in _SUMMARY:
        lead += 1
    if 0 < lead < len(rest) and _is_number(rest[lead][0]):
        notes.append("ignored row(s) with an unknown label: " + ", ".join(r[0] for r in rest[:lead])
                     + " (use Nominal, USL, LSL, or UpperTol / LowerTol)")
        rest = rest[lead:]
    body, skipped = [], []
    for r in rest:
        if _norm(r[0]) in _SUMMARY:
            skipped.append(r[0])
            continue
        body.append(r[1:])
    if skipped:
        notes.append("skipped summary row(s): " + ", ".join(skipped))
    data = pd.DataFrame(body, columns=names).replace("", None)
    if decimal == ",":
        data = data.apply(lambda c: c.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v))
    return TableInput(data, specs, "specs on top", units, desc, notes)


def _transposed(rows, h, keys, decimal) -> TableInput:
    notes = [f"skipped {h} title row(s) above the header"] if h else []
    dim_j = keys.index("dimension")
    spec_j = {k: keys.index(k) for k in ("lsl", "usl", "target") if k in keys}
    tol_j = {k: keys.index(k) for k in TOLERANCE_KEYS if k in keys}
    meta = {dim_j, *spec_j.values(), *tol_j.values(),
            *(j for j, k in enumerate(keys) if k in ("unit", "description"))}
    value_j = [j for j in range(len(rows[h])) if j not in meta and j > dim_j]
    specs, cols, units, desc = {}, {}, {}, {}
    for r in rows[h + 1:]:
        name = r[dim_j]
        if not name:
            continue
        specs[name] = {k: _to_float(r[j], decimal) if r[j] else None for k, j in spec_j.items()}
        for k in ("lsl", "usl", "target"):
            specs[name].setdefault(k, None)
        apply_tolerances(specs[name], {k: _to_float(r[j], decimal) for k, j in tol_j.items() if r[j]},
                         name, notes)
        if "unit" in keys and r[keys.index("unit")]:
            units[name] = r[keys.index("unit")]
        vals = [r[j].replace(",", ".") if decimal == "," else r[j] for j in value_j]
        cols[name] = pd.Series([v or None for v in vals], dtype=object)
    if not value_j:
        return TableInput(pd.DataFrame(), specs, "specification table", units, desc, notes)
    return TableInput(pd.DataFrame(cols), specs, "transposed (one row per dimension)", units, desc, notes)


def _plain(rows, decimal) -> TableInput:
    notes = []
    header = None
    for i, r in enumerate(rows[:30]):
        nonempty = [c for c in r if c]
        if nonempty and all(_is_number(c) for c in nonempty):
            break
        header = i
    start = 0 if header is None else header + 1
    if header is not None and header > 0:
        notes.append(f"skipped {header} title row(s) above the header")
    names = rows[header] if header is not None else [f"Dim{j + 1}" for j in range(len(rows[0]))]
    names = [n or f"Dim{j + 1}" for j, n in enumerate(names)]
    body = [[c.replace(",", ".") if decimal == "," else c for c in r] for r in rows[start:]
            if _norm(r[0]) not in _SUMMARY]
    data = pd.DataFrame(body, columns=names).replace("", None)
    from .batch import _find
    if _find(data.columns, "dimension") is not None and _find(data.columns, "value") is not None:
        return TableInput(data, {}, "long", notes=notes)
    return TableInput(data, {}, "wide", notes=notes)
