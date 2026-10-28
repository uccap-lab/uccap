"""Capability studies and approval decisions for many dimensions at once.

Input layouts
-------------
* Wide: one column per dimension, one row per measurement, rows in time order.
* Long: one row per measurement with a dimension column and a value column
  (optionally an order column giving the time order).
* A dict mapping dimension names to sequences of measurements.

Specifications come as a table with one row per dimension (columns such as
``dimension, lsl, usl, target``; common spellings like "LSL", "Lower", "Nominal"
are recognised) or a dict ``{dimension: {"lsl": ..., "usl": ..., "target": ...}}``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .io import TOLERANCE_KEYS
from .hybrid import hybrid_risk, model_available
from .risk import risk_decision
from .workflow.pipeline import capability_study

ALIASES = {
    "dimension": ["dimension", "dim", "feature", "characteristic", "name", "id", "尺寸", "特征"],
    "value": ["value", "measurement", "measure", "x", "reading", "测量值", "数值"],
    "order": ["order", "time", "timestamp", "seq", "sequence", "index", "顺序", "时间"],
    "lsl": ["lsl", "lower", "lower limit", "lower spec", "lower_spec_limit", "lo", "下限"],
    "usl": ["usl", "upper", "upper limit", "upper spec", "upper_spec_limit", "hi", "上限"],
    "target": ["target", "nominal", "nom", "目标", "标称", "名义值"],
    "utol": ["upper tol", "uppertol", "upper tolerance", "+tol", "tol+", "上公差", "上偏差"],
    "ltol": ["lower tol", "lowertol", "lower tolerance", "-tol", "tol-", "下公差", "下偏差"],
    "tol": ["tol", "tolerance", "±tol", "+/-tol", "公差"],
}


def _norm(name) -> str:
    return str(name).strip().lower().replace("-", " ").replace("_", " ")


def _find(columns, key) -> Optional[str]:
    wanted = {_norm(a) for a in ALIASES[key]}
    for c in columns:
        if _norm(c) in wanted:
            return c
    return None


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def parse_specs(specs) -> dict[str, dict]:
    """Normalise a spec table or dict to ``{dimension: {"lsl", "usl", "target"}}``."""
    if specs is None:
        return {}
    if isinstance(specs, dict):
        out = {}
        for dim, row in specs.items():
            row = {_norm(k): v for k, v in row.items()}
            vals = {k: _num(next((row[_norm(a)] for a in ALIASES[k] if _norm(a) in row), None))
                    for k in ("lsl", "usl", "target", *TOLERANCE_KEYS)}
            out[str(dim)] = _limits(vals)
        return out
    df = pd.DataFrame(specs)
    dim_col = _find(df.columns, "dimension")
    if dim_col is None:
        raise ValueError("the spec table needs a dimension column (e.g. 'dimension')")
    cols = {k: _find(df.columns, k) for k in ("lsl", "usl", "target", *TOLERANCE_KEYS)}
    if all(cols[k] is None for k in ("lsl", "usl", *TOLERANCE_KEYS)):
        raise ValueError("the spec table needs an LSL or USL column (or tolerances with a nominal)")
    return {str(r[dim_col]): _limits({k: _num(r[c]) if c else None for k, c in cols.items()})
            for _, r in df.iterrows()}


def _limits(vals: dict) -> dict:
    """lsl / usl / target from a row that may give tolerances relative to the nominal."""
    from .io import apply_tolerances
    spec = {k: vals.get(k) for k in ("lsl", "usl", "target")}
    apply_tolerances(spec, {k: vals[k] for k in TOLERANCE_KEYS if vals.get(k) is not None}, "", [])
    return spec


def split_dimensions(data) -> dict[str, pd.Series]:
    """Turn wide, long or dict input into ``{dimension: measurements in time order}``."""
    if isinstance(data, dict):
        return {str(k): pd.Series(list(v)) for k, v in data.items()}
    df = pd.DataFrame(data)
    dim_col, val_col = _find(df.columns, "dimension"), _find(df.columns, "value")
    if dim_col is not None and val_col is not None:  # long layout
        order_col = _find(df.columns, "order")
        if order_col is not None:
            df = df.sort_values(order_col, kind="stable")
        return {str(k): g[val_col].reset_index(drop=True) for k, g in df.groupby(dim_col, sort=False)}
    out = {}
    for c in df.columns:
        col = df[c]
        last = col.last_valid_index()  # columns may have different lengths: trim the padding
        out[str(c)] = col.loc[:last].reset_index(drop=True) if last is not None else col.iloc[:0]
    return out


DISPLAY_NAMES = {
    "normal": "Normal", "lognormal": "Lognormal", "lognormal3": "3-parameter lognormal",
    "exponential": "Exponential", "exponential2": "2-parameter exponential",
    "weibull": "Weibull", "weibull3": "3-parameter Weibull", "gamma": "Gamma",
    "gamma3": "3-parameter gamma", "sev": "Smallest extreme value", "lev": "Largest extreme value",
    "logistic": "Logistic", "loglogistic": "Loglogistic", "loglogistic3": "3-parameter loglogistic",
    "johnson_su": "Johnson SU", "johnson_sb": "Johnson SB", "shash": "SHASH", "boxcox": "Box-Cox normal",
    "normal_mix2": "Mixture of 2 normals", "normal_mix3": "Mixture of 3 normals",
}


def _shape(skew: float, kurt: float) -> str:
    if abs(skew) >= 1 or kurt >= 3:
        return "strongly non-normal"
    if abs(skew) >= 0.5:
        return "skewed"
    if kurt >= 1:
        return "heavy-tailed"
    return "normal-like"


# Default colour rules: column -> function(value) -> "good" | "warn" | "bad" | None.
def _cap_rule(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return "bad" if v < 1.0 else "warn" if v < 1.33 else "good"


DEFAULT_RULES: dict[str, Callable] = {
    "normality": lambda v: {"Pass": "good", "Fail": "bad"}.get(v),
    "shape": lambda v: {"normal-like": "good", "skewed": "warn", "heavy-tailed": "warn"}.get(v, "bad" if v else None),
    "method": lambda v: "good" if v == "Normal" else "warn" if v else None,
    "ppk_mode": lambda v: "good" if v == "Normal" else "warn" if v else None,
    "Cpk": _cap_rule, "Ppk": _cap_rule, "Cp": _cap_rule, "Pp": _cap_rule, "CNpk": _cap_rule,
    "capability": _cap_rule,
    "decision": lambda v: {"ACCEPT": "good", "REJECT": "bad", "ERROR": "bad"}.get(v),
    "threshold_decision": lambda v: {"ACCEPT": "good", "REJECT": "bad"}.get(v),
    "reversed": lambda v: "warn" if v is True else None,
    "p_fail": lambda v: None if v is None or v != v else "bad" if v > 0.5 else "warn" if v > 0.05 else "good",
    "warnings": lambda v: "warn" if v else None,
    "risk_level": lambda v: {"Low": "good", "Medium": "warn", "High": "bad"}.get(v),
}

_WITHIN_SHORT = {"amr": "AMR", "mmr": "MMR", "srmssd": "SRMSSD", "pooled": "Pooled", "rbar": "R-bar", "sbar": "S-bar"}


def _within_label(s) -> str:
    """Within-sigma estimator actually used, e.g. "AMR" or "MMR, span 3"; "not used" for non-normal data."""
    if s.method != "normal" or s.sigma_within != s.sigma_within:
        return "not used"
    label = _WITHIN_SHORT.get(s.sigma_method, str(s.sigma_method))
    span = (s.settings or {}).get("mr_span", 2)
    return f"{label}, span {span}" if s.sigma_method in ("amr", "mmr") and span != 2 else label


DEFAULT_COLUMNS = [
    "dimension", "target", "LSL", "USL", "spec_width", "yield", "n", "mean", "sd_within", "sd_within_method",
    "sd_overall", "normality", "p_value", "shape", "best_fit", "method",
    "Cp", "Cpk", "Pp", "Ppk", "ppk_mode", "p_fail", "required", "threshold_decision",
    "decision", "risk_score", "risk_level", "risk_driver", "risk_action", "warnings",
]


@dataclass
class BatchResult:
    studies: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    risks: dict = field(default_factory=dict)  # dimension -> HybridRisk (UC-Cap)
    frame: pd.DataFrame = None  # every computed column, one row per dimension

    def table(self, columns=None, rename: Optional[dict] = None, decimals: Optional[int] = 4) -> pd.DataFrame:
        """Pick, order and rename columns of the results table.

        Numbers are rounded to ``decimals`` places (4 by default) for display and export;
        ``decimals=None`` keeps full precision, as does ``self.frame``.
        """
        cols = [c for c in (columns or DEFAULT_COLUMNS) if c in self.frame.columns]
        out = self.frame[cols]
        if decimals is not None:
            out = out.round(decimals)
        return out.rename(columns=rename) if rename else out

    def subset(self, dimensions) -> "BatchResult":
        """The results of the named dimensions only, in their original order (for export or display)."""
        keep = {str(d) for d in dimensions}
        pick = lambda d: {k: v for k, v in d.items() if str(k) in keep}
        frame = self.frame[self.frame["dimension"].astype(str).isin(keep)].reset_index(drop=True)
        return BatchResult(pick(self.studies), pick(self.decisions), pick(self.errors), pick(self.risks), frame)

    def report(self, dimension) -> str:
        dimension = str(dimension)
        if dimension in self.errors:
            return f"{dimension}: {self.errors[dimension]}"
        text = self.studies[dimension].report()
        if dimension in self.decisions:
            text += "\n" + self.decisions[dimension].report()
        if dimension in self.risks:
            text += "\n" + self.risks[dimension].report()
        return text

    def plot(self, ax=None):
        """Estimate and lower bound per dimension against C0 (needs matplotlib)."""
        from .plot import batch_plot
        return batch_plot(self, ax)

    def to_html(self, path=None, columns=None, rename=None, rules=None, title="Capability results") -> str:
        from .export import decision_notes, table_html
        html = table_html(self.table(columns), rules or DEFAULT_RULES, title, rename, notes=decision_notes(self))
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    def to_excel(self, path, columns=None, rename=None, rules=None, details: bool = True):
        from .export import table_excel
        table_excel(self, path, self.table(columns), rules or DEFAULT_RULES, rename, details)

    def _repr_html_(self):
        from .export import decision_notes, table_html
        return table_html(self.table(), DEFAULT_RULES, None, None, standalone=False, notes=decision_notes(self))


def _resolve_input(data, specs):
    """Accept a file path, pasted text, a TableInput or a table with specification rows."""
    import os
    from .io import TableInput, read_table
    if isinstance(data, (str, os.PathLike)):
        data = read_table(data)
    elif isinstance(data, pd.DataFrame) and specs is None:
        first = data.iloc[:, 0].astype(str) if data.shape[1] else pd.Series(dtype=str)
        if any(_norm(v) in {_norm(a) for k in ("lsl", "usl", "target") for a in ALIASES[k]} for v in first):
            data = read_table(data)
    if isinstance(data, TableInput):
        if specs is None:
            specs = data.specs or None
        data = data.data
    return data, specs


def batch_study(data, specs=None, *, c0: Optional[float] = 1.33, alpha: Optional[float] = None,
                cost_ratio: Optional[float] = None, n_boot: int = 1000, rng=0,
                hybrid: bool = True, risk_thresholds: Optional[tuple] = None, risk_costs: Optional[tuple] = None,
                zero_lsl_gdt: bool = True, progress: bool = False, **study_kwargs) -> BatchResult:
    """Run :func:`capability_study` and :func:`risk_decision` for every dimension.

    ``data`` may be a file path or pasted text in the standard layout (specification
    rows above the measurements, see :func:`uccap.read_table`), in which case
    ``specs`` can be omitted; a DataFrame, dict or :class:`~uccap.io.TableInput`.

    Dimensions without a specification, or whose data cannot support a study, get
    a row with ``decision = "ERROR"`` and the reason in ``warnings``. Set ``c0=None``
    to skip approval decisions. Without ``alpha`` or ``cost_ratio``, alpha = 0.05.

    With ``hybrid=True`` every dimension with both limits or an upper limit only gets the UC-Cap risk
    (:func:`uccap.hybrid_risk`) in ``risk_p``, ``risk_score`` (0-100) and ``risk_level``.
    The model is fixed at C0 = 1.33 whatever ``c0`` is. A dimension with an upper limit only (GD&T)
    is coded with LSL = 0, Ppk = Ppu and no Pp, as in the training data; a lower limit alone gets no score.
    GD&T characteristics are commonly recorded as nominal 0, LSL 0 and USL > 0. With
    ``zero_lsl_gdt=True`` (default) such a dimension is read as an upper limit only (Ppu), 0 being a
    physical bound, and a warning says so. A real lower limit at 0 (for example a step between two
    parts that may deviate to one side only) is entered with LSL = 0 and no nominal, or with
    ``zero_lsl_gdt=False``; it gives the two-sided tolerance [0, USL].
    ``risk_thresholds`` or ``risk_costs`` set the risk bands as in :func:`uccap.hybrid_risk`,
    and the decision layer of the paper (score -> level -> reason -> action) fills
    ``risk_driver`` and ``risk_action`` (compact labels, as in the paper's Table 1) and
    ``risk_reason`` and ``risk_action_detail`` (the same, explained in full). The score ranks dimensions for
    review; the engineer confirms the process condition and decides the final action.
    The UC-Cap score needs the trained parameters, which are not included in this release (see
    :func:`uccap.hybrid.set_model_path`); without them the risk columns are left out with a warning.
    """
    data, specs = _resolve_input(data, specs)
    if hybrid and not model_available():
        warnings.warn("UC-Cap risk columns left out: the trained parameters are not included in this release "
                      "(available from the authors on request; see uccap.hybrid.set_model_path)", stacklevel=2)
        hybrid = False
    series = split_dimensions(data)
    spec_map = parse_specs(specs)
    unused = [str(k) for k in spec_map if k not in series]
    if unused:
        more = f" and {len(unused) - 5} more" if len(unused) > 5 else ""
        warnings.warn(f"specifications for dimensions that are not in the data: {', '.join(unused[:5])}{more}",
                      stacklevel=2)
    if alpha is None and cost_ratio is None:
        alpha = 0.05
    result = BatchResult()
    rows = []
    for i, (dim, x) in enumerate(series.items(), 1):
        if progress:
            print(f"[{i}/{len(series)}] {dim}", flush=True)
        spec = spec_map.get(dim, {})
        row = {"dimension": dim, "target": spec.get("target"), "LSL": spec.get("lsl"), "USL": spec.get("usl")}
        gdt = (zero_lsl_gdt and spec.get("lsl") == 0 and spec.get("target") == 0
               and spec.get("usl") is not None and spec["usl"] > 0)
        if gdt:  # 0 is a physical bound, not a limit: upper limit only
            spec = {**spec, "lsl": None}
        if row["LSL"] is not None and row["USL"] is not None and not gdt:
            row["spec_width"] = row["USL"] - row["LSL"]
        try:
            if row["LSL"] is None and row["USL"] is None:
                raise ValueError("no specification limits")
            s = capability_study(x, spec.get("lsl"), spec.get("usl"), spec.get("target"), **study_kwargs)
        except Exception as exc:
            result.errors[dim] = str(exc)
            row.update(decision="ERROR", warnings=str(exc))
            rows.append(row)
            continue
        result.studies[dim] = s
        d = s.data
        inside = np.ones(d.size, bool)
        if s.spec.lsl is not None:
            inside &= d >= s.spec.lsl
        if s.spec.usl is not None:
            inside &= d <= s.spec.usl
        row.update({
            "n": s.n, "yield": f"{int(inside.sum())}|{s.n}", "in_spec": int(inside.sum()),
            "mean": s.mean, "sd_overall": s.sigma_overall,
            "sd_within": s.sigma_within if s.method == "normal" else float("nan"),
            "normality": "Pass" if s.normality.is_normal else "Fail", "p_value": s.normality.p_value,
            "skewness": s.skewness, "excess_kurtosis": s.excess_kurtosis,
            "shape": _shape(s.skewness, s.excess_kurtosis),
            "best_fit": DISPLAY_NAMES.get(s.fit.name, s.fit.name) if s.fit else "Normal",
            "method": ("Normal" if s.method == "normal"
                       else "Empirical" if s.percentile_method == "empirical" else "Percentile"),
            "sd_within_method": _within_label(s),
            "n_outliers": int(s.outlier_mask.sum()), "n_excluded": s.n_excluded, "n_missing": s.n_missing,
            "capability_index": s.cpk_name, "capability": s.cpk,
            "warnings": "; ".join((["LSL = 0 read as a GD&T bound (upper limit only)"] if gdt else []) + list(s.warnings)),
        })
        for k, v in s.indices.items():
            row[k] = v
        if s.method == "percentile":
            # Overall capability columns always hold the approval index; Cp/Cpk (within
            # sigma, normal theory) stay empty for non-normal data.
            row["Pp"], row["Ppk"] = s.indices.get("CNp"), s.indices["CNpk"]
            row["ppk_mode"] = "Empirical" if s.percentile_method == "empirical" else "Parametric"
        else:
            row["ppk_mode"] = "Normal"
        if c0 is not None:
            dec = risk_decision(s, c0=c0, alpha=None if cost_ratio is not None else alpha,
                                cost_ratio=cost_ratio, n_boot=n_boot, rng=rng)
            result.decisions[dim] = dec
            row.update({
                "p_fail": dec.p_fail, "required": dec.margin,
                "threshold_decision": "ACCEPT" if dec.threshold_accept else "REJECT",
                "decision": "ACCEPT" if dec.accept else "REJECT",
                "reversed": dec.accept != dec.threshold_accept,
            })
        if hybrid and s.spec.usl is None:
            # UC-Cap covers bilateral tolerances and upper limits only (GD&T, coded LSL = 0).
            row["risk_level"] = "n/a (lower limit only)"
        if hybrid and s.spec.usl is not None:
            try:
                r = hybrid_risk(x.to_numpy(float), s.spec.lsl, s.spec.usl, s.spec.target, rng=rng,
                                thresholds=risk_thresholds, costs=risk_costs)
            except Exception as exc:  # the score is supplementary: never fail the row
                row["warnings"] = "; ".join(filter(None, [row["warnings"], f"UC-Cap: {exc}"]))
            else:
                result.risks[dim] = r
                row.update(risk_p=r.p, risk_score=round(r.score, 1), risk_level=r.level,
                           risk_driver=r.driver, risk_action=r.action, risk_reason=r.reason,
                           risk_action_detail=r.action_detail)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for c in ("n", "in_spec", "n_outliers", "n_excluded", "n_missing"):
        if c in frame:
            frame[c] = frame[c].astype("Int64")
    result.frame = frame
    return result
