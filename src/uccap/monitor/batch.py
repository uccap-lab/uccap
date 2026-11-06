"""Operational acceptability monitoring of many dimensions at once (the industrial replay of Sec. 5.7).

Every dimension is monitored on its own (:func:`monitor_study`); the result has one row per dimension
with its final and lowest pi, its first intervention, its last zone and its monitoring category
(Table 14), plus the full trajectory of each dimension.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .sboam import CATEGORIES, monitor_study

DEFAULT_COLUMNS = ["dimension", "target", "LSL", "USL", "n", "subgroups", "mean", "sd_overall", "Ppk_normal",
                   "pi_final", "pi_min", "first_intervention", "zone", "category", "warnings"]

ZONE_RULE = {"continue": "good", "monitor": "warn", "intervene": "bad"}
CATEGORY_RULE = {CATEGORIES[0]: "bad", CATEGORIES[1]: "warn", CATEGORIES[2]: "good", CATEGORIES[3]: "warn"}


def _rules(settings: dict) -> dict:
    ga, gr = settings.get("gamma_a", 0.95), settings.get("gamma_r", 0.80)
    pi_rule = lambda v: None if v != v else "good" if v >= ga else "bad" if v <= gr else "warn"
    return {"zone": ZONE_RULE.get, "category": CATEGORY_RULE.get, "pi_final": pi_rule, "pi_min": pi_rule,
            "warnings": lambda v: "warn" if v else None}


@dataclass
class MonitorBatchResult:
    results: dict = field(default_factory=dict)   # dimension -> MonitorResult
    errors: dict = field(default_factory=dict)
    frame: pd.DataFrame = None                    # one row per dimension, every computed column
    settings: dict = field(default_factory=dict)

    def table(self, columns=None, rename: Optional[dict] = None, decimals: Optional[int] = 4) -> pd.DataFrame:
        """Pick, order and rename columns; numbers rounded to ``decimals`` places (None keeps them)."""
        cols = [c for c in (columns or DEFAULT_COLUMNS) if c in self.frame.columns]
        out = self.frame[cols]
        if decimals is not None:
            out = out.round(decimals)
        return out.rename(columns=rename) if rename else out

    def trajectories(self, decimals: Optional[int] = 4) -> pd.DataFrame:
        """Every subgroup of every dimension in one long table (dimension, t, n, mean, ..., pi, zone)."""
        parts = [r.frame.assign(dimension=d) for d, r in self.results.items()]
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out = out[["dimension"] + [c for c in out.columns if c != "dimension"]]
        return out.round(decimals) if decimals is not None else out

    def report(self, dimension) -> str:
        dimension = str(dimension)
        if dimension in self.errors:
            return f"{dimension}: {self.errors[dimension]}"
        return self.results[dimension].report()

    def plot(self):
        """Population overview: final pi, empirical capability against final pi, first interventions."""
        from ..plot import monitor_overview
        return monitor_overview(self)

    def notes(self) -> list:
        return monitor_notes(self.settings)

    def to_html(self, path=None, columns=None, rename=None, rules=None, title="Operational acceptability monitoring") -> str:
        from ..export import table_html
        html = table_html(self.table(columns), rules or _rules(self.settings), title, rename, notes=self.notes())
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    def to_excel(self, path, columns=None, rename=None, rules=None, details: bool = True, trajectories: bool = True):
        """Coloured workbook: Results, How decided, Reports (``details``) and Trajectories."""
        from ..export import table_excel
        table_excel(self, path, self.table(columns), rules or _rules(self.settings), rename, details,
                    notes=self.notes(), notes_sheet="How decided")
        if trajectories and self.results:
            from openpyxl import load_workbook
            from openpyxl.styles import Font, PatternFill
            wb = load_workbook(path)
            ws = wb.create_sheet("Trajectories")
            tr = self.trajectories()
            ws.append(list(tr.columns))
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="EEF1F4")
            for row in tr.itertuples(index=False):
                ws.append([None if (isinstance(v, float) and math.isnan(v)) else
                           (v.item() if isinstance(v, np.generic) else v) for v in row])
            for col in ws.iter_cols(min_row=2):
                for cell in col:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"
            ws.freeze_panes = "B2"
            wb.save(path)

    def _repr_html_(self):
        from ..export import table_html
        return table_html(self.table(), _rules(self.settings), None, None, standalone=False, notes=self.notes())


def monitor_notes(s: dict) -> list:
    """How the zones and categories are assigned, with the settings used, as (heading, text) pairs."""
    if not s:
        return []
    gauge = (f" Gauge-corrected: q is evaluated on sigma^2 - {s['gauge_sd']:g}^2 for every draw (Sec. 5.5)."
             if s.get("gauge_sd") else "")
    return [
        ("Acceptability probability (pi_final, pi_min)",
         "Each dimension is cut into consecutive subgroups of "
         f"{s['subgroup_size']} in the recorded order. After every subgroup, an exponentially discounted "
         f"Normal-Inverse-Gamma posterior of the process mean and standard deviation (forgetting factor "
         f"{s['forgetting']:g}) gives pi_t = P(q >= q0 | data so far), where q = Phi((USL - mu)/sigma) - "
         f"Phi((LSL - mu)/sigma) is the conformance probability and q0 = {s['q0']:g}; pi_t is estimated from "
         f"{s['n_draws']} posterior draws. pi_final is pi at the last subgroup, pi_min the lowest pi from "
         f"subgroup t_min = {s['t_min']} on. The default (tolerance-aware) prior is centred at the nominal with "
         f"scale (USL - LSL)/8 and carries almost no weight.{gauge}"),
        ("Zones (zone, first_intervention)",
         f"continue when pi >= {s['gamma_a']:g}; intervene when pi <= {s['gamma_r']:g} and t >= {s['t_min']}; "
         "monitor otherwise. zone is the zone of the last subgroup, first_intervention the first subgroup in the "
         "intervene zone."),
        ("Category (category)",
         f"persistent low-confidence: pi_final <= {s['gamma_r']:g}; recovered transient: pi_min <= "
         f"{s['gamma_r']:g} and pi_final >= {s['gamma_a']:g}; no intervention-zone entry: pi_min > "
         f"{s['gamma_r']:g}; borderline final: the rest. Assigned in this order."),
        ("Ppk_normal",
         "Empirical capability from all values with the overall standard deviation, for comparison with "
         "pi_final; it plays no part in the monitoring."),
        ("Read with care",
         "q0, the thresholds, the forgetting factor, the prior, the subgroup size and t_min are engineering and "
         "operational policy parameters, not universal constants; these defaults are the validation settings "
         "of the article. The Gaussian model is a working approximation: skewed or contaminated data, serial "
         "correlation and gauge error change how often the rule acts. A first intervention at t_min usually "
         "means pi was already low at the start of the record, not that a change began there. The zones support "
         "the engineer's decision after approval; they do not replace control charts or the approval itself."),
    ]


def batch_monitor(data, specs=None, *, zero_lsl_gdt: bool = True, progress: bool = False,
                  **monitor_kwargs) -> MonitorBatchResult:
    """Run :func:`monitor_study` for every dimension of a measurement table.

    ``data`` and ``specs`` are read as in :func:`uccap.batch_study` (a file in the standard layout,
    wide or long tables, dicts). The rows of each dimension must be in production order.
    ``monitor_kwargs`` go to :func:`monitor_study` (subgroup_size, q0, gamma_a, gamma_r, forgetting,
    t_min, n_draws, gauge_sd, prior, rng). With the default prior a dimension with a lower limit only gets no
    monitoring (the tolerance-aware prior needs a tolerance width); an upper limit only, or a GD&T dimension
    recorded as nominal 0 and LSL 0, is monitored against its USL with the prior scaled by USL / 8.
    """
    from ..batch import _resolve_input, parse_specs, split_dimensions
    data, specs = _resolve_input(data, specs)
    series = split_dimensions(data)
    spec_map = parse_specs(specs)
    unused = [str(k) for k in spec_map if k not in series]
    if unused:
        more = f" and {len(unused) - 5} more" if len(unused) > 5 else ""
        warnings.warn(f"specifications for dimensions that are not in the data: {', '.join(unused[:5])}{more}",
                      stacklevel=2)
    result = MonitorBatchResult()
    rows = []
    for i, (dim, x) in enumerate(series.items(), 1):
        if progress:
            print(f"[{i}/{len(series)}] {dim}", flush=True)
        spec = spec_map.get(dim, {})
        lsl, usl, target = spec.get("lsl"), spec.get("usl"), spec.get("target")
        row = {"dimension": dim, "target": target, "LSL": lsl, "USL": usl}
        values = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
        row["n"] = len(values)
        if len(values):
            row["mean"] = float(values.mean())
        if len(values) > 1:
            sd = float(values.std(ddof=1))
            row["sd_overall"] = sd
            gdt = zero_lsl_gdt and lsl == 0 and target == 0 and usl is not None and usl > 0
            sides = [(usl - row["mean"]) / (3 * sd) if usl is not None else None,
                     (row["mean"] - lsl) / (3 * sd) if lsl is not None and not gdt else None]
            sides = [v for v in sides if v is not None]
            if sides and sd > 0:
                row["Ppk_normal"] = min(sides)
        try:
            if lsl is None and usl is None:
                raise ValueError("no specification limits")
            if usl is None and "prior" not in monitor_kwargs:
                raise ValueError("lower limit only: no tolerance width for the prior")
            r = monitor_study(values, lsl, usl, target, zero_lsl_gdt=zero_lsl_gdt, name=dim, **monitor_kwargs)
        except Exception as exc:
            result.errors[dim] = str(exc)
            row.update(zone="n/a", warnings=str(exc))
            rows.append(row)
            continue
        result.results[dim] = r
        result.settings = r.settings
        row.update(subgroups=len(r.frame), pi_final=r.pi_final, pi_min=r.pi_min,
                   first_intervention=r.intervention, zone=r.zone, category=r.category,
                   warnings="; ".join(r.notes))
        rows.append(row)
    frame = pd.DataFrame(rows)
    for c in ("n", "subgroups", "first_intervention"):
        if c in frame:
            frame[c] = frame[c].astype("Int64")
    result.frame = frame
    if not result.settings:  # settings for the notes even when every dimension failed
        defaults = monitor_study.__kwdefaults__
        result.settings = {k: monitor_kwargs.get(k, defaults[k]) for k in
                           ("subgroup_size", "q0", "gamma_a", "gamma_r", "forgetting", "t_min", "n_draws", "gauge_sd")}
    return result
