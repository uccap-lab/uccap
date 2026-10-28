"""Coloured HTML and Excel output for batch results."""
from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd

COLOURS = {"good": ("#d8f0dc", "#1e6b34"), "warn": ("#fff1c2", "#7a5a00"), "bad": ("#f9d6d9", "#a3202e")}


def _missing(v) -> bool:
    try:
        return v is None or bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _fmt(v) -> str:
    if _missing(v):
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if v == 0 or 1e-3 <= abs(v) < 1e5:
            return f"{v:.4f}"
        return f"{v:.3e}"
    return str(v)


def _level(rules, col, v):
    rule = rules.get(col)
    if rule is None or _missing(v):
        return None
    try:
        return rule(v)
    except Exception:
        return None


# Header tooltips: what each column holds (manual, Table "Result columns").
_WITHIN = "with the within sigma (normal method only)"
_OVERALL = "with the overall sigma"
COLUMN_HELP = {
    "dimension": "Dimension name.",
    "target": "Target (nominal) value.",
    "LSL": "Lower specification limit.",
    "USL": "Upper specification limit.",
    "spec_width": "Tolerance width, USL - LSL.",
    "n": "Number of usable observations.",
    "yield": "Parts within the specification limits, shown as in_spec|n.",
    "in_spec": "Number of parts within the specification limits.",
    "mean": "Sample mean.",
    "sd_within": "Within (short-term) sigma; normal method only.",
    "sd_within_method": "Estimator of the within sigma (AMR, MMR, SRMSSD, Pooled, R-bar or S-bar); "
                        "'not used' when the method is not Normal.",
    "sd_overall": "Overall sigma: the sample standard deviation of all parts.",
    "normality": "Anderson-Darling normality test at alpha = 0.05: Pass or Fail.",
    "p_value": "p-value of the Anderson-Darling normality test.",
    "skewness": "Sample skewness g1.",
    "excess_kurtosis": "Sample excess kurtosis.",
    "shape": "Sample shape: normal-like, skewed (|g1| >= 0.5), heavy-tailed (excess kurtosis >= 1) "
             "or strongly non-normal (|g1| >= 1 or excess kurtosis >= 3).",
    "best_fit": "Distribution used; for non-normal data the candidate with the lowest AICc.",
    "method": "Normal (normal-based indices), Percentile (indices from the fitted distribution) or Empirical.",
    "n_outliers": "Points flagged by the IQR outlier screen.",
    "n_excluded": "Points excluded from the analysis.",
    "n_missing": "Missing or non-numeric readings.",
    "capability_index": "Name of the index used for approval.",
    "capability": "Value of the index used for approval: the estimate both approval rules compare.",
    "warnings": "Data-quality and method warnings.",
    "Cp": f"Potential capability {_WITHIN}.", "Cpk": f"Capability including centring {_WITHIN}.",
    "Cpu": f"Upper one-sided index {_WITHIN}.", "Cpl": f"Lower one-sided index {_WITHIN}.",
    "Cpm": f"Target-based index {_WITHIN}.", "Cpmk": f"Target-based index including centring {_WITHIN}.",
    "Pp": f"Potential capability {_OVERALL}; for non-normal data CNp.",
    "Ppk": f"Capability including centring {_OVERALL}; for non-normal data CNpk.",
    "Ppu": f"Upper one-sided index {_OVERALL}.", "Ppl": f"Lower one-sided index {_OVERALL}.",
    "Ppm": f"Target-based index {_OVERALL}.", "Ppmk": f"Target-based index including centring {_OVERALL}.",
    "CNp": "Percentile-based potential capability (non-normal data).",
    "CNpk": "Percentile-based capability including centring (non-normal data).",
    "CNpu": "Percentile-based upper index (non-normal data).", "CNpl": "Percentile-based lower index (non-normal data).",
    "CNpm": "Percentile-based target index (non-normal data).", "CNpmk": "Percentile-based target index including centring (non-normal data).",
    "ppk_mode": "How Ppk was computed: Normal, Parametric (fitted distribution) or Empirical.",
    "p_fail": "Probability that the true capability is below C0: analytic for normal data, "
              "nonparametric bootstrap for non-normal data.",
    "required": "C0 + k x SE at the current estimate: the smallest estimate the risk-calibrated rule "
                "accepts (analytic; for reference when the bootstrap decides).",
    "threshold_decision": "Plain threshold rule: ACCEPT when the estimate is at least C0.",
    "decision": "Risk-calibrated rule: ACCEPT when p_fail is at most alpha.",
    "reversed": "True when the threshold rule and the risk-calibrated rule disagree.",
    "risk_p": "UC-Cap: estimated probability that the true Ppk is below 1.33. An upper limit alone "
              "(GD&T) is coded LSL = 0 with Ppk = Ppu and no Pp.",
    "risk_score": "UC-Cap risk score, 100 x risk_p.",
    "risk_level": "Low, Medium or High from illustrative probability bands; the conservative layer may raise "
                  "Low to Medium. 'n/a (lower limit only)': no score for a lower limit alone.",
    "risk_driver": "Reason behind the risk (compact label): the mechanism signals found.",
    "risk_action": "Suggested follow-up (compact label). Decision support; the engineer decides.",
    "risk_reason": "The reason in full, with the statistics behind it.",
    "risk_action_detail": "The suggested follow-up in full.",
    # Operational acceptability monitoring (uccap.monitor)
    "subgroups": "Number of complete subgroups monitored, in production order.",
    "pi_final": "Posterior acceptability probability pi = P(q >= q0 | data) at the last subgroup.",
    "pi_min": "Lowest pi from subgroup t_min on.",
    "first_intervention": "First subgroup in the intervene zone (pi <= gamma_R, t >= t_min); empty if none.",
    "Ppk_normal": "Ppk from all values with the overall sigma (normal theory), for comparison with pi_final.",
    "zone": "Zone of the last subgroup: continue, monitor or intervene.",
    "category": "Monitoring category: persistent low-confidence, recovered transient, no intervention-zone entry "
                "or borderline final.",
}


def column_help(col) -> str:
    """Tooltip text for a result column ('' when unknown)."""
    return COLUMN_HELP.get(col, "")


def decision_notes(result) -> list:
    """How the decisions in ``result`` were made, as (heading, text) pairs, with its own settings."""
    dec = next(iter(result.decisions.values()), None)
    notes = []
    if dec is not None:
        a = f"alpha = {dec.alpha:.4g}" + (f" (from the cost ratio {dec.cost_ratio:g}: alpha = 1/(1 + ratio))"
                                         if dec.cost_ratio is not None else "")
        notes += [
            ("Threshold rule (threshold_decision)",
             f"ACCEPT when the capability estimate is at least C0 = {dec.c0:g}. It ignores how uncertain the "
             "estimate is."),
            ("Risk-calibrated rule (decision)",
             f"ACCEPT when the probability that the true capability is below C0 (p_fail) is at most {a}. "
             f"For normal data p_fail = Phi((C0 - estimate) / SE) with SE = sqrt((1/9 + estimate^2/2) / n), which "
             f"is the same as: estimate at least required = C0 + k x SE, k = z(1 - alpha) = {dec.k:.4f}. For "
             "non-normal data (percentile method) p_fail comes from a nonparametric bootstrap (resample, refit); "
             "required is then shown for reference."),
            ("Reversed (reversed)", "The two rules disagree. With alpha below 0.5 this happens when the estimate is at "
             "least C0 but not far enough above it for its uncertainty."),
        ]
    risk = next(iter(result.risks.values()), None)
    if risk is not None:
        lo, hi = risk.thresholds
        notes += [
            ("UC-Cap score (risk_p, risk_score)",
             "The hybrid model's estimate of the probability that the true Ppk is below 1.33: a statistical "
             "anchor plus a learned correction. A dimension with an upper limit only (GD&T) is coded as in "
             "the training data: LSL = 0, nominal = 0, Ppk = Ppu, Pp not computed (no mean-shift or spread signal). "
             "A lower limit alone gets no score."),
            ("Risk level (risk_level)",
             f"Low when risk_p < {lo:g}, Medium when {lo:g} <= risk_p < {hi:g}, High when risk_p >= {hi:g}. "
             "Two or more mechanism signals raise Low to Medium."),
            ("Reason and action (risk_driver, risk_action)",
             "Mechanism signals: mean shift when Pp - Ppk > SE(Ppk); excessive variation when Pp < 1.00; skewed "
             "when the Anderson-Darling test rejects normality at 0.05. Low without a compound mechanism: "
             "Acceptable, Accept. One signal: its own reason and action. Two or more: Mixed mechanism. Medium "
             "or High without a signal: Latent model risk, Investigate latent risk."),
        ]
    if notes:
        notes.append(("Read with care",
                      "These rules support decisions under limited samples; they do not replace the capability "
                      "requirement itself. The UC-Cap bands and signal rules are illustrative, not general "
                      "standards, and the score orders the review: the engineer confirms the process condition "
                      "and chooses the action. Each dimension's report gives the numbers behind its decision."))
    return notes


def notes_html(notes) -> str:
    if not notes:
        return ""
    items = "".join(f"<dt>{_html.escape(h)}</dt><dd>{_html.escape(t)}</dd>" for h, t in notes)
    return f'<details class="how"><summary>How the decisions are made</summary><dl>{items}</dl></details>'


CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --head:#f6f8fa; --hover:#f3f6fa; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#30363d; --head:#161b22; --hover:#1c2129; } }
body { background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; margin:16px; }
h1 { font-size:18px; margin:0 0 4px; }
p.meta { color:var(--muted); margin:0 0 12px; }
.wrap { overflow:auto; max-height:85vh; border:1px solid var(--line); border-radius:6px; }
table.uccap { border-collapse:collapse; width:max-content; min-width:100%; }
table.uccap th, table.uccap td { padding:6px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
table.uccap th { position:sticky; top:0; background:var(--head); text-align:center; cursor:pointer; user-select:none; z-index:1; }
table.uccap th:after { content:" \\21C5"; color:var(--muted); font-size:11px; }
table.uccap td.num { text-align:right; font-variant-numeric:tabular-nums; }
table.uccap td.txt { text-align:center; }
table.uccap td.first { text-align:left; font-weight:600; }
table.uccap td.long { text-align:left; max-width:420px; overflow:hidden; text-overflow:ellipsis; }
table.uccap tr:hover td { background-color:var(--hover); }
.tools { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; margin:0 0 10px; }
.seg { display:inline-flex; align-items:center; gap:0; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
.seg span { padding:4px 10px; color:var(--muted); font-size:13px; }
.seg button { font:inherit; font-size:13px; padding:4px 12px; border:0; border-left:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
.seg button[aria-pressed="true"] { background:var(--head); font-weight:600; }
.cols { position:relative; }
.cols summary { cursor:pointer; padding:4px 12px; border:1px solid var(--line); border-radius:6px; font-size:13px; list-style:none; }
.cols .list { position:absolute; z-index:5; margin-top:4px; padding:8px 12px; background:var(--bg); border:1px solid var(--line); border-radius:6px;
  display:grid; grid-template-columns:repeat(3, minmax(130px, auto)); gap:2px 16px; font-size:13px; box-shadow:0 4px 16px rgba(0,0,0,.12); }
.cols .list label { white-space:nowrap; }
.how { margin:0 0 12px; border:1px solid var(--line); border-radius:6px; padding:6px 12px; max-width:960px; }
.how summary { cursor:pointer; font-weight:600; }
.how dl { margin:8px 0 4px; }
.how dt { font-weight:600; margin-top:8px; }
.how dd { margin:2px 0 0; color:var(--fg); line-height:1.5; }
table.uccap th[title] { text-decoration:underline dotted var(--muted); text-underline-offset:3px; }
table.uccap .off { display:none; }
""" + "".join(f"table.uccap td.{k} {{ background:{bg}; color:{fg}; }}\n" for k, (bg, fg) in COLOURS.items())

JS = """
(() => {
  const table = document.querySelector('table.uccap'); if (!table) return;
  const heads = [...table.tHead.rows[0].cells];
  const boxes = [...document.querySelectorAll('.cols input[type=checkbox]')];
  const store = { get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
                  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} } };
  function apply() {
    const off = new Set(boxes.filter(b => !b.checked).map(b => +b.value));
    for (const row of table.rows) [...row.cells].forEach((c, i) => c.classList.toggle('off', off.has(i)));
    store.set('uccap-hidden', [...off].map(i => heads[i].textContent));
  }
  function setSigma(mode) {
    document.querySelectorAll('.seg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    boxes.forEach(b => { const g = heads[+b.value].dataset.sigma;
      if (g) b.checked = mode === 'both' || mode === g; });
    store.set('uccap-sigma', mode); apply();
  }
  boxes.forEach(b => b.addEventListener('change', apply));
  document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => setSigma(b.dataset.mode)));
  const hidden = store.get('uccap-hidden');
  if (Array.isArray(hidden)) boxes.forEach(b => { if (!heads[+b.value].dataset.sigma) b.checked = !hidden.includes(heads[+b.value].textContent); });
  setSigma(store.get('uccap-sigma') || 'overall');
})();
document.querySelectorAll('table.uccap th').forEach((th, i) => th.addEventListener('click', () => {
  const tb = th.closest('table').tBodies[0], rows = [...tb.rows];
  const dir = th.dataset.dir === 'asc' ? -1 : 1; th.dataset.dir = dir === 1 ? 'asc' : 'desc';
  const key = r => { const t = r.cells[i].dataset.v ?? r.cells[i].textContent; const n = parseFloat(t);
                     return t === '' ? null : isNaN(n) ? t : n; };
  rows.sort((a, b) => { const x = key(a), y = key(b); if (x === null) return 1; if (y === null) return -1;
    return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * dir; });
  rows.forEach(r => tb.appendChild(r));
}));
"""


WITHIN_COLUMNS = {"sd_within", "sd_within_method", "sigma_within", "Cp", "Cpk", "Cpu", "Cpl", "Cpm", "Cpmk"}
OVERALL_COLUMNS = {"sd_overall", "Pp", "Ppk", "Ppu", "Ppl"}


def _sigma_group(col) -> str:
    if col in WITHIN_COLUMNS:
        return ' data-sigma="within"'
    if col in OVERALL_COLUMNS:
        return ' data-sigma="overall"'
    return ""


def _toolbar(df: pd.DataFrame, rename) -> str:
    seg = ""
    if any(c in WITHIN_COLUMNS for c in df.columns) and any(c in OVERALL_COLUMNS for c in df.columns):
        seg = ('<div class="seg" role="group" aria-label="Sigma columns"><span>Sigma</span>'
               + "".join(f'<button type="button" data-mode="{m}" aria-pressed="false">{label}</button>'
                         for m, label in (("within", "Within"), ("overall", "Overall"), ("both", "Both")))
               + "</div>")
    boxes = "".join(f'<label><input type="checkbox" id="col-{j}" value="{j}" checked> {_html.escape(str(rename.get(c, c)))}</label>'
                    for j, c in enumerate(df.columns) if j > 0)
    return f'<div class="tools">{seg}<details class="cols"><summary>Columns</summary><div class="list">{boxes}</div></details></div>'


def table_html(df: pd.DataFrame, rules, title=None, rename=None, standalone: bool = True, notes=None) -> str:
    rename = rename or {}
    tip = lambda c: f' title="{_html.escape(column_help(c))}"' if column_help(c) else ""
    head = "".join(f"<th tabindex=\"0\"{_sigma_group(c)}{tip(c)}>{_html.escape(str(rename.get(c, c)))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for j, c in enumerate(df.columns):
            v = row[c]
            numeric = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            text = _fmt(v)
            classes = ["first" if j == 0 else "num" if numeric else "long" if len(text) > 30 else "txt"]
            lvl = _level(rules, c, v)
            if lvl:
                classes.append(lvl)
            sort_key = f' data-v="{v}"' if numeric and not _missing(v) else ""
            tip = f' title="{_html.escape(text)}"' if "long" in classes else ""
            cells.append(f'<td class="{" ".join(classes)}"{sort_key}{tip}>{_html.escape(text)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    table = notes_html(notes) + _toolbar(df, rename) + f'<div class="wrap"><table class="uccap"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    if not standalone:
        return f"<style>{CSS}</style>{table}"
    heading = f"<h1>{_html.escape(title)}</h1>" if title else ""
    meta = (f'<p class="meta">{len(df)} dimensions · generated by uccap · click a column header to sort, '
            'point at it for its meaning</p>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_html.escape(title or 'uccap results')}</title><style>{CSS}</style></head>"
            f"<body>{heading}{meta}{table}<script>{JS}</script></body></html>")


def table_excel(result, path, df: pd.DataFrame, rules, rename=None, details: bool = True, notes=None,
                notes_sheet: str = "How decided"):
    try:
        from openpyxl import Workbook
        from openpyxl.comments import Comment
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Excel export needs openpyxl: pip install openpyxl") from exc
    rename = rename or {}
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append([rename.get(c, c) for c in df.columns])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.fill = PatternFill("solid", fgColor="EEF1F4")
    for j, c in enumerate(df.columns, start=1):
        if column_help(c):
            ws.cell(row=1, column=j).comment = Comment(column_help(c), "uccap", width=260, height=90)
    fills = {k: (PatternFill("solid", fgColor=bg[1:]), Font(color=fg[1:])) for k, (bg, fg) in COLOURS.items()}
    for r, (_, row) in enumerate(df.iterrows(), start=2):
        for j, c in enumerate(df.columns, start=1):
            v = row[c]
            if _missing(v):
                v = None
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            cell = ws.cell(row=r, column=j, value=v)
            if isinstance(v, float):
                cell.number_format = "0.0000"
            lvl = _level(rules, c, row[c])
            if lvl:
                cell.fill, cell.font = fills[lvl]
    for j, c in enumerate(df.columns, start=1):
        width = max([len(str(rename.get(c, c)))] + [len(_fmt(v)) for v in df[c]]) + 2
        ws.column_dimensions[get_column_letter(j)].width = min(width, 60)
    # Within-sigma columns form a collapsible group, collapsed by default.
    for j, c in enumerate(df.columns, start=1):
        if c in WITHIN_COLUMNS:
            ws.column_dimensions[get_column_letter(j)].outlineLevel = 1
            ws.column_dimensions[get_column_letter(j)].hidden = True
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    notes = decision_notes(result) if notes is None else notes
    if notes:
        wn = wb.create_sheet(notes_sheet)
        for h, t in notes:
            wn.append([h])
            wn.cell(row=wn.max_row, column=1).font = Font(bold=True)
            wn.append([t])
            wn.cell(row=wn.max_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
            wn.append([])
        wn.column_dimensions["A"].width = 110
    if details:
        wr = wb.create_sheet("Reports")
        mono = Font(name="Courier New")
        for dim in df[df.columns[0]] if df.columns[0] == "dimension" else result.frame["dimension"]:
            for line in result.report(dim).splitlines() + ["", ""]:
                wr.append([line])
                wr.cell(row=wr.max_row, column=1).font = mono
        wr.column_dimensions["A"].width = 80
    wb.save(path)
