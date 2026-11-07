# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org).

## Unreleased

- `uccap.monitor`: sequential Bayesian operational acceptability monitoring after approval, from the
  monitoring article (accepted). `monitor_study` updates an exponentially discounted Normal–Inverse-Gamma
  posterior subgroup by subgroup (Algorithm 1) and gives the posterior acceptability probability
  π_t = P(q(θ_t) ≥ q0 | data) with the zone continue, monitor or intervene; `batch_monitor` monitors every
  dimension of a file with the tolerance-aware prior and assigns the monitoring categories of the article's
  Table 14, with coloured HTML and Excel output (including every trajectory). Optional gauge correction
  (`gauge_sd`). Command `uccap monitor`; plots `monitor_figure` and `monitor_overview`;
  `validation/monitor_simulation.py` reproduces Tables 4 and 11; notebook 04 and `examples/monitor_example.py`;
  a manual chapter. The other modules are unchanged.
- `uccap.hybrid`: the trained UC-Cap parameters are not included in this release; they are available from
  the authors on request. `set_model_path`, the environment variable `UCCAP_MODEL` and `uccap --model PATH`
  give their location; `model_available` tells whether they are found. Without them `hybrid_risk` raises
  `ModelNotAvailable`, `batch_study` and the command line leave out the UC-Cap columns and plots with a
  warning, and the model inputs and mechanism signals work as before.
- Plots: the values written by `show_data=True` have as many decimals as the data, at most three; the new
  `decimals` argument (command line `--decimals N`) sets the number.

## 0.1.0 — first public release

- `uccap.workflow`: capability study following the practical PCI workflow (outlier screening,
  Anderson–Darling normality test, AICc distribution selection among 10 default and 10 optional
  candidates, within and overall sigma, normal and percentile-based indices for bilateral, asymmetric
  and unilateral tolerances).
- `uccap.risk`: risk-calibrated approval on Ppk (CNpk for non-normal data) with the analytic standard
  error or a nonparametric bootstrap; the threshold rule, lower confidence bound and probability rule as special cases.
- `sample_size`: parts needed so that a process with an assumed true Cpk is approved with a wanted
  probability (simulated for a centred normal process); `min_accepted`, `acceptance_probability` and
  `uccap.plot.planning_plot`.
- `batch_study`: many dimensions at once from a table, CSV or Excel file, with coloured HTML and Excel
  output and one text report per dimension; UC-Cap risk columns (`risk_p`, `risk_score`,
  `risk_level`) for dimensions with both limits or an upper limit only.
- `read_table`: standard data layout (specifications on top, one row per part below), with detection
  of title rows, summary rows, transposed tables and separate specification tables. Tolerance rows or
  columns (UpperTol / LowerTol, +Tol / -Tol, Tol) are converted to limits with the nominal; labels may
  carry a unit in brackets; unrecognised specification rows are skipped, not read as measurements.
- Plots (`uccap[plot]`), one figure per question: outlier screen with its fences, stability (I-MR
  control charts), capability (histogram, probability plot), risk-calibrated approval (uncertainty and
  the estimate needed against n), UC-Cap (anchor and learned correction, risk score); plus the overview
  and the batch overview. `study.plot(kind=...)`, `decision.plot()`, `risk.plot()`, `result.plot()`.
- Templates for single and multiple dimensions; user manual (typeset PDF with diagrams, and
  its full Markdown version generated from the LaTeX source by `docs/manual_tex/to_markdown.py`).
- Validation against R (`fitdistrplus`, `MASS`, `nortest`) and the NIST/SEMATECH e-Handbook example,
  and reproduction of the articles' published results (workflow case study and constants, risk
  simulation Tables 1 and 2: `validation/workflow_case_study.py`, `validation/risk_simulation.py`,
  `tests/test_paper_reproduction.py`); see `docs/validation.md`.
- Median moving range constants d4 for windows 2 to 10, computed numerically (previously 2 to 5,
  with slightly different values for 3 to 5).
- Example notebooks, one per article, in `examples/notebooks/`.
- `outlier_k`: fence coefficient of the IQR and adjusted-boxplot outlier screens (default 1.5), also in
  `batch_study`.
- `uccap.hybrid`: UC-Cap risk score of the third paper, `hybrid_risk()`, evaluating the V4.2 network with
  numpy.
- UC-Cap decision layer (score -> level -> reason -> action, hybrid article Sections 5.1-5.3): compact
  reason and action labels in the style of the article's Table 1 (`driver`, `action`) and the same in
  full (`reason`, `action_detail`); a conservative layer that raises Low to Medium only for a compound
  mechanism; signals and labels chosen to reproduce the article's Table 1 (spread Pp < 1.00, a Low level
  without a compound mechanism is Acceptable), illustrative rather than a general standard; risk bands from `thresholds=(low, high)` or the three-action costs (`costs=`,
  `review_thresholds`). Batch columns `risk_driver`, `risk_action`, `risk_reason`, `risk_action_detail`;
  options `risk_thresholds`, `risk_costs`.
- UC-Cap best-fit input: for a non-normal sample, the named distribution (lognormal, Weibull, gamma,
  logistic) with the lowest AICc when its Anderson–Darling goodness-of-fit test passes, otherwise
  "empirical"; the percentile indices from that distribution are added as `Fit_Pp` / `Fit_Ppk` and
  printed in the report (the model inputs Pp and Ppk stay empirical, as in the training data).
- Displayed numbers: results tables, HTML/Excel/CSV exports, reports and the validation tables show at
  most 4 decimal places (`BatchResult.table(decimals=None)` and `BatchResult.frame` keep full precision).
- Approval report: with the bootstrap method, the standard error and the required estimate are marked as
  analytic values for reference, since the decision uses the bootstrap probability.
- Asymmetric tolerances use the workflow article's starred indices by default
  (`asymmetric="paper"`); the Pearn–Chen form is optional (`asymmetric="pearn-chen"`).
- Normality: chi-square goodness-of-fit test (`"chi2"`, `chi_square_gof`) as in the workflow article;
  D'Agostino–Pearson K² is now `"k2"`.
- `CapabilityStudy.prediction_interval()` and `tolerance_interval()` (normal studies), shown in the
  report; the approval report lists the estimate required at this estimate and the smallest estimate
  that passes for the sample size, with a note when the two rules differ.
- Plots: `show_data=True` labels the measured values on run charts and outlier plots; legends above the
  axes.
- Command-line tool `uccap` (`python -m uccap`): batch study of a file to HTML, Excel or CSV,
  PNG figures of every dimension (`--plots`, `--show-values`) and sample-size planning (`uccap plan`);
  `uccap.cite()` and `uccap --cite` print the BibTeX entries.
- Batch results: `sd_within_method` names the within-sigma estimator used (replaces `sigma_mode`);
  `BatchResult.subset(dimensions)` keeps the named dimensions for display and export.
- GD&T characteristics: nominal 0 with LSL 0 is read as an upper limit only (Ppu) in `batch_study`
  (`zero_lsl_gdt=True`, default), with a warning; such records
  previously failed with "target must lie inside the specification limits". A real lower limit at 0
  is entered with no nominal or `zero_lsl_gdt=False`.
- UC-Cap for upper-limit-only (GD&T) dimensions: coded as in the training data, LSL = 0, nominal = 0
  and Ppk = Ppu, with Pp not computed (so the mean-shift and spread signals do not apply) and the nearest-limit distance taken to LSL = 0 or USL, whichever is closer; only a lower limit alone gets no score (`n/a (lower limit only)`).
- Exported results explain themselves: column headers carry their meaning (HTML tooltips, Excel
  comments), and *How the decisions are made* (HTML) and the *How decided* sheet (Excel) state the
  rules of the run with its settings (`uccap.export.decision_notes`).
- Input checks: infinite values and numbers with a comma but no decimal point (`4,618`, a decimal or a
  thousands separator) are not read and are listed in the warnings; `risk_decision` checks `n`,
  `cost_ratio`, `n_boot` and the estimate, and accepts NumPy scalars; `hybrid_risk` reports
  non-numeric and identical measurements clearly; `batch_study` warns about specifications for
  dimensions that are not in the data.
- `risk_decision` seeds its bootstrap with `rng=0` by default, as `batch_study` and `hybrid_risk` do,
  so the same data give the same decision (`rng=None` for a fresh seed).
- Issue templates, a pull request template, `SECURITY.md`, Colab links in the
  notebooks, and a Minitab / JMP menu map in `docs/validation.md`.
