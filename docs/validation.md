# Validation

uccap is checked against independent implementations, not only against itself. This page records what
was compared, how, and the result. Everything here can be rerun from the repository.

## Summary

| Check | Reference | Items | Result |
|---|---|---:|---|
| Distribution fits (MLE log-likelihood, fitted P0.135 / P50 / P99.865) | R `fitdistrplus` 1.1-11, `MASS` 7.3-60 | 30 | all agree |
| Normality tests (Anderson–Darling A² and p, Shapiro–Wilk p, Lilliefors p) | R `nortest` 1.0-4, `stats::shapiro.test` | 32 | all agree |
| Mean, overall σ, within σ (average moving range) | R base | 24 | all agree |
| Empirical percentiles P0.135 / P50 / P99.865 | R `quantile(type = 7)` | 8 | all agree |
| Control-chart constants d2(n), c4(n) | numerical integration in R | 33 | all agree |
| Textbook example: Cp, Cpk, Cpl, Cpu | NIST/SEMATECH e-Handbook §6.1.6 | 4 | exact |
| Constants d2, c4, d4 for w = 2…10 | Workflow article, Table 1 | 27 | all agree |
| Case study, dimension 105: σ overall + 18 within-σ estimates, Pp/Ppk, Cp/Cpk | Workflow article, Tables 3–5 | 57 | σ, Pp, Ppk exact; Cp, Cpk within 0.7 % |
| Margin k for λ = 1, 4, 9, 19, 99 | Risk article, Table 1 | 5 | exact |
| Expected loss, deterministic vs risk-calibrated, λ = 2…100 | Risk article, Table 2 (simulation) | 12 | within Monte Carlo error |
| Operating characteristics of the monitoring rule | Monitoring article, Tables 4 and 11 (simulation) | 12 | close; see below |

**127 of 127** R comparisons are within tolerance, and the NIST example matches exactly.
R version: 4.3.3.

## Method

1. `validation/make_data.py` simulates eight data sets of n = 50 (seed 20261027): normal, lognormal,
   Weibull, gamma, exponential, largest extreme value, smallest extreme value and logistic. They are
   stored in `validation/data/*.csv`, so R and Python read identical numbers.
2. `validation/reference.R` computes the reference values with R and writes `validation/reference.json`.
3. `validation/compare.py` computes the same quantities with uccap and prints the table below.
4. `tests/test_validation.py` turns every row into a pytest case, so the comparison runs with the normal
   test suite (R is not needed for that: the reference values are stored in the JSON file).

To regenerate the reference values:

```bash
python validation/make_data.py
Rscript validation/reference.R          # needs fitdistrplus, MASS, nortest, jsonlite
PYTHONPATH=src python validation/compare.py
```

### Tolerances

| Quantity | Tolerance | Why |
|---|---|---|
| Mean, overall σ, empirical percentiles | 10⁻¹² relative | Closed form. |
| Within σ (AMR) | 5 × 10⁻⁴ relative | d2(2) from integration vs tabulated constant. |
| d2(n) / c4(n) | 5 × 10⁻⁴ / 10⁻⁹ absolute | d2 is a numerical integral; c4 is closed form. |
| Anderson–Darling A² / p | 10⁻⁹ / 10⁻⁶ absolute | Same formulas (D'Agostino–Stephens). |
| Shapiro–Wilk p | 10⁻⁴ absolute | Same algorithm (Royston). |
| Lilliefors p | 0.05 absolute | uccap uses a Monte Carlo p-value; `nortest` uses the Dallal–Wilkinson approximation. Largest observed: 0.007. |
| Log-likelihood | R − uccap ≤ 0.001 | Both are maximisers; uccap may find a slightly higher maximum, never a materially lower one. In all 15 fits uccap ≥ R. |
| Fitted percentiles | 0.2 % relative | Optimisers stop at slightly different points on a flat likelihood. Largest observed: 9 × 10⁻⁴. |

## Published results of the articles

`validation/workflow_case_study.py`, `validation/risk_simulation.py` and `validation/monitor_simulation.py` recompute
results printed in the articles with uccap; `tests/test_paper_reproduction.py` runs them with the test suite
(the monitoring study in a short version of 100 replications).

**Workflow article** (doi:10.1007/s00170-026-17782-7), case study of Section 6, dimension 105
(32 parts, target 23.580 ± 0.100). The control-chart constants of Table 1 (d2, c4 and d4 for windows 2 to
10) agree to the four printed decimals. The overall σ and the 18 within-σ estimates of Table 3 (average and
median moving range, windows 2 to 10) agree to the four printed decimals, and Pp = 2.172, Ppk = 1.368
exactly. The 18 within-σ Cp and Cpk of Tables 4 and 5 agree within 0.7 %: Table 2 prints the readings to
three decimals, and the indices were presumably computed before rounding, which moves a within σ in its
fifth decimal (simulating the rounding gives a spread of about 0.5 % for the average moving range).

**Risk article** (doi:10.1007/s00170-026-18284-2), Section 5. Table 1 (k = z₁₋α with α = 1/(1+λ)) is
reproduced exactly. Table 2 compares the expected loss λ · P(accept) of the deterministic rule and the
cost-sensitive rule at n = 32 for a centred normal process just below the requirement, true Cpk = 1.32 (the
near-threshold value of the article's Figure 2), so that every acceptance is a false acceptance. With
10⁶ replications:

| λ | Deterministic, article | uccap | Risk-calibrated, article | uccap | Reduction %, article | uccap |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.793 | 0.802 | 0.483 | 0.497 | 39.16 | 37.98 |
| 5 | 2.037 | 2.005 | 0.565 | 0.551 | 72.29 | 72.50 |
| 10 | 4.021 | 4.010 | 0.553 | 0.545 | 86.26 | 86.41 |
| 20 | 8.052 | 8.020 | 0.527 | 0.505 | 93.46 | 93.70 |
| 50 | 20.038 | 20.050 | 0.479 | 0.437 | 97.61 | 97.82 |
| 100 | 40.567 | 40.101 | 0.375 | 0.387 | 99.08 | 99.03 |

The article used 12,000 replications; every difference is within 2 of its Monte Carlo standard errors.
The cost-sensitive decisions are made with `uccap.risk` (`min_accepted`, identical to `risk_decision`).

**Hybrid article** (doi:10.1007/s00170-026-18833-9). The network parameters were exported from the training
run of the article and reproduced its test metrics (Table 10) on the original data when they were exported.
The data are not public, so this check is not part of the test suite. The parameters are not included in
this release (they are available from the authors on request); the tests that need them are skipped
without them.

**Monitoring article** (accepted), Sections 5.3 and 5.4. `validation/monitor_simulation.py` repeats the simulation
study with B = 1000 replications per scenario and the settings of the article: μ = 0, σ = 1, LSL = −4,
USL = 4, T = 120 subgroups of 5, change point t0 = 60, q0 = 0.997, γA = 0.95, γR = 0.80, t_min = 10, forgetting
factor 0.95, M = 500 posterior draws, prior (0, 10⁻³, 2, 1). The gradual-drift scenario is left out because its
path is not specified in the article.

| Scenario | Quantity | uccap | Article |
|---|---|---:|---:|
| Stable | P(any intervention), Tables 4 and 11 | 0.012 | 0.015 |
| Shift to 1.0 after t0 | P(any intervention), Table 4 / Table 11 | 0.831 | 0.841 / 0.874 |
| Shift to 1.0 after t0 | median delay from t0, Table 4 | 22 | 20 |
| Shift to 0.5 after t0 | P(any intervention), Table 11 | 0.016 | 0.017 |
| Transient 1.2, 20 subgroups | P(any intervention), Table 11 | 0.878 | 0.867 |
| Shift to 1.5 after t0 | detection / median delay, Table 11 | 1.00 / 6 | 1.00 / 6 |
| Transient 1.5, 10 subgroups | detection / median delay, Table 11 | 0.89 / 6 | 0.88 / 6 |
| σ from 1.0 to 1.8 after t0 | detection / median delay, Table 11 | 1.00 / 9 | 1.00 / 9 |

The probabilities agree within the Monte Carlo error of 1000 replications (about 0.016 at most), except the
probability of acting on the persistent shift to 1.0 in Table 11 (0.831 against 0.874), which is close to the
value of Table 4 of the same article (0.841). The median delays agree, except 22 against 20 subgroups for the
shift to 1.0. Table 11 measures the delay from the first unacceptable subgroup.

## Textbook example (NIST/SEMATECH §6.1.6)

USL = 20, LSL = 8, process mean 16, σ = 2: Cp = 1.000, Cpk = 0.667, Cpl = 1.333, Cpu = 0.667.
uccap returns the same four values (`tests/test_validation.py::test_nist_handbook_6_1_6`).

Note: the NIST Dataplot command `CNPK` uses the 0.5 % and 99.5 % percentiles. uccap, Minitab and ISO 22514
use 0.135 % and 99.865 %, so non-normal indices from Dataplot are not directly comparable.

## Check against Minitab

`validation/minitab/uccap_minitab_check.xlsx` contains the same eight data sets, their specification limits,
step-by-step Minitab instructions (Stat → Quality Tools → Capability Analysis → Normal / Nonnormal), and the
values uccap reports (N, mean, StDev overall and within, AD statistic and p, Cp, Cpk, Pp, Ppk, and for the
non-normal sets X0.135 / X50 / X99.865 and Ppk). It is provided so that users with Minitab can check
uccap against it on the same data; the validation above does not depend on it.

Expected small differences: Minitab's default within σ for individual data is the average moving range
(same as uccap's default); Minitab's three-parameter fits use its own threshold estimator, so only
two-parameter fits are compared.

## The same analysis in Minitab and JMP

For users who check uccap against a commercial package, this table shows where the corresponding
analysis is found. Index definitions follow the workflow article; where a package's default differs
(for example its within-σ estimator or its percentile convention), the results differ accordingly.

| uccap | Minitab | JMP |
|---|---|---|
| `capability_study`, normal data | Stat → Quality Tools → Capability Analysis → Normal | Analyze → Quality and Process → Process Capability |
| `capability_study`, non-normal data (best fit, percentile indices) | Stat → Quality Tools → Capability Analysis → Nonnormal | Process Capability, with a fitted distribution |
| Distribution ranking | Stat → Quality Tools → Individual Distribution Identification | Analyze → Distribution → Fit Distribution |
| Normality test | Stat → Basic Statistics → Normality Test | Analyze → Distribution → Fit Normal → Goodness of Fit |
| Stability (I-MR charts) | Stat → Control Charts → Variables Charts for Individuals → I-MR | Analyze → Quality and Process → Control Chart Builder |
| `risk_decision`, `sample_size`, `hybrid_risk` | — | — |

The last row implements the risk-calibrated approval and the UC-Cap score of the two later articles;
these are computed by uccap only and are compared with the articles' published results (above), not
with Minitab or JMP.

## Full comparison

127 of 127 comparisons within tolerance

| Data | Quantity | uccap | R | Difference | Tolerance | Pass |
|---|---|---|---|---|---|---|
| constants | d2(2) | 1.1284 | 1.1284 | 1.67e-07 (abs) | 0.0005 | yes |
| constants | d2(3) | 1.6926 | 1.6926 | 2.49e-07 (abs) | 0.0005 | yes |
| constants | d2(4) | 2.0588 | 2.0588 | 2.54e-07 (abs) | 0.0005 | yes |
| constants | d2(5) | 2.3259 | 2.3259 | 5.27e-08 (abs) | 0.0005 | yes |
| constants | d2(6) | 2.5344 | 2.5344 | 2.79e-07 (abs) | 0.0005 | yes |
| constants | d2(7) | 2.7044 | 2.7044 | 2.49e-07 (abs) | 0.0005 | yes |
| constants | d2(8) | 2.8472 | 2.8472 | 3.88e-07 (abs) | 0.0005 | yes |
| constants | d2(9) | 2.9700 | 2.9700 | 3.24e-07 (abs) | 0.0005 | yes |
| constants | d2(10) | 3.0775 | 3.0775 | 4.62e-07 (abs) | 0.0005 | yes |
| constants | c4(2) | 0.7979 | 0.7979 | 1.11e-16 (abs) | 1e-09 | yes |
| constants | c4(3) | 0.8862 | 0.8862 | 2.22e-16 (abs) | 1e-09 | yes |
| constants | c4(4) | 0.9213 | 0.9213 | 4.44e-16 (abs) | 1e-09 | yes |
| constants | c4(5) | 0.9400 | 0.9400 | 6.66e-16 (abs) | 1e-09 | yes |
| constants | c4(6) | 0.9515 | 0.9515 | 2.22e-16 (abs) | 1e-09 | yes |
| constants | c4(7) | 0.9594 | 0.9594 | 1.11e-16 (abs) | 1e-09 | yes |
| constants | c4(8) | 0.9650 | 0.9650 | 6.66e-16 (abs) | 1e-09 | yes |
| constants | c4(9) | 0.9693 | 0.9693 | 2.22e-16 (abs) | 1e-09 | yes |
| constants | c4(10) | 0.9727 | 0.9727 | 1.11e-15 (abs) | 1e-09 | yes |
| constants | c4(11) | 0.9754 | 0.9754 | 1.22e-15 (abs) | 1e-09 | yes |
| constants | c4(12) | 0.9776 | 0.9776 | 5.55e-16 (abs) | 1e-09 | yes |
| constants | c4(13) | 0.9794 | 0.9794 | 3.33e-15 (abs) | 1e-09 | yes |
| constants | c4(14) | 0.9810 | 0.9810 | 3.77e-15 (abs) | 1e-09 | yes |
| constants | c4(15) | 0.9823 | 0.9823 | 1.11e-16 (abs) | 1e-09 | yes |
| constants | c4(16) | 0.9835 | 0.9835 | 1.11e-15 (abs) | 1e-09 | yes |
| constants | c4(17) | 0.9845 | 0.9845 | 1.22e-15 (abs) | 1e-09 | yes |
| constants | c4(18) | 0.9854 | 0.9854 | 4.00e-15 (abs) | 1e-09 | yes |
| constants | c4(19) | 0.9862 | 0.9862 | 1.67e-15 (abs) | 1e-09 | yes |
| constants | c4(20) | 0.9869 | 0.9869 | 2.00e-15 (abs) | 1e-09 | yes |
| constants | c4(21) | 0.9876 | 0.9876 | 1.55e-15 (abs) | 1e-09 | yes |
| constants | c4(22) | 0.9882 | 0.9882 | 5.00e-15 (abs) | 1e-09 | yes |
| constants | c4(23) | 0.9887 | 0.9887 | 5.44e-15 (abs) | 1e-09 | yes |
| constants | c4(24) | 0.9892 | 0.9892 | 3.77e-15 (abs) | 1e-09 | yes |
| constants | c4(25) | 0.9896 | 0.9896 | 3.11e-15 (abs) | 1e-09 | yes |
| normal | mean | 9.8047 | 9.8047 | 1.81e-16 (rel) | 1e-12 | yes |
| normal | sigma overall | 0.4427 | 0.4427 | 3.76e-16 (rel) | 1e-12 | yes |
| normal | sigma within (AMR) | 0.4657 | 0.4657 | 1.48e-07 (rel) | 0.0005 | yes |
| normal | empirical P0.135/P50/P99.865 | 8.7952 / 9.8265 / 10.6378 | 8.7952 / 9.8265 / 10.6378 | 0.00e+00 (rel) | 1e-12 | yes |
| normal | Anderson-Darling A² | 0.1738 | 0.1738 | 4.16e-16 (abs) | 1e-09 | yes |
| normal | Anderson-Darling p | 0.9221 | 0.9221 | 2.22e-16 (abs) | 1e-06 | yes |
| normal | Shapiro-Wilk p | 0.7429 | 0.7429 | 2.56e-14 (abs) | 0.0001 | yes |
| normal | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.9080 | 0.9084 | 3.61e-04 (abs) | 0.05 | yes |
| normal | normal: log-likelihood | -29.6987 | -29.6987 | -1.42e-14 (R − uccap) | 0.001 | yes |
| normal | normal: fitted P0.135/P50/P99.865 | 8.4900 / 9.8047 / 11.1195 | 8.4900 / 9.8047 / 11.1195 | 4.63e-15 (rel) | 0.002 | yes |
| lognormal | mean | 1.1678 | 1.1678 | 0.00e+00 (rel) | 1e-12 | yes |
| lognormal | sigma overall | 0.6427 | 0.6427 | 5.18e-16 (rel) | 1e-12 | yes |
| lognormal | sigma within (AMR) | 0.5601 | 0.5601 | 1.48e-07 (rel) | 0.0005 | yes |
| lognormal | empirical P0.135/P50/P99.865 | 0.3760 / 0.9918 / 4.0236 | 0.3760 / 0.9918 / 4.0236 | 2.21e-16 (rel) | 1e-12 | yes |
| lognormal | Anderson-Darling A² | 1.9859 | 1.9859 | 2.66e-15 (abs) | 1e-09 | yes |
| lognormal | Anderson-Darling p | 3.915e-05 | 3.915e-05 | 1.36e-20 (abs) | 1e-06 | yes |
| lognormal | Shapiro-Wilk p | 1.746e-06 | 1.746e-06 | 4.02e-21 (abs) | 0.0001 | yes |
| lognormal | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.0030 | 0.0031 | 6.13e-05 (abs) | 0.05 | yes |
| lognormal | lognormal: log-likelihood | -35.4383 | -35.4383 | -1.42e-14 (R − uccap) | 0.001 | yes |
| lognormal | lognormal: fitted P0.135/P50/P99.865 | 0.2510 / 1.0385 / 4.2964 | 0.2510 / 1.0385 / 4.2964 | 4.06e-15 (rel) | 0.002 | yes |
| lognormal | normal: log-likelihood | -48.3349 | -48.3349 | 0.00e+00 (R − uccap) | 0.001 | yes |
| lognormal | normal: fitted P0.135/P50/P99.865 | -0.7408 / 1.1678 / 3.0763 | -0.7408 / 1.1678 / 3.0763 | 7.22e-16 (rel) | 0.002 | yes |
| weibull | mean | 0.8512 | 0.8512 | 0.00e+00 (rel) | 1e-12 | yes |
| weibull | sigma overall | 0.4218 | 0.4218 | 0.00e+00 (rel) | 1e-12 | yes |
| weibull | sigma within (AMR) | 0.4100 | 0.4100 | 1.48e-07 (rel) | 0.0005 | yes |
| weibull | empirical P0.135/P50/P99.865 | 0.1852 / 0.7751 / 2.1182 | 0.1852 / 0.7751 / 2.1182 | 2.10e-16 (rel) | 1e-12 | yes |
| weibull | Anderson-Darling A² | 0.4947 | 0.4947 | 6.72e-15 (abs) | 1e-09 | yes |
| weibull | Anderson-Darling p | 0.2057 | 0.2057 | 7.91e-15 (abs) | 1e-06 | yes |
| weibull | Shapiro-Wilk p | 0.0741 | 0.0741 | 6.83e-15 (abs) | 0.0001 | yes |
| weibull | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.3800 | 0.3811 | 1.11e-03 (abs) | 0.05 | yes |
| weibull | weibull: log-likelihood | -24.2964 | -24.2964 | -6.31e-08 (R − uccap) | 0.001 | yes |
| weibull | weibull: fitted P0.135/P50/P99.865 | 0.0459 / 0.8140 / 2.3010 | 0.0459 / 0.8140 / 2.3009 | 1.99e-04 (rel) | 0.002 | yes |
| weibull | normal: log-likelihood | -27.2846 | -27.2846 | -3.20e-14 (R − uccap) | 0.001 | yes |
| weibull | normal: fitted P0.135/P50/P99.865 | -0.4016 / 0.8512 / 2.1040 | -0.4016 / 0.8512 / 2.1040 | 1.69e-15 (rel) | 0.002 | yes |
| gamma | mean | 1.3783 | 1.3783 | 1.61e-16 (rel) | 1e-12 | yes |
| gamma | sigma overall | 0.9648 | 0.9648 | 1.15e-16 (rel) | 1e-12 | yes |
| gamma | sigma within (AMR) | 0.9180 | 0.9180 | 1.48e-07 (rel) | 0.0005 | yes |
| gamma | empirical P0.135/P50/P99.865 | 0.1834 / 1.2859 / 4.2923 | 0.1834 / 1.2859 / 4.2923 | 0.00e+00 (rel) | 1e-12 | yes |
| gamma | Anderson-Darling A² | 1.7279 | 1.7279 | 4.22e-15 (abs) | 1e-09 | yes |
| gamma | Anderson-Darling p | 1.716e-04 | 1.716e-04 | 6.94e-18 (abs) | 1e-06 | yes |
| gamma | Shapiro-Wilk p | 3.427e-05 | 3.427e-05 | 3.79e-19 (abs) | 0.0001 | yes |
| gamma | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.0290 | 0.0238 | 5.18e-03 (abs) | 0.05 | yes |
| gamma | gamma: log-likelihood | -58.8576 | -58.8576 | -2.46e-07 (R − uccap) | 0.001 | yes |
| gamma | gamma: fitted P0.135/P50/P99.865 | 0.0472 / 1.1755 / 5.8431 | 0.0472 / 1.1755 / 5.8436 | 8.77e-05 (rel) | 0.002 | yes |
| gamma | normal: log-likelihood | -68.6496 | -68.6496 | -5.68e-14 (R − uccap) | 0.001 | yes |
| gamma | normal: fitted P0.135/P50/P99.865 | -1.4869 / 1.3783 / 4.2436 | -1.4869 / 1.3783 / 4.2436 | 2.69e-15 (rel) | 0.002 | yes |
| exponential | mean | 0.4641 | 0.4641 | 0.00e+00 (rel) | 1e-12 | yes |
| exponential | sigma overall | 0.4012 | 0.4012 | 1.25e-15 (rel) | 1e-12 | yes |
| exponential | sigma within (AMR) | 0.3577 | 0.3577 | 1.48e-07 (rel) | 0.0005 | yes |
| exponential | empirical P0.135/P50/P99.865 | 0.0278 / 0.3574 / 1.5837 | 0.0278 / 0.3574 / 1.5837 | 0.00e+00 (rel) | 1e-12 | yes |
| exponential | Anderson-Darling A² | 2.4286 | 2.4286 | 7.11e-15 (abs) | 1e-09 | yes |
| exponential | Anderson-Darling p | 3.118e-06 | 3.118e-06 | 1.25e-19 (abs) | 1e-06 | yes |
| exponential | Shapiro-Wilk p | 2.839e-05 | 2.839e-05 | 8.84e-19 (abs) | 0.0001 | yes |
| exponential | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.0010 | 8.936e-05 | 9.11e-04 (abs) | 0.05 | yes |
| exponential | exponential: log-likelihood | -11.6197 | -11.6197 | -2.84e-14 (R − uccap) | 0.001 | yes |
| exponential | exponential: fitted P0.135/P50/P99.865 | 6.270e-04 / 0.3217 / 3.0668 | 6.270e-04 / 0.3217 / 3.0668 | 6.19e-09 (rel) | 0.002 | yes |
| exponential | normal: log-likelihood | -24.7764 | -24.7764 | -2.13e-14 (R − uccap) | 0.001 | yes |
| exponential | normal: fitted P0.135/P50/P99.865 | -0.7274 / 0.4641 / 1.6556 | -0.7274 / 0.4641 / 1.6556 | 1.48e-15 (rel) | 0.002 | yes |
| lev | mean | 0.3142 | 0.3142 | 1.77e-16 (rel) | 1e-12 | yes |
| lev | sigma overall | 1.1110 | 1.1110 | 3.60e-15 (rel) | 1e-12 | yes |
| lev | sigma within (AMR) | 1.2487 | 1.2487 | 1.48e-07 (rel) | 0.0005 | yes |
| lev | empirical P0.135/P50/P99.865 | -1.6070 / 0.1721 / 2.2800 | -1.6070 / 0.1721 / 2.2800 | 1.95e-16 (rel) | 1e-12 | yes |
| lev | Anderson-Darling A² | 0.8285 | 0.8285 | 2.22e-16 (abs) | 1e-09 | yes |
| lev | Anderson-Darling p | 0.0302 | 0.0302 | 2.08e-17 (abs) | 1e-06 | yes |
| lev | Shapiro-Wilk p | 0.0370 | 0.0370 | 2.05e-15 (abs) | 0.0001 | yes |
| lev | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.0560 | 0.0527 | 3.26e-03 (abs) | 0.05 | yes |
| lev | lev: log-likelihood | -76.0593 | -76.0593 | -3.08e-07 (R − uccap) | 0.001 | yes |
| lev | lev: fitted P0.135/P50/P99.865 | -2.0415 / 0.1285 / 6.1345 | -2.0416 / 0.1284 / 6.1342 | 9.35e-04 (rel) | 0.002 | yes |
| lev | normal: log-likelihood | -75.7038 | -75.7038 | 0.00e+00 (R − uccap) | 0.001 | yes |
| lev | normal: fitted P0.135/P50/P99.865 | -2.9852 / 0.3142 / 3.6136 | -2.9852 / 0.3142 / 3.6136 | 2.46e-16 (rel) | 0.002 | yes |
| sev | mean | -0.4387 | -0.4387 | 0.00e+00 (rel) | 1e-12 | yes |
| sev | sigma overall | 1.2948 | 1.2948 | 3.77e-15 (rel) | 1e-12 | yes |
| sev | sigma within (AMR) | 1.1180 | 1.1180 | 1.48e-07 (rel) | 0.0005 | yes |
| sev | empirical P0.135/P50/P99.865 | -3.2445 / -0.2580 / 1.8527 | -3.2445 / -0.2580 / 1.8527 | 0.00e+00 (rel) | 1e-12 | yes |
| sev | Anderson-Darling A² | 0.5072 | 0.5072 | 7.55e-15 (abs) | 1e-09 | yes |
| sev | Anderson-Darling p | 0.1913 | 0.1913 | 7.61e-15 (abs) | 1e-06 | yes |
| sev | Shapiro-Wilk p | 0.1953 | 0.1953 | 2.75e-15 (abs) | 0.0001 | yes |
| sev | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.0750 | 0.0694 | 5.56e-03 (abs) | 0.05 | yes |
| sev | sev: log-likelihood | -83.0368 | -83.0368 | -5.49e-07 (R − uccap) | 0.001 | yes |
| sev | sev: fitted P0.135/P50/P99.865 | -7.1770 / -0.2305 / 2.2793 | -7.1779 / -0.2306 / 2.2795 | 5.25e-04 (rel) | 0.002 | yes |
| sev | normal: log-likelihood | -83.3603 | -83.3603 | -7.11e-14 (R − uccap) | 0.001 | yes |
| sev | normal: fitted P0.135/P50/P99.865 | -4.2841 / -0.4387 / 3.4067 | -4.2841 / -0.4387 / 3.4067 | 7.82e-16 (rel) | 0.002 | yes |
| logistic | mean | 5.0199 | 5.0199 | 0.00e+00 (rel) | 1e-12 | yes |
| logistic | sigma overall | 0.4541 | 0.4541 | 3.67e-16 (rel) | 1e-12 | yes |
| logistic | sigma within (AMR) | 0.5078 | 0.5078 | 1.48e-07 (rel) | 0.0005 | yes |
| logistic | empirical P0.135/P50/P99.865 | 3.8870 / 5.0628 / 6.0029 | 3.8870 / 5.0628 / 6.0029 | 1.14e-16 (rel) | 1e-12 | yes |
| logistic | Anderson-Darling A² | 0.1989 | 0.1989 | 7.30e-15 (abs) | 1e-09 | yes |
| logistic | Anderson-Darling p | 0.8797 | 0.8797 | 1.65e-14 (abs) | 1e-06 | yes |
| logistic | Shapiro-Wilk p | 0.9622 | 0.9622 | 1.41e-14 (abs) | 0.0001 | yes |
| logistic | Lilliefors p (Monte Carlo vs Dallal-Wilkinson) | 0.4330 | 0.4398 | 6.82e-03 (abs) | 0.05 | yes |
| logistic | logistic: log-likelihood | -31.6815 | -31.6815 | -6.17e-07 (R − uccap) | 0.001 | yes |
| logistic | logistic: fitted P0.135/P50/P99.865 | 3.3215 / 5.0317 / 6.7418 | 3.3215 / 5.0317 / 6.7420 | 2.56e-05 (rel) | 0.002 | yes |
| logistic | normal: log-likelihood | -30.9751 | -30.9751 | -3.55e-15 (R − uccap) | 0.001 | yes |
| logistic | normal: fitted P0.135/P50/P99.865 | 3.6712 / 5.0199 / 6.3687 | 3.6712 / 5.0199 / 6.3687 | 6.05e-16 (rel) | 0.002 | yes |
