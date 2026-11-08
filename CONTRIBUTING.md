# Contributing

Thank you for your interest in uccap. Bug reports, questions about the methods, and pull requests are
all welcome.

## Reporting a problem

Open an issue on GitHub (the templates ask for the details below):

- the uccap, Python, NumPy, SciPy and pandas versions (`python -c "import uccap; print(uccap.__version__)"`);
- a minimal example that reproduces the problem. If your measurements are confidential, simulated data
  with the same shape is enough; please do not post production data;
- what you expected and what happened, including the full error message.

Questions about the methods are welcome as issues too. Please name the article and section you are
asking about (see the README).

Security problems: see [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/uccap-lab/uccap.git
cd uccap
pip install -e ".[dev]"          # the package, pytest, openpyxl, matplotlib, ruff, nbclient
pytest                           # the full test suite, about 20 s
ruff check src tests validation examples
```

Continuous integration runs the tests on Linux (Python 3.9, 3.11, 3.13), Windows (3.12) and with the
lowest dependency versions declared in `pyproject.toml`, plus the lint check.

## Proposing a change

1. Fork the repository and create a branch.
2. Add or update tests in `tests/` and run `pytest` and `ruff`.
3. If the change affects numerical results, rerun `PYTHONPATH=src python validation/compare.py` and
   explain any difference in the pull request.
4. Update the documentation where relevant (see below) and add a line to `CHANGELOG.md`.
5. Open a pull request describing what changed and why.

Methods in uccap follow the published articles. A change that departs from an article (a different
default, estimator, test or decision rule) must say so in the pull request and in the documentation.
The thresholds and decision rules of the examples and defaults illustrate the methods, as in the
articles; please keep them presented that way.

### Files that are generated

Some files are built from others. Edit the source and regenerate:

| Generated file | Source | Command |
|---|---|---|
| `docs/manual_tex/uccap_manual.pdf` | `docs/manual_tex/uccap_manual.tex` | `xelatex uccap_manual.tex` twice, in `docs/manual_tex/` |
| `docs/MANUAL.md`, `docs/figures/manual/` | `docs/manual_tex/uccap_manual.tex` | `python docs/manual_tex/to_markdown.py` (after the PDF) |
| `examples/*_output.txt` | `examples/*.py` | `PYTHONPATH=src python examples/quickstart.py > examples/quickstart_output.txt`, likewise for `batch_example.py` |

The UC-Cap model parameters are not included in this release and are not
retrained here. Tests that need them are skipped unless `UCCAP_MODEL` points to the
file. The industrial training data are not public.

## Code of conduct

Be respectful and constructive. Harassment or personal attacks are not tolerated in issues, pull
requests or any other project space. The maintainers may edit or remove contributions that break
this rule.
