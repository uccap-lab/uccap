import pathlib

import pytest

import uccap
from uccap.cli import main

TEMPLATE = pathlib.Path(__file__).resolve().parent.parent / "templates" / "multiple_dimensions.csv"


def test_cli_writes_html_and_csv(tmp_path, capsys):
    assert main([str(TEMPLATE), "--out", str(tmp_path) + "/", "--html", "--csv", "--cost-ratio", "19"]) == 0
    assert (tmp_path / "multiple_dimensions_uccap.html").exists()
    assert (tmp_path / "multiple_dimensions_uccap.csv").read_text().startswith("dimension")
    assert "D101" in capsys.readouterr().out


def test_cli_cite_and_errors(capsys):
    assert main(["--cite"]) == 0
    assert "jiang2026risk" in capsys.readouterr().out
    assert main(["no_such_file.csv"]) == 2
    assert "jiang2026hybrid" in uccap.cite("hybrid") and "jiang2026risk" not in uccap.cite("hybrid")
    with pytest.raises(ValueError):
        uccap.cite("other")


def test_cli_plots(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    assert main([str(TEMPLATE), "--out", str(tmp_path) + "/", "--plots", "overview,uccap", "--quiet"]) == 0
    folder = tmp_path / "multiple_dimensions_uccap_plots"
    assert (folder / "all_dimensions.png").exists()
    assert (folder / "D101_overview.png").exists()
    assert (folder / "D101_uccap.png").exists() == uccap.hybrid.model_available()
    assert "figures in" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([str(TEMPLATE), "--out", str(tmp_path) + "/", "--plots", "pie"])


def test_cli_plan(tmp_path, capsys):
    assert main(["plan", "1.5", "2.0", "--cost-ratio", "19", "--parts", "32"]) == 0
    out = capsys.readouterr().out
    assert "alpha = 0.0500" in out and "283" in out and "1.6909" in out
    pytest.importorskip("matplotlib")
    assert main(["plan", "1.67", "--plot", str(tmp_path / "plan.png")]) == 0
    assert (tmp_path / "plan.png").exists()
