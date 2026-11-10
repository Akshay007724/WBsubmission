"""Task L: requirements.txt mirrors pyproject; the console script is declared."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+; 3.10 skips these two checks

ROOT = Path(__file__).resolve().parent.parent


def test_requirements_txt_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    reqs = [
        ln.strip()
        for ln in (ROOT / "requirements.txt").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert sorted(reqs) == sorted(project["dependencies"])
    assert project["scripts"]["acquirer-engine"] == "acquirer_engine.cli:main"


def test_templates_are_packaged():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    assert "templates/*.j2" in cfg["acquirer_engine"]
    assert len(list((ROOT / "acquirer_engine" / "templates").glob("*.j2"))) == 4
