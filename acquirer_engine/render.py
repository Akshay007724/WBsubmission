"""Write HTML / Markdown / JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .errors import RenderError

_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html", "j2"], default_for_string=True),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
# Markdown must not be HTML-escaped.
_MD_ENV = Environment(loader=_ENV.loader, autoescape=False, trim_blocks=True, lstrip_blocks=True)


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise RenderError(f"cannot write {path}: {e.strerror or e}") from e


def write_report(report: dict, out_dir: Path) -> dict[str, Path]:
    """Render report.html / report.md / report.json into out_dir. Raises RenderError if not writable."""
    paths = {k: out_dir / f"report.{k}" for k in ("html", "md", "json")}
    if "evaluation" in report:
        paths["eval"] = out_dir / "eval.json"
        _write(paths["eval"], json.dumps(report["evaluation"], indent=1))
    _write(paths["html"], _ENV.get_template("report.html.j2").render(r=report))
    _write(paths["md"], _MD_ENV.get_template("report.md.j2").render(r=report))
    slim = {**report, "data_profile": {k: v for k, v in report["data_profile"].items() if k != "charts"}}
    _write(paths["json"], json.dumps(slim, indent=1, default=str))  # charts live in the HTML / eda output
    return paths


def write_eda(profile_dict: dict, target: dict, source: str, sha: str, out_dir: Path) -> dict[str, Path]:
    """Write eda.md, eda.json and charts/*.svg. Raises RenderError if the directory is not writable."""
    paths = {"md": out_dir / "eda.md", "json": out_dir / "eda.json"}
    for name, svg in profile_dict.get("charts", {}).items():
        _write(out_dir / "charts" / f"{name}.svg", svg)
    _write(paths["md"], _MD_ENV.get_template("eda.md.j2").render(e=profile_dict, target=target, source=source, sha=sha))
    _write(paths["json"], json.dumps({k: v for k, v in profile_dict.items() if k != "charts"}, indent=1, default=str))
    return paths


def write_compare(a: dict, b: dict, link_a: str, link_b: str, out: Path) -> Path:
    """Side-by-side HTML for two reports (overlap highlighted). Raises RenderError if not writable."""
    ra = {p["candidate"]["acquirer"]: p for p in a["pages"]}
    rb = {p["candidate"]["acquirer"]: p for p in b["pages"]}
    overlap = [n for n in ra if n in rb]
    lvl = lambda p: p["rationale"]["conviction"]["level"] if p["rationale"] else "—"  # noqa: E731
    rows = [{"name": n, "ra": ra[n]["rank"], "rb": rb[n]["rank"], "ca": lvl(ra[n]), "cb": lvl(rb[n])} for n in overlap]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _ENV.get_template("compare.html.j2").render(
            a={"r": a, "link": link_a},
            b={"r": b, "link": link_b},
            overlap=overlap,
            overlap_rows=rows,
        ),
        encoding="utf-8",
    )
    return out
