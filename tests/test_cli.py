import json
from pathlib import Path

from acquirer_engine.cli import main

ROOT = Path(__file__).resolve().parent.parent


def _args(tmp_path, *extra):
    return [
        "--provider",
        "offline",
        "--no-cache",
        "--out-dir",
        str(tmp_path / "out"),
        "--feedback-file",
        str(tmp_path / "fb.json"),
        *extra,
    ]


def test_end_to_end_offline(tmp_path):
    assert main(_args(tmp_path)) == 0
    out = tmp_path / "out" / "healthcare-services-200m-regional-private"
    report = json.loads((out / "report.json").read_text())
    assert len(report["pages"]) == 10
    assert all(len(p["rationale"]["risk_flags"]) >= 2 for p in report["pages"])
    assert len({p["rationale"]["conviction"]["level"] for p in report["pages"]}) >= 2
    assert (out / "report.html").read_text().count(
        'class="page"'
    ) == 14  # 10 acquirers + also-considered + data profile + evaluation + methodology
    displaced = [a for a in report["also_considered"] if "displaced" in a["note"]]
    assert displaced, "buyers bumped by the mix floor must stay visible to the MD"


def test_exclude_persists_and_reranks(tmp_path):
    main(_args(tmp_path))
    out = tmp_path / "out" / "healthcare-services-200m-regional-private" / "report.json"
    top = json.loads(out.read_text())["pages"][0]["candidate"]["acquirer"]
    assert main(_args(tmp_path, "--exclude", top.split()[0])) == 0
    assert top not in [p["candidate"]["acquirer"] for p in json.loads(out.read_text())["pages"]]
    assert main(_args(tmp_path)) == 0  # still excluded on next run without the flag
    assert top not in [p["candidate"]["acquirer"] for p in json.loads(out.read_text())["pages"]]


def test_unknown_exclude_and_bad_sector_exit_2(tmp_path):
    assert main(_args(tmp_path, "--exclude", "Zzzz Nonexistent")) == 4  # ConfigError
    assert main(_args(tmp_path, "--sector", "Space Mining")) == 4


def test_compare_mode(tmp_path):
    rc = main(
        _args(tmp_path, "--compare", str(ROOT / "targets/default.yaml"), str(ROOT / "targets/behavioral_500m.yaml"))
    )
    assert rc == 0 and (tmp_path / "out" / "compare.html").is_file()


def test_missing_api_key_is_clear(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--provider", "anthropic", "--out-dir", str(tmp_path)]) == 5  # LLMError
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_target_with_no_comps_renders_but_is_not_a_success(tmp_path, capsys):
    # $25B Dental deal: nothing comparable. The report still renders (with every widening step explained),
    # but an empty buyer list must not exit 0.
    assert main(_args(tmp_path, "--sector", "Dental", "--ev", "25000", "--geography", "West Coast")) == 1
    out = capsys.readouterr()
    assert "0 eligible acquirers" in out.err and "Valuation anchor widened" in out.out
    assert next(tmp_path.rglob("report.html")).is_file()
