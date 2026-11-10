"""Task H: deterministic report scoring."""

import json

from acquirer_engine import eval as ev
from acquirer_engine.cli import main


def test_offline_report_scores_and_eval_json_written(tmp_path):
    assert (
        main(
            [
                "--provider",
                "offline",
                "--no-cache",
                "--out-dir",
                str(tmp_path),
                "--feedback-file",
                str(tmp_path / "f.json"),
            ]
        )
        == 0
    )
    s = json.loads(next(tmp_path.rglob("eval.json")).read_text())
    assert s["citation_validity"] == 1.0 and s["number_validity"] == 1.0
    assert s["pages_with_narrative"] == "10/10" and s["conviction_levels_used"] >= 2
    assert "Evaluation (deterministic" in next(tmp_path.rglob("report.html")).read_text()


def test_eval_detects_fabrications_and_inversions():
    page = {
        "rank": 1,
        "candidate": {"acquirer": "A"},
        "status": "unverified",
        "evidence": {"precedents": [{"transaction_id": "MA-2020-0001", "ev_ebitda": 12.0}]},
        "rationale": {
            "headline": "x",
            "acquirer_overview": "x",
            "valuation_context": "paid 99.9x [MA-1999-0001]",
            "strategic_fit_thesis": "a b c d",
            "conviction": {"level": "Low", "rationale": "x"},
            "risk_flags": [{"detail": "r"}],
            "precedent_activity": [],
        },
    }
    second = {
        **page,
        "rank": 2,
        "candidate": {"acquirer": "B"},
        "rationale": {**page["rationale"], "conviction": {"level": "High", "rationale": "x"}},
    }
    rep = {"pages": [page, second], "market": {}, "target": {}}
    s = ev.score(rep)
    assert s["citation_validity"] == 0.0 and s["number_validity"] < 1 and s["rank_conviction_inversions"] == 1


def test_before_after_table_and_cli(tmp_path, capsys):
    rep = tmp_path / "r.json"
    main(
        ["--provider", "offline", "--no-cache", "--out-dir", str(tmp_path), "--feedback-file", str(tmp_path / "f.json")]
    )
    src = next(tmp_path.rglob("report.json"))
    rep.write_text(src.read_text())
    assert ev.main([str(rep)]) == 0 and (tmp_path / "eval.json").is_file()
    assert ev.main([str(rep), str(rep)]) == 0
    assert "| Metric | Before | After |" in capsys.readouterr().out
    assert ev.main([]) == 2
