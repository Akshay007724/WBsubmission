"""Task L: CSV text is data (escaped in HTML), and structured logs carry timing but never prompts or keys."""

import json

import pandas as pd

from acquirer_engine import config as C
from acquirer_engine.cli import main

EVIL = '<script>alert("x")</script> Capital'


def test_csv_text_is_escaped_in_html(tmp_path):
    df = pd.read_csv(C.DEFAULT_DATA)
    df.loc[df["acquirer"] == "Nordic Capital", "acquirer"] = EVIL
    df.to_csv(tmp_path / "evil.csv", index=False)
    assert (
        main(
            [
                "--provider",
                "offline",
                "--no-cache",
                "--data",
                str(tmp_path / "evil.csv"),
                "--out-dir",
                str(tmp_path / "o"),
                "--feedback-file",
                str(tmp_path / "f.json"),
            ]
        )
        == 0
    )
    html = next(tmp_path.rglob("report.html")).read_text()
    assert "<script>alert" not in html and "&lt;script&gt;alert" in html


def test_json_logs_have_timing_and_no_secrets(tmp_path, monkeypatch, capsys):
    secret = "sk-ant-" + "x" * 30
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    assert (
        main(
            [
                "--provider",
                "offline",
                "--no-cache",
                "--log-json",
                "--log-level",
                "INFO",
                "--out-dir",
                str(tmp_path),
                "--feedback-file",
                str(tmp_path / "f.json"),
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    records = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    pages = [r for r in records if r["msg"] == "page done"]
    assert len(pages) == 10 and all("elapsed_s" in r and "acquirer" in r for r in pages)
    assert secret not in err and "<evidence>" not in err  # no keys, no prompt text
