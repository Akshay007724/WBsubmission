"""Task A: secret scan, replay provider."""

import json

from acquirer_engine import secretscan
from acquirer_engine.cli import main

# Built at runtime so this test file itself never contains a key-shaped literal.
FAKE_ANTHROPIC = "sk-" + "ant-" + "a1B2" * 8
FAKE_GOOGLE_AQ = "AQ" + "." + "Ab8RN6JP" * 4


def test_scan_flags_each_key_shape(tmp_path):
    cases = {
        "a.txt": FAKE_ANTHROPIC,
        "b.txt": "AIza" + "x" * 35,
        "c.txt": FAKE_GOOGLE_AQ,
        "d.env": "GEMINI_API_KEY=" + "abcdEFGH1234ijkl",
        "e.py": "key = '" + "sk-" + "proj" + "Z" * 30 + "'",
    }
    for name, text in cases.items():
        (tmp_path / name).write_text(text)
    hits = {rel for rel, _ in secretscan.scan_tree(tmp_path)}
    assert hits == set(cases)


def test_scan_ignores_placeholders_and_empty_assignments(tmp_path):
    (tmp_path / "README.md").write_text("export ANTHROPIC_API_KEY=sk-ant-...\nrisk-adjusted multiples\n")
    (tmp_path / ".env.example").write_text("ANTHROPIC_API_KEY=\nOPENAI_API_KEY=\n")
    assert secretscan.scan_tree(tmp_path) == []
    assert secretscan.main([str(tmp_path)]) == 0


def test_scan_cli_fails_loudly_without_echoing_secret(tmp_path, capsys):
    (tmp_path / "leak.txt").write_text(FAKE_ANTHROPIC)
    assert secretscan.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "leak.txt" in err and FAKE_ANTHROPIC not in err


def _args(tmp_path, *extra):
    return ["--out-dir", str(tmp_path / "out"), "--feedback-file", str(tmp_path / "fb.json"), *extra]


def test_replay_needs_no_key_and_fails_clearly_on_miss(tmp_path, monkeypatch, capsys):
    from acquirer_engine import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "empty-cache")
    assert main(_args(tmp_path, "--provider", "replay")) == 1
    assert "replay cache miss for 10 acquirer(s)" in capsys.readouterr().err


def test_replay_reproduces_cached_run_byte_for_byte(tmp_path, monkeypatch):
    """Populate a cache with the offline writer under the replay key, then replay twice."""
    from acquirer_engine import config, llm

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    real_make = llm.make_client

    class SeedClient(llm.OfflineClient):  # writes entries replay can read
        provider = "anthropic"
        model = config.DEFAULT_MODELS["replay"]

    monkeypatch.setattr(
        "acquirer_engine.cli.make_client", lambda p, m: SeedClient() if p == "seed" else real_make(p, m)
    )
    monkeypatch.setattr("acquirer_engine.engine.OfflineClient", llm.OfflineClient)
    import acquirer_engine.cli as cli_mod

    parser = cli_mod._parser

    def patched():
        p = parser()
        for a in p._actions:
            if a.dest == "provider":
                a.choices = [*a.choices, "seed"]
        return p

    monkeypatch.setattr(cli_mod, "_parser", patched)

    assert main(_args(tmp_path, "--provider", "seed")) == 0
    assert main(_args(tmp_path, "--provider", "replay", "--out-dir", str(tmp_path / "r1"))) == 0
    assert main(_args(tmp_path, "--provider", "replay", "--out-dir", str(tmp_path / "r2"))) == 0
    a = next((tmp_path / "r1").rglob("report.json")).read_bytes()
    b = next((tmp_path / "r2").rglob("report.json")).read_bytes()
    assert a == b
    assert json.loads(a)["meta"]["cache_hits"] == 10


def test_replay_rejects_refresh(tmp_path):
    assert main(_args(tmp_path, "--provider", "replay", "--refresh")) == 4


def test_replay_shows_the_live_run_stamp(tmp_path, monkeypatch):
    from acquirer_engine import config
    from acquirer_engine.engine import run_identity
    from acquirer_engine.llm import Cache

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    test_replay_reproduces_cached_run_byte_for_byte(tmp_path, monkeypatch)  # seeds the cache, replays twice
    rep = json.loads(next((tmp_path / "r1").rglob("report.json")).read_text())
    stamp = {"generated_at": "2026-01-01T00:00:00+00:00", "elapsed_s": 25.0, "model": "m", "statuses": {"verified": 10}}
    Cache(config.CACHE_DIR).put(f"run-{run_identity(rep['pages'])}", stamp)
    assert main(_args(tmp_path, "--provider", "replay", "--out-dir", str(tmp_path / "r3"))) == 0
    rep3 = json.loads(next((tmp_path / "r3").rglob("report.json")).read_text())
    assert rep3["meta"]["source_run"] == stamp
    assert "Live run behind this report: 2026-01-01" in next((tmp_path / "r3").rglob("report.html")).read_text()
