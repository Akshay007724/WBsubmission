"""Task L: every failure path ends in a clear message or a degraded page, never a lost run or a traceback."""

import json
import os
import stat
from types import SimpleNamespace

import pandas as pd
import pytest

from acquirer_engine import config as C
from acquirer_engine import evidence, offline, scoring, settings
from acquirer_engine.cli import main
from acquirer_engine.errors import ConfigError, LLMError
from acquirer_engine.llm import AnthropicClient, Cache, GeminiClient, OpenAIClient, _Usage, generate, make_client


def _setup(ds, profile):
    s, picks, _ = scoring.rank(ds, profile)
    m = evidence.market_comps(s, profile, ds.ref_year)
    packs = [evidence.build(s, c, profile, m, ds.ref_year) for c in picks]
    evidence.add_closest_competitors(packs)
    return packs[0], m, {**profile.model_dump(), "ebitda_margin_pct": m["target_margin_assumed_pct"]}


# ---------------------------------------------------------------- Anthropic client paths


def _anthropic(parse):
    client = object.__new__(AnthropicClient)  # skip __init__: no key, no network
    client.model, client.usage = "claude-sonnet-5-5", _Usage()
    client._c = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    return client


def _resp(stop, parsed=None):
    return SimpleNamespace(
        stop_reason=stop,
        parsed_output=parsed,
        stop_details=SimpleNamespace(category="cyber"),
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


@pytest.mark.parametrize(
    "stop,msg", [("refusal", "declined"), ("max_tokens", "truncated"), ("end_turn", "no structured")]
)
def test_anthropic_refusal_truncation_and_missing_output_raise_llmerror(stop, msg):
    with pytest.raises(LLMError, match=msg):
        _anthropic(lambda **kw: _resp(stop)).complete("s", "u")


def test_anthropic_success_records_tokens_and_sends_effort(ds, profile):
    pack, m, _ = _setup(ds, profile)
    seen = {}

    def parse(**kw):
        seen.update(kw)
        from acquirer_engine.schemas import Rationale

        return _resp("end_turn", Rationale.model_validate(offline.write(pack, m)))

    client = _anthropic(parse)
    out = client.complete("s", "u")
    assert out["headline"] and client.usage.tokens_in == 100
    assert seen["output_config"] == {"effort": C.ANTHROPIC_EFFORT} and "temperature" not in seen


class Timeout(Exception):
    pass


class RateLimitError(Exception):
    pass


@pytest.mark.parametrize("exc", [Timeout("timed out"), RateLimitError("429 Too Many Requests")])
def test_timeout_and_rate_limit_become_a_failed_page_not_a_crash(tmp_path, ds, profile, exc):
    pack, m, target = _setup(ds, profile)

    def boom(**kw):
        raise exc

    res = generate(_anthropic(boom), Cache(tmp_path), "s", "u", pack, m, target)
    assert res.status == "failed" and type(exc).__name__ in res.issues[0] and res.attempts == 2


# ---------------------------------------------------------------- OpenAI-compatible client paths


def _openai(content, finish="stop", refusal=None, cls=OpenAIClient):
    client = object.__new__(cls)
    client.model, client.usage = "gpt-4o", _Usage()
    msg = SimpleNamespace(content=content, refusal=refusal)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=msg)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    client._c = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    return client


def test_openai_malformed_json_truncation_and_refusal():
    with pytest.raises(LLMError, match="invalid JSON"):
        _openai("not json at all").complete("s", "u")
    with pytest.raises(LLMError, match="truncated"):
        _openai("{}", finish="length").complete("s", "u")
    with pytest.raises(LLMError, match="refused"):
        _openai("{}", refusal="no").complete("s", "u")
    assert _openai('```json\n{"a": 1}\n```').complete("s", "u") == {"a": 1}


def test_gemini_daily_quota_stops_further_calls():
    GeminiClient._daily_quota_hit = True
    try:
        with pytest.raises(LLMError, match="quota"):
            _openai("{}", cls=GeminiClient).complete("s", "u")
    finally:
        GeminiClient._daily_quota_hit = False


def test_malformed_model_json_goes_through_repair(tmp_path, ds, profile):
    pack, m, target = _setup(ds, profile)
    good = offline.write(pack, m)
    calls = iter([{"headline": "missing every other field"}, good])
    client = SimpleNamespace(provider="fake", model="f", complete=lambda s, u: next(calls))
    res = generate(client, Cache(tmp_path), "s", "u", pack, m, target)
    assert res.status == "repaired"


def test_unknown_provider_and_missing_keys(monkeypatch):
    with pytest.raises(LLMError, match="Unknown provider"):
        make_client("nope", None)
    for provider in ("openai", "gemini"):
        with pytest.raises(LLMError, match="_API_KEY is not set"):
            make_client(provider, None)


# ---------------------------------------------------------------- cache, output, data


def test_corrupt_cache_entry_is_ignored_and_regenerated(tmp_path, ds, profile):
    pack, m, target = _setup(ds, profile)
    client = SimpleNamespace(provider="fake", model="f", complete=lambda s, u: offline.write(pack, m))
    cache = Cache(tmp_path)
    (tmp_path / f"{cache.key(client, 's', 'u')}.json").write_text("{ not json")
    res = generate(client, cache, "s", "u", pack, m, target)
    assert res.status == "verified" and not res.cached
    assert json.loads((tmp_path / f"{res.cache_key}.json").read_text())["status"] == "verified"  # healed


def test_unwritable_cache_does_not_fail_the_page(tmp_path, ds, profile):
    pack, m, target = _setup(ds, profile)
    blocker = tmp_path / "file"
    blocker.write_text("x")  # cache root is a file: mkdir fails
    client = SimpleNamespace(provider="fake", model="f", complete=lambda s, u: offline.write(pack, m))
    assert generate(client, Cache(blocker), "s", "u", pack, m, target).status == "verified"


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="permission bits not enforced")
def test_unwritable_output_dir_exits_with_render_code(tmp_path, capsys):
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert (
            main(
                [
                    "--provider",
                    "offline",
                    "--no-cache",
                    "--out-dir",
                    str(out),
                    "--feedback-file",
                    str(tmp_path / "f.json"),
                ]
            )
            == 6
        )
        assert "cannot write" in capsys.readouterr().err
    finally:
        out.chmod(stat.S_IRWXU)


def test_zero_eligible_acquirers_is_reported_not_crashed(tmp_path, capsys):
    df = pd.read_csv(C.DEFAULT_DATA)
    df["outcome"] = "Withdrawn"  # nobody has a closed deal -> nobody eligible
    df.to_csv(tmp_path / "none.csv", index=False)
    code = main(
        [
            "--provider",
            "offline",
            "--no-cache",
            "--data",
            str(tmp_path / "none.csv"),
            "--out-dir",
            str(tmp_path / "o"),
            "--feedback-file",
            str(tmp_path / "f.json"),
        ]
    )
    out = capsys.readouterr()
    assert code == 1 and "0 eligible acquirers" in out.out + out.err


def test_unexpected_error_is_one_line_unless_debug(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise KeyError("surprise")

    monkeypatch.setattr("acquirer_engine.cli.data.load", boom)
    assert main(["--provider", "offline", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "unexpected KeyError" in err and "Traceback" not in err
    with pytest.raises(KeyError):
        main(["--provider", "offline", "--out-dir", str(tmp_path), "--debug"])


# ---------------------------------------------------------------- settings precedence


def test_settings_precedence_file_env_set(tmp_path, monkeypatch):
    f = tmp_path / "s.yaml"
    f.write_text("MIN_COMPS: 6\nRECENT_YEARS: 4\nTOP_PRECEDENTS: 4\n")
    monkeypatch.setenv("ACQ_RECENT_YEARS", "5")
    try:
        over = settings.apply(f, ["TOP_PRECEDENTS=7"])
        assert (C.MIN_COMPS, C.RECENT_YEARS, C.TOP_PRECEDENTS) == (6, 5, 7)  # file < env < --set
        assert over == {"MIN_COMPS": 6, "RECENT_YEARS": 5, "TOP_PRECEDENTS": 7}
        assert settings.effective()["overrides"] == over
    finally:
        settings.reset()
    assert settings.DEFAULTS["MIN_COMPS"] == C.MIN_COMPS


@pytest.mark.parametrize(
    "sets,msg",
    [
        (["NOPE=1"], "unknown setting"),
        (["MIN_COMPS=abc"], "MIN_COMPS"),
        (["W_MASS=0.9"], "sum to 1"),
        (["MIN_COMPS"], "NAME=VALUE"),
    ],
)
def test_bad_settings_are_config_errors(sets, msg):
    with pytest.raises(ConfigError, match=msg):
        settings.apply(None, sets)
    settings.reset()


def test_bad_settings_file_and_cli_exit_code(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        settings.apply(bad, [])
    assert main(["--provider", "offline", "--set", "NOPE=1", "--out-dir", str(tmp_path)]) == 4
    settings.reset()
