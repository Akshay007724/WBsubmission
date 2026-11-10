import copy

from acquirer_engine import evidence, offline, scoring, validate
from acquirer_engine.llm import Cache, generate
from acquirer_engine.schemas import Rationale


def _setup(ds, profile):
    scored, picks, _ = scoring.rank(ds, profile)
    market = evidence.market_comps(scored, profile)
    pack = evidence.build(scored, picks[0], profile, market, ds.ref_year)
    target = {**profile.model_dump(), "ebitda_margin_pct": market["target_margin_assumed_pct"]}
    return pack, market, target


def test_comps_use_closed_deals_only(ds, profile):
    scored, _, _ = scoring.rank(ds, profile)
    market = evidence.market_comps(scored, profile)
    ids = set(market["peer_set"]["transaction_ids"])
    assert ids and set(ds.df[ds.df["transaction_id"].isin(ids)]["outcome"]) == {"Closed"}


def test_comp_set_widens_when_thin(ds):
    from acquirer_engine.profile import TargetProfile

    scored = scoring.score_deals(ds, TargetProfile(ev_mm=9000))
    market = evidence.market_comps(scored, TargetProfile(ev_mm=9000))
    assert len(market["widening_steps"]) > 1


def test_offline_draft_passes_checker(ds, profile):
    pack, market, target = _setup(ds, profile)
    r = Rationale.model_validate(offline.write(pack, market))
    assert validate.check(r, pack, market, target) == []


def test_checker_flags_fabrications(ds, profile):
    pack, market, target = _setup(ds, profile)
    raw = offline.write(pack, market)
    raw["strategic_fit_thesis"] += " It paid 42.7x for [MA-1999-0001], a leading platform."
    issues = validate.check(Rationale.model_validate(raw), pack, market, target)
    text = " ".join(issues)
    assert "42.7" in text and "MA-1999-0001" in text and "leading" in text


def test_similarity_detects_near_duplicates():
    a = "The sponsor has closed four platform deals in physician groups at a median multiple"
    assert validate.similar_pairs(
        {"A": a, "B": a + " too", "C": "Completely different words about dental roll ups"}
    ) == [("A", "B", validate.similar_pairs({"A": a, "B": a + " too"})[0][2])]


class FlakyClient:
    """First answer invents a number; second answer is clean. Exercises the repair loop."""

    provider, model = "fake", "fake-1"

    def __init__(self, good):
        self.good, self.calls = good, 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls == 1:
            bad = copy.deepcopy(self.good)
            bad["valuation_context"] = "Pays 99.9x [MA-2020-0001]."
            return bad
        assert "<previous_answer>" in user and "99.9" in user
        return self.good


def test_repair_loop_and_cache(tmp_path, ds, profile):
    pack, market, target = _setup(ds, profile)
    client = FlakyClient(offline.write(pack, market))
    cache = Cache(tmp_path)
    res = generate(client, cache, "sys", "user", pack, market, target)
    assert res.status == "repaired" and client.calls == 2
    again = generate(client, cache, "sys", "user", pack, market, target)
    assert again.cached and client.calls == 2


class BrokenClient:
    provider, model = "fake", "broken"

    def complete(self, system, user):
        raise TimeoutError("simulated timeout")


def test_total_failure_yields_failed_status_not_crash(tmp_path, ds, profile):
    pack, market, target = _setup(ds, profile)
    res = generate(BrokenClient(), Cache(tmp_path), "s", "u", pack, market, target)
    assert res.status == "failed" and res.rationale is None and "TimeoutError" in res.issues[0]


def test_numbers_from_roster_and_assumptions_are_allowed(ds, profile):
    pack, market, target = _setup(ds, profile)
    raw = offline.write(pack, market)
    raw["strategic_fit_thesis"] += " Unlike a peer at a $88.8M median, it targets the 75th percentile margin."
    r = Rationale.model_validate(raw)
    assert any("88.8" in i for i in validate.check(r, pack, market, target))
    shown = {
        **target,
        "assumptions": ["margin = dataset 75th percentile"],
        "other_candidates": ["X: median size $88.8M"],
    }
    assert validate.check(r, pack, market, shown) == []


def test_strip_fences():
    from acquirer_engine.llm import _strip_fences

    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('{"a": {"b": 2}}') == '{"a": {"b": 2}}'


def test_multiple_must_belong_to_the_cited_deal(ds, profile):
    pack, _market, _target = _setup(ds, profile)
    d = pack["precedents"][0]
    deals = {d["transaction_id"]: d}
    tid = d["transaction_id"]
    assert validate.misattributed(f"[{tid}] closed at {d['ev_ebitda']}x EBITDA.", deals) == []
    wrong = round(d["ev_ebitda"] + 9.8, 1)
    assert validate.misattributed(f"[{tid}] illustrating this at {wrong}x.", deals)
    # A different multiple later in another clause is not attributed to the deal.
    assert (
        validate.misattributed(f"[{tid}] closed at {d['ev_ebitda']}x, versus a peer median of {wrong}x.", deals) == []
    )


def test_live_runs_ignore_unverified_cache_but_replay_serves_it(tmp_path, ds, profile):
    from acquirer_engine.llm import ReplayClient

    pack, market, target = _setup(ds, profile)
    cache = Cache(tmp_path)
    replay = ReplayClient()
    key = cache.key(replay, "s", "u")
    cache.put(key, {"rationale": offline.write(pack, market), "status": "unverified", "issues": ["x"]})
    assert generate(replay, cache, "s", "u", pack, market, target).status == "unverified"
    calls = []

    class Live:
        provider, model, cache_provider = "anthropic", replay.model, "anthropic"

        def complete(self, s, u):
            calls.append(1)
            return offline.write(pack, market)

    assert generate(Live(), cache, "s", "u", pack, market, target).status == "verified" and calls
