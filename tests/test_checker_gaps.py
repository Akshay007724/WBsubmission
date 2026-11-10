"""Task G: each new checker rule with a passing and a failing example."""

from acquirer_engine import evidence, offline, scoring, validate
from acquirer_engine.schemas import Rationale

DEALS = {"MA-2022-0001": {"ev_ebitda": 13.4, "ev_revenue": 2.1}, "MA-2022-0002": {"ev_ebitda": 25.7, "ev_revenue": 5.0}}


def test_multiple_before_id_checked():
    assert validate.misattributed("acquired at 13.4x [MA-2022-0001].", DEALS) == []
    assert validate.misattributed("acquired at 21.4x [MA-2022-0001].", DEALS)
    assert validate.misattributed("at 13.4x EV/EBITDA ([MA-2022-0001])", DEALS) == []


def test_multiple_range_over_a_cluster_of_ids_is_not_misattributed():
    assert validate.misattributed("priced at 13.4x to 25.7x [MA-2022-0002][MA-2022-0001].", DEALS) == []
    assert validate.misattributed("priced at 30.1x [MA-2022-0002][MA-2022-0001].", DEALS)


def test_earlier_peer_median_not_attributed_to_later_id():
    assert validate.misattributed("Peer median 15.6x, versus its deal [MA-2022-0001] at 13.4x.", DEALS) == []


def test_small_ints_free_unless_counting_something():
    nums = {24.0, 7.0}
    ctx = dict(validate.numbers_with_context("it has 2 risks and 3 deals"))
    assert ctx == {2.0: False, 3.0: True}
    assert validate._num_ok(2, nums, counts_something=False)
    assert not validate._num_ok(3, nums, counts_something=True)  # "3 deals" must be in evidence
    assert validate._num_ok(7, nums, counts_something=True)  # "7 closed deals" is in evidence
    assert validate.numbers_with_context("6 closed in-band deals")[0][1] is True


def test_dollar_sign_required():
    assert validate.BARE_DOLLAR_RE.findall("a 168.1-391.6M range") == ["168.1-391.6M"]
    assert validate.BARE_DOLLAR_RE.findall("a $168.1-391.6M range and $94.1M deal") == []
    assert validate.BARE_DOLLAR_RE.findall("EBITDA of 12.8M") == ["12.8M"]


def _advent(ds, profile):
    s, picks, _ = scoring.rank(ds, profile)
    m = evidence.market_comps(s, profile, ds.ref_year)
    c = next(c for c in picks if c.acquirer == "Advent International")
    return evidence.build(s, c, profile, m, ds.ref_year), m


def test_specialist_claim_on_a_tie_is_rejected(ds, profile):
    pack, _ = _advent(ds, profile)  # four-way tie at 3 deals
    assert validate.specialist_claims("Advent is a Behavioral Health specialist.", pack, [])
    assert validate.specialist_claims("Behavioral Health has 3 of its deals.", pack, []) == []  # factual, no claim
    assert (
        validate.specialist_claims("Unlike Welsh Carson, which focuses on Behavioral Health.", pack, ["Welsh Carson"])
        == []
    )


def test_specialist_claim_on_clear_top_sector_passes(ds, profile):
    s, picks, _ = scoring.rank(ds, profile)
    m = evidence.market_comps(s, profile, ds.ref_year)
    nordic = evidence.build(s, picks[0], profile, m, ds.ref_year)  # Physician Groups 10 of 24 = 42%
    assert validate.specialist_claims("Nordic is focused on Physician Groups.", nordic, []) == []
    assert validate.specialist_claims("Nordic is focused on Dental.", nordic, [])


def test_checker_wires_new_rules(ds, profile):
    pack, m = _advent(ds, profile)
    target = {**profile.model_dump(), "ebitda_margin_pct": m["target_margin_assumed_pct"]}
    raw = offline.write(pack, m)
    raw["headline"] = "Advent is a Behavioral Health specialist with deals of 168.1-391.6M."
    issues = " ".join(validate.check(Rationale.model_validate(raw), pack, m, target))
    assert "missing '$'" in issues and "focused on Behavioral Health" in issues


# Regression tests from the live v1.4 run: two checker false positives and one correct flag.
LIVE = {
    "MA-2022-0103": {"ev_ebitda": 11.6, "ev_revenue": 3.41},
    "MA-2020-0386": {"ev_ebitda": 16.4, "ev_revenue": 3.04},
    "MA-2024-0120": {"ev_ebitda": 9.3, "ev_revenue": 3.36},
    "MA-2022-0419": {"ev_ebitda": 20.3, "ev_revenue": 6.0},
}


def test_range_across_two_ids_is_attributed_correctly():
    assert validate.misattributed("comps range from 11.6x [MA-2022-0103] to 16.4x [MA-2020-0386].", LIVE) == []
    assert (
        validate.misattributed("9.3x on Crest Care [MA-2024-0120] against 20.3x on Star Practice [MA-2022-0419].", LIVE)
        == []
    )
    assert validate.misattributed(
        "comps range from 12.0x [MA-2022-0103] to 16.4x [MA-2020-0386].", LIVE
    )  # still caught


def test_specialist_rule_needs_the_sector_as_object(ds, profile):
    pack, _ = _advent(ds, profile)
    s = (
        "Only 1 of 18 deals is in Healthcare Services, spread across 9 sectors, so sector depth is lower "
        "than for sponsors concentrated in the vertical."
    )
    assert validate.specialist_claims(s, pack, []) == []
    assert validate.specialist_claims("Advent is concentrated in Healthcare Services.", pack, [])


def test_hedged_below_peers_is_not_counted():
    from acquirer_engine.eval import BELOW_PEERS_RE

    assert not BELOW_PEERS_RE.search("We cannot say whether it pays above, in line with or below peers.")
    assert BELOW_PEERS_RE.search("it prices at 9.3x, below its peers.")


def test_rounded_range_not_in_evidence_is_still_flagged():
    nums = {15.6, 15.3, 21.5}
    ctx = validate.numbers_with_context("margins are near 17-20%")
    assert [n for n, c in ctx if not validate._num_ok(n, nums, c)] == [17.0, 20.0]


def test_respectively_construction_is_not_misattributed():
    deals = {
        "MA-2022-0377": {"ev_ebitda": 15.6, "ev_revenue": 2.54},
        "MA-2021-0464": {"ev_ebitda": 15.3, "ev_revenue": 4.37},
    }
    ok = "its higher-margin precedents [MA-2022-0377] and [MA-2021-0464] cleared at 15.6x and 15.3x."
    assert validate.misattributed(ok, deals) == []
    bad = "its precedents [MA-2022-0377] and [MA-2021-0464] cleared at 18.0x and 19.0x."
    assert validate.misattributed(bad, deals)


def test_benchmark_multiple_after_an_id_is_not_attributed_to_that_deal():
    # Live-run false positive: 14.2x is the strategic peer median, not MA-2022-0001's multiple.
    text = "its comp median is 13.4x on 1 deal [MA-2022-0001] against a strategic peer median of 14.2x."
    assert validate.misattributed(text, DEALS) == []
    assert validate.misattributed("1 deal [MA-2022-0001] versus the market anchor of 15.6x.", DEALS) == []
    # A multiple written straight after the id is still checked.
    assert validate.misattributed("1 deal [MA-2022-0001] at 14.2x.", DEALS)
    assert validate.misattributed("[MA-2022-0001] cleared at 14.2x against peers.", DEALS)


def test_signed_evidence_value_matches_its_written_magnitude():
    # Live-run false positive: the pack holds gap -4.9; the page writes "a gap of -4.9 turns".
    nums, _ = validate.allowed({"pricing": {"gap_turns": -4.9}})
    assert validate._num_ok(4.9, nums)
    assert not validate._num_ok(5.9, nums)
