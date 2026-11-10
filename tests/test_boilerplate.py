"""Task F: market-wide facts once per report; repeated risks and missing competitor caught."""

from acquirer_engine import evidence, offline, scoring, validate
from acquirer_engine.engine import cross_page_feedback
from acquirer_engine.llm import Result
from acquirer_engine.schemas import Rationale


def _setup(ds, profile):
    s, picks, _ = scoring.rank(ds, profile)
    m = evidence.market_comps(s, profile, ds.ref_year)
    packs = [evidence.build(s, c, profile, m, ds.ref_year) for c in picks]
    evidence.add_closest_competitors(packs)
    target = {**profile.model_dump(), "ebitda_margin_pct": m["target_margin_assumed_pct"]}
    return packs, m, target


def test_restating_market_figures_on_a_page_is_rejected(ds, profile):
    packs, m, target = _setup(ds, profile)
    raw = offline.write(packs[0], m)
    raw["valuation_context"] += f" Implied target EBITDA is ${m['implied_target_ebitda_mm']}M."
    issues = validate.check(Rationale.model_validate(raw), packs[0], m, target)
    assert any("market-wide figures" in i for i in issues)


def test_own_figure_equal_to_market_figure_is_allowed(ds, profile):
    packs, m, _ = _setup(ds, profile)
    own = {round(float(x), 2) for x in validate.allowed(packs[0])[0]}
    assert not (validate.market_only_figures(packs[0], m) & own)


def test_fit_thesis_must_name_closest_competitor(ds, profile):
    packs, m, target = _setup(ds, profile)
    raw = offline.write(packs[0], m)
    rival = packs[0]["closest_competitor"]["acquirer"]
    raw["strategic_fit_thesis"] = raw["strategic_fit_thesis"].replace(rival, "another sponsor")
    assert any("closest competitor" in i for i in validate.check(Rationale.model_validate(raw), packs[0], m, target))


def test_closest_competitor_is_deterministic_and_never_self(ds, profile):
    a, _, _ = _setup(ds, profile)
    b, _, _ = _setup(ds, profile)
    assert [p["closest_competitor"]["acquirer"] for p in a] == [p["closest_competitor"]["acquirer"] for p in b]
    assert all(p["closest_competitor"]["acquirer"] != p["acquirer"] for p in a)


def test_repeated_risk_needs_three_other_pages():
    boiler = "Regulatory context: several states require advance notice of healthcare deals, which may delay timing"
    flags = {
        "A": [boiler, "A-specific withdrawn deal MA-2020-0001"],
        "B": [boiler],
        "C": [boiler],
        "D": [boiler + " slightly"],
        "E": ["unique risk about bidders"],
    }
    out = validate.repeated_risks(flags)
    assert set(out) == {"A", "B", "C", "D"} and out["A"] == [boiler]
    assert validate.repeated_risks({k: flags[k] for k in "ABE"}) == {}  # only 1 other page: not repeated


def test_cross_page_feedback_targets_the_right_pages(ds, profile):
    packs, m, _ = _setup(ds, profile)
    results = []
    for pk in packs[:5]:
        r = Rationale.model_validate(offline.write(pk, m))
        r.risk_flags[0].detail = "Regulatory context: HSR filing and state notice laws apply and may delay closing"
        results.append(Result(pk["acquirer"], r, "verified"))
    fixes = cross_page_feedback(packs[:5], results)
    assert set(fixes) == {pk["acquirer"] for pk in packs[:5]} and "Market context" in next(iter(fixes.values()))
