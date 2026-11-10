"""Task E: deterministic 'who they are' line."""

from acquirer_engine import config as C
from acquirer_engine import evidence, offline, scoring, validate
from acquirer_engine.evidence import _top
from acquirer_engine.schemas import Rationale


def _packs(ds, profile):
    s, picks, _ = scoring.rank(ds, profile)
    m = evidence.market_comps(s, profile, ds.ref_year)
    return [evidence.build(s, c, profile, m, ds.ref_year) for c in picks], m


def test_top_reports_ties_instead_of_picking_one():
    assert _top({"Private": 9, "PE-Backed": 9, "Public": 5}, 24) == "Private and PE-Backed (9 each of 24)"
    assert _top({"All Cash": 7, "Stock": 3}, 24) == "All Cash (7 of 24)"
    assert _top({}, 0) == "n/a"


def test_descriptor_has_every_required_element_and_fits_overview(ds, profile):
    packs, _ = _packs(ds, profile)
    for pk in packs:
        d = pk["descriptor"]
        for part in ("deals", "last 3 years", "most deals in", "size sweet spot $", "(IQR)", "sellers", "financing"):
            assert part in d, (pk["acquirer"], part)
        assert len(d.split()) <= C.WORD_LIMITS["acquirer_overview"] * 0.6  # leaves room for priorities


def test_overview_led_by_descriptor_passes_checker(ds, profile):
    packs, market = _packs(ds, profile)
    target = {**profile.model_dump(), "ebitda_margin_pct": market["target_margin_assumed_pct"]}
    for pk in packs:
        r = Rationale.model_validate(offline.write(pk, market))
        assert r.acquirer_overview.startswith(pk["descriptor"])
        assert validate.check(r, pk, market, target) == []
