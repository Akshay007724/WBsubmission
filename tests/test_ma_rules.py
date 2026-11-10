"""Domain rules a senior M&A practitioner would insist on."""

from acquirer_engine import config as C
from acquirer_engine import evidence, scoring
from acquirer_engine.profile import TargetProfile


def _market(ds, profile):
    scored = scoring.score_deals(ds, profile)
    return scored, evidence.market_comps(scored, profile, ds.ref_year)


def test_comps_exclude_non_control_and_non_closed_deals(ds, profile):
    _, market = _market(ds, profile)
    comp = ds.df[ds.df["transaction_id"].isin(market["peer_set"]["transaction_ids"])]
    assert set(comp["outcome"]) == {"Closed"}
    assert not set(comp["deal_type"]) & C.COMP_EXCLUDE_DEAL_TYPES


def test_peers_split_by_buyer_type(ds, profile):
    _, market = _market(ds, profile)
    assert set(market["pricing_peers_by_buyer_type"]) == {"Strategic", "Financial Sponsor"}


def test_ubiquitous_tag_is_not_called_distinctive(ds, profile):
    scored, market = _market(ds, profile)
    _, picks, _ = scoring.rank(ds, profile)
    for c in picks:
        tags = evidence.build(scored, c, profile, market, ds.ref_year)["profile"]["distinctive_tags_lift_vs_market"]
        assert "Geographic Expansion" not in tags  # on ~90% of deals: no signal


def test_minority_stake_weighs_less_than_control_deal():
    assert (
        C.DEAL_TYPE_W["Minority Investment"] < C.DEAL_TYPE_W["Recapitalization"] < C.DEAL_TYPE_W["Platform Investment"]
    )


def test_size_position_flags_stretch_and_add_on(ds):
    g = ds.df[ds.df["acquirer"] == "Nordic Capital"]
    assert evidence.size_position(g, 10_000)["size_role"].startswith("stretch")
    assert (
        "add-on" in evidence.size_position(g, 1)["size_role"]
        or evidence.size_position(g, 1)["target_size_percentile_in_buyer_history"] == 0
    )


def test_same_region_overlap_raises_antitrust_for_strategics(ds):
    prof = TargetProfile(geography="Southwest")
    scored, market = _market(ds, prof)
    cands = {c.acquirer: c for c in scoring.aggregate(scored)}
    pack = evidence.build(scored, cands["Ardent Health Services"], prof, market, ds.ref_year)
    assert pack["risk"]["deals_in_target_region"] >= 1 and pack["risk"]["antitrust_signal"]
