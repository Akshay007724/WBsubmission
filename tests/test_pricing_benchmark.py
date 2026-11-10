"""Task B: a buyer's pricing is compared like-for-like, never against the high-margin anchor."""

import pandas as pd

from acquirer_engine import evidence
from acquirer_engine.profile import TargetProfile
from acquirer_engine.scoring import Candidate


def _row(i, acquirer, margin, mult, typ="Financial Sponsor"):
    return {
        "transaction_id": f"MA-2020-{i:04d}",
        "target_company": f"T{i}",
        "acquirer": acquirer,
        "acquirer_key": acquirer.casefold(),
        "sector": "Healthcare Services",
        "deal_year": 2022,
        "deal_time": 2022.5,
        "deal_type": "Platform Investment",
        "geography": "Midwest",
        "financing_type": "All Cash",
        "deal_size_mm": 200.0,
        "target_revenue_mm": 50.0,
        "target_ebitda_mm": 10.0,
        "ebitda_margin_pct": margin,
        "revenue_growth_pct": 10.0,
        "ev_ebitda_multiple": mult,
        "ev_revenue_multiple": 3.0,
        "outcome": "Closed",
        "tags": (),
        "num_bidders": 2,
        "days_to_close": 100.0,
        "acquirer_type": typ,
        "target_ownership_pre": "Private",
        "f_sector": 1.0,
        "w": 0.5,
    }


def toy():
    # Low-margin deals trade at 14x, high-margin deals at 18x (margin drives the multiple).
    rows = [_row(i, "Buyer", 8 + i, 14.0) for i in range(3)]  # buyer: low margin, 14x
    rows += [_row(10 + i, f"Peer{i}", 8 + i, 14.0) for i in range(4)]  # peers: low margin, 14x
    rows += [_row(20 + i, f"High{i}", 24 + i, 18.0) for i in range(5)]  # peers: high margin, 18x
    return pd.DataFrame(rows)


def test_low_margin_buyer_paying_peer_level_is_in_line_not_below():
    df, prof = toy(), TargetProfile(ev_mm=200, ebitda_margin_pct=25)
    market = evidence.market_comps(df, prof, 2024)
    cand = Candidate("Buyer", "Financial Sponsor", 1.0, 1.0, 0.1, 0.5, 0.0, 3, 3, 2022)
    pack = evidence.build(df, cand, prof, market, 2024)
    assert market["peer_set"]["ev_ebitda_median"] == 18.0  # the high-margin anchor would say "below"
    assert pack["pricing_vs_peers"]["peer_median"] == 14.0  # like-for-like peers: same filters, any margin
    assert pack["pricing_vs_peers"]["position"] == "in line with"


def test_position_labels_use_the_in_line_band():
    peer = {"n": 5, "ev_ebitda_median": 15.0}
    assert evidence.pricing_vs_peers({"n": 3, "ev_ebitda_median": 13.5}, peer, "Strategic")["position"] == "below"
    assert evidence.pricing_vs_peers({"n": 3, "ev_ebitda_median": 16.5}, peer, "Strategic")["position"] == "above"
    assert (
        evidence.pricing_vs_peers({"n": 3, "ev_ebitda_median": 15.9}, peer, "Strategic")["position"] == "in line with"
    )
    assert evidence.pricing_vs_peers({"n": 0}, peer, "Strategic") is None


def test_below_peers_counter():
    from acquirer_engine.eval import below_peers_count

    def page(txt):
        return {
            "rationale": {
                "headline": "",
                "acquirer_overview": "",
                "strategic_fit_thesis": txt,
                "valuation_context": "",
                "conviction": {"rationale": ""},
                "risk_flags": [],
                "precedent_activity": [],
            }
        }

    rep = {
        "pages": [
            page("It paid below sponsor peers."),
            page("Priced in line with peers."),
            page("1.7 turns below the Financial Sponsor peer median."),
            {"rationale": None},
        ]
    }
    assert below_peers_count(rep) == 2
