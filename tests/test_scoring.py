import math

from acquirer_engine import scoring
from acquirer_engine.scoring import Candidate


def test_size_kernel_peaks_at_target_and_is_log_symmetric():
    assert scoring.size_kernel(200, 200) == 1.0
    assert math.isclose(scoring.size_kernel(80, 200), scoring.size_kernel(500, 200), rel_tol=1e-9)
    assert scoring.size_kernel(2000, 200) < 0.1


def test_sector_adjacency_symmetric_with_default():
    assert scoring.sector_adjacency("Healthcare Services", "Dental") == scoring.sector_adjacency(
        "Dental", "Healthcare Services"
    )
    assert scoring.sector_adjacency("Dental", "Dental") == 1.0


def test_default_run_returns_ten_with_buyer_mix(ds, profile):
    _, picks, warnings = scoring.rank(ds, profile)
    assert len(picks) == 10 and not warnings
    types = [c.acquirer_type for c in picks]
    assert types.count("Strategic") >= 3 and types.count("Financial Sponsor") >= 3
    assert [c.score for c in picks] == sorted((c.score for c in picks), reverse=True)


def test_ranking_is_deterministic(ds, profile):
    a = [c.acquirer for c in scoring.rank(ds, profile)[1]]
    b = [c.acquirer for c in scoring.rank(ds, profile)[1]]
    assert a == b


def test_excluded_acquirer_is_replaced(ds, profile):
    first = scoring.rank(ds, profile)[1]
    top = first[0].acquirer
    again = scoring.rank(ds, profile, excluded=frozenset({top.casefold()}))[1]
    assert top not in [c.acquirer for c in again] and len(again) == 10


def test_closer_deal_raises_relevance(ds, profile):
    scored = scoring.score_deals(ds, profile)
    row = scored.iloc[[0]].copy()
    near, far = row.copy(), row.copy()
    near["deal_size_mm"], far["deal_size_mm"] = 210.0, 2100.0
    assert scoring.size_kernel(near["deal_size_mm"].iat[0], 200) > scoring.size_kernel(far["deal_size_mm"].iat[0], 200)


def _cand(name, typ, score):
    return Candidate(name, typ, score, 1.0, 0.1, 0.5, 0.0, 5, 3, 2023)


def test_mix_floor_and_shortfall_warnings():
    cands = [_cand(f"PE{i}", "Financial Sponsor", 1 - i / 100) for i in range(12)] + [_cand("S1", "Strategic", 0.1)]
    picks, warnings, _ = scoring.select(cands, 10)
    assert "S1" in [c.acquirer for c in picks]
    assert any("mix floor" in w for w in warnings)  # only 1 strategic exists
    few, w2, _ = scoring.select(cands[:4], 10)
    assert len(few) == 4 and any("Only 4 eligible" in w for w in w2)


def test_tiny_target_still_ranks(ds):
    from acquirer_engine.profile import TargetProfile

    _, picks, _ = scoring.rank(ds, TargetProfile(sector="Pharma/Biotech", ev_mm=5000, geography="West Coast"))
    assert picks
