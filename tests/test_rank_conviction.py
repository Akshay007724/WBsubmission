"""Task D: rank vs conviction divergence is explained, never silently reordered."""

from acquirer_engine.engine import rank_conviction_notes, top_signals


def page(rank, name, level, signals=("+2 a", "+1 b", "-1 c")):
    return {
        "rank": rank,
        "candidate": {"acquirer": name},
        "rationale": {"conviction": {"level": level}},
        "evidence": {"suggested_conviction": {"signals": list(signals)}},
    }


def test_consistent_order_gets_no_note():
    pages = [page(1, "A", "High"), page(2, "B", "Medium"), page(3, "C", "Low")]
    rank_conviction_notes(pages)
    assert all(p["rank_vs_conviction"] is None for p in pages)


def test_both_sides_of_a_divergence_are_explained():
    pages = [page(1, "A", "High"), page(2, "B", "Medium"), page(3, "C", "High")]
    rank_conviction_notes(pages)
    assert pages[0]["rank_vs_conviction"] is None
    assert "outranks C" in pages[1]["rank_vs_conviction"]
    assert "B rank higher" in pages[2]["rank_vs_conviction"] and "Ranked #3" in pages[2]["rank_vs_conviction"]


def test_failed_page_is_skipped():
    pages = [
        page(1, "A", "Low"),
        {
            "rank": 2,
            "candidate": {"acquirer": "B"},
            "rationale": None,
            "evidence": {"suggested_conviction": {"signals": []}},
        },
    ]
    rank_conviction_notes(pages)
    assert pages[1]["rank_vs_conviction"] is None


def test_top_signals_by_absolute_points():
    assert top_signals(["+1 a", "-2 b", "+0.5 c", "+2 d"]) == ["-2 b", "+2 d"]
