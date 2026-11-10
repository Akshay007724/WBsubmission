"""Task I: the strategic/sponsor floor is explicit, configurable and evidence-gated."""

import json

import pandas as pd

from acquirer_engine import scoring
from acquirer_engine.cli import main
from acquirer_engine.profile import TargetProfile
from acquirer_engine.scoring import Candidate


def test_default_floor_swaps_are_recorded(ds, profile):
    s = scoring.score_deals(ds, profile)
    _, warnings, displaced = scoring.select(scoring.aggregate(s), 10, 3, s, profile)
    assert [d["displaced"] for d in displaced] == ["KKR", "Vista Equity"]
    assert [d["replaced_by"] for d in displaced] == ["Advocate Health", "Ardent Health Services"]
    assert all(d["displaced_score"] > d["replacement_score"] for d in displaced)  # the trade-off, made visible
    assert not warnings


def test_floor_zero_is_pure_score_order(ds, profile):
    s = scoring.score_deals(ds, profile)
    ranked = scoring.aggregate(s)
    picks, _, displaced = scoring.select(ranked, 10, 0, s, profile)
    assert [c.acquirer for c in picks] == [c.acquirer for c in ranked[:10]] and displaced == []


def test_floor_pick_without_evidence_is_rejected_and_slot_kept():
    rows = [
        {"acquirer": "S1", "is_closed": True, "deal_size_mm": 5.0, "sector": "Dental"}
    ]  # far too small, wrong sector
    scored = pd.DataFrame(rows)
    cands = [Candidate(f"PE{i}", "Financial Sponsor", 1 - i / 100, 1, 0.1, 0.5, 0, 5, 3, 2023) for i in range(10)]
    cands.append(Candidate("S1", "Strategic", 0.1, 1, 0.1, 0.5, 0, 1, 1, 2023))
    picks, warnings, displaced = scoring.select(cands, 10, 3, scored, TargetProfile(ev_mm=200))
    assert "S1" not in [c.acquirer for c in picks] and displaced == []
    assert any("lacked a closed deal in the size band or target sector" in w for w in warnings)


def test_cli_flag_and_report_table(tmp_path):
    base = ["--provider", "offline", "--no-cache", "--feedback-file", str(tmp_path / "f.json")]
    assert main([*base, "--out-dir", str(tmp_path / "a")]) == 0
    rep = json.loads(next((tmp_path / "a").rglob("report.json")).read_text())
    assert rep["mix_floor"]["min_per_type"] == 3 and len(rep["mix_floor"]["displaced"]) == 2
    assert "Replaced by floor pick" in next((tmp_path / "a").rglob("report.html")).read_text()
    assert main([*base, "--out-dir", str(tmp_path / "b"), "--min-per-type", "0"]) == 0
    names = [
        p["candidate"]["acquirer"] for p in json.loads(next((tmp_path / "b").rglob("report.json")).read_text())["pages"]
    ]
    assert "KKR" in names and "Advocate Health" not in names
