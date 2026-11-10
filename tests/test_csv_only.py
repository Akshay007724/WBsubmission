"""Every fact on a page must come from the deals CSV: no outside knowledge anywhere in the pipeline."""

import json

from acquirer_engine import config as C
from acquirer_engine import evidence, offline, prompts, scoring, validate
from acquirer_engine.cli import main
from acquirer_engine.schemas import Rationale


def _setup(ds, profile):
    s, picks, _ = scoring.rank(ds, profile)
    m = evidence.market_comps(s, profile, ds.ref_year)
    packs = [evidence.build(s, c, profile, m, ds.ref_year) for c in picks]
    evidence.add_closest_competitors(packs)
    return packs[0], m, {**profile.model_dump(), "ebitda_margin_pct": m["target_margin_assumed_pct"]}


def test_output_schema_has_no_field_for_outside_facts():
    assert "external_context" not in Rationale.model_fields


def test_no_hand_written_world_knowledge_in_config_or_prompt():
    for name in ("REGULATORY_CONTEXT", "NONPROFIT_SYSTEMS"):
        assert not hasattr(C, name)
    assert "outside knowledge" in prompts.SYSTEM and "<regulatory_context>" not in prompts.SYSTEM


def test_prompt_contains_only_data_derived_blocks(ds, profile):
    pack, m, _ = _setup(ds, profile)
    user = prompts.build_user(profile, pack, m, ["X: ..."], ["assumption"])
    blocks = {
        b
        for b in (
            "target_profile",
            "assumptions",
            "score_breakdown",
            "suggested_conviction",
            "evidence",
            "report_market_context",
            "other_candidates",
            "regulatory_context",
        )
        if f"<{b}>" in user
    }
    assert "regulatory_context" not in blocks and len(blocks) == 7


def test_a_year_not_in_the_data_is_rejected(ds, profile):
    pack, m, target = _setup(ds, profile)
    raw = offline.write(pack, m)
    raw["acquirer_overview"] += " It was acquired by a larger rival in 2031."
    issues = " ".join(validate.check(Rationale.model_validate(raw), pack, m, target))
    assert "2031" in issues
    assert validate.check(Rationale.model_validate(offline.write(pack, m)), pack, m, target) == []  # data years pass


def test_rendered_report_has_no_outside_sections(tmp_path):
    assert (
        main(
            [
                "--provider",
                "offline",
                "--no-cache",
                "--out-dir",
                str(tmp_path),
                "--feedback-file",
                str(tmp_path / "f.json"),
            ]
        )
        == 0
    )
    html = next(tmp_path.rglob("report.html")).read_text()
    rep = json.loads(next(tmp_path.rglob("report.json")).read_text())
    for phrase in ("Outside the dataset", "Regulatory background", "HSR", "corporate-practice"):
        assert phrase not in html
    assert "regulatory_context" not in rep and all(
        "nonprofit_health_system" not in p["evidence"]["risk"] for p in rep["pages"]
    )
