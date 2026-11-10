import pandas as pd

from acquirer_engine import eda
from acquirer_engine.cli import main


def _flags(ds, profile):
    return {f["id"]: f for f in eda.run(ds, profile)["flags"]}


def test_known_quirks_are_detected_on_the_provided_csv(ds, profile):
    f = _flags(ds, profile)
    for fid in [
        "sub_sector_redundant",
        "ev_does_not_reconcile",
        "thin_target_sector",
        "buyer_concentration",
        "size_skew",
        "stale_pending",
        "noisy_deal_type",
        "ubiquitous_tags",
        "synergy_backwards",
    ]:
        assert f[fid]["observed"], fid
    assert f["ev_does_not_reconcile"]["metrics"]["rows"] == 403
    assert (
        f["thin_target_sector"]["metrics"]["same_sector"] == 12
        or f["thin_target_sector"]["metrics"]["same_sector"] < 20
    )


def test_every_flag_states_the_engine_response(ds, profile):
    assert all(f["engine_response"] for f in eda.run(ds, profile)["flags"])


def test_cleaner_data_is_reported_as_not_observed(ds, profile):
    clean = ds.df.copy()
    clean["sub_sector"] = clean["sector"] + " / sub"
    clean["deal_size_mm"] = clean["ev_ebitda_multiple"] * clean["target_ebitda_mm"]
    f = {x["id"]: x for x in eda.quality_flags(clean, profile, ds.ref_year)}
    assert not f["sub_sector_redundant"]["observed"]
    assert not f["ev_does_not_reconcile"]["observed"]


def test_cramers_v_bounds():
    a = pd.Series(list("xxyy") * 10)
    assert eda.cramers_v(a, a) == 1.0
    assert eda.cramers_v(a, pd.Series(list("pqpq") * 10)) == 0.0


def test_eda_only_cli_needs_no_key(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert main(["--eda-only", "--out-dir", str(tmp_path)]) == 0
    md = next(tmp_path.rglob("eda.md")).read_text()
    assert "Findings and engine response" in md and "403" in md


def test_report_carries_data_profile(tmp_path):
    import json

    assert (
        main(
            [
                "--provider",
                "offline",
                "--no-cache",
                "--out-dir",
                str(tmp_path),
                "--feedback-file",
                str(tmp_path / "fb.json"),
            ]
        )
        == 0
    )
    rep = json.loads(next(tmp_path.rglob("report.json")).read_text())
    assert "sub_sector_redundant" in rep["data_profile"]["observed_flags"]


def test_financial_contradictions_are_flagged_with_row_ids(ds, profile):
    f = {x["id"]: x for x in eda.run(ds, profile)["flags"]}
    small = f["ebitda_too_small_for_deal_size"]
    assert small["observed"] is True and "MA-2022-0272" in small["metrics"]["transaction_ids"]
    assert "never used to compute value" in small["engine_response"]
    margin = f["margin_disagrees_with_financials"]
    assert margin["observed"] is True and margin["metrics"]["rows"] == 322


def test_consistent_financials_are_not_flagged(ds, profile):
    clean = ds.df.copy()
    clean["target_ebitda_mm"] = clean["deal_size_mm"] / clean["ev_ebitda_multiple"]
    clean["target_revenue_mm"] = 100 * clean["target_ebitda_mm"] / clean["ebitda_margin_pct"]
    f = {x["id"]: x for x in eda.financial_consistency_flags(clean)}
    assert not f["ebitda_too_small_for_deal_size"]["observed"] and not f["margin_disagrees_with_financials"]["observed"]
