"""Task K: the engine works on other M&A files and degrades with explicit notes, never invented numbers."""

from pathlib import Path

import pytest

from acquirer_engine import data, eda, scoring, units
from acquirer_engine.cli import main
from acquirer_engine.errors import DataError
from acquirer_engine.profile import TargetProfile, validate_against
from acquirer_engine.schema import map_headers

FIX = Path(__file__).parent / "fixtures"
BASELINE_TOP10 = [
    "Nordic Capital",
    "New Mountain Capital",
    "Blackstone",
    "Advent International",
    "Hellman & Friedman",
    "Warburg Pincus",
    "Aspen Dental (ADMI)",
    "Welsh Carson",
    "Advocate Health",
    "Ardent Health Services",
]


def top10(ds, **kw):
    return [c.acquirer for c in scoring.rank(ds, validate_against(TargetProfile(**kw), ds))[1]]


def test_original_csv_top10_unchanged(ds):
    assert top10(ds) == BASELINE_TOP10


def test_renamed_headers_same_answer():
    ds = data.load(FIX / "renamed_headers.csv")
    methods = {k: m.method for k, m in ds.schema.items()}
    assert methods["acquirer"] == "alias" and methods["deal_size_mm"] == "fuzzy"
    assert top10(ds) == BASELINE_TOP10  # same deals, different export -> same buyers


def test_tech_file_in_billions_with_fractions():
    ds = data.load(FIX / "tech_billions.csv")
    notes = " ".join(ds.notes)
    assert "billions" in notes and "fractions" in notes and "adjacency derived" in notes
    assert ds.vocab.adjacency_derived and ds.vocab.sector_preset is None
    assert 100 < ds.df["deal_size_mm"].median() < 200  # $154M median survived the unit round trip
    assert 10 < ds.df["ebitda_margin_pct"].median() < 20
    assert len(top10(ds, sector="Enterprise Software")) == 10


def test_missing_multiples_are_computed_and_labelled_missing_bidders_switch_off():
    ds = data.load(FIX / "missing_multiples_bidders.csv")
    assert ds.features["valuation"] == "computed"
    assert ds.features["competitive_process"].startswith("off")
    assert ds.df["num_bidders"].isna().all()  # stays missing; never filled


def test_no_multiples_and_no_financials_switches_valuation_off(tmp_path):
    import pandas as pd

    df = pd.read_csv(FIX / "missing_multiples_bidders.csv").drop(columns=["target_ebitda_mm", "target_revenue_mm"])
    df.to_csv(tmp_path / "nofin.csv", index=False)
    ds = data.load(tmp_path / "nofin.csv")
    assert ds.features["valuation"].startswith("off")
    assert (
        main(
            [
                "--provider",
                "offline",
                "--no-cache",
                "--data",
                str(tmp_path / "nofin.csv"),
                "--out-dir",
                str(tmp_path / "o"),
                "--feedback-file",
                str(tmp_path / "f.json"),
            ]
        )
        == 0
    )


def test_missing_core_column_is_a_clear_error():
    with pytest.raises(DataError, match=r"acquirer.*buyer_category"):
        data.load(FIX / "missing_core.csv")


def test_messy_file_is_cleaned_with_notes():
    ds = data.load(FIX / "messy.csv")
    notes = " ".join(ds.notes)
    for expected in ("blank row", "duplicate", "unit suffixes", "could not be read", "name variant"):
        assert expected in notes, expected
    assert ds.load_stats["coercion_failures_deal_size_mm"] == 1


def test_infer_acquirer_type_only_behind_flag(tmp_path):
    import pandas as pd

    pd.read_csv(FIX.parent.parent / "ma_transactions_500.csv").drop(columns=["acquirer_type"]).to_csv(
        tmp_path / "x.csv", index=False
    )
    off = data.load(tmp_path / "x.csv")
    assert off.features["buyer_mix"].startswith("off")
    on = data.load(tmp_path / "x.csv", infer_acquirer_type=True)
    assert on.features["buyer_mix"] == "inferred"
    assert (
        on.df.loc[on.df["acquirer"] == "Blackstone", "acquirer_type"].iat[0] == "Strategic"
    )  # name has no cue: honest miss
    assert on.df.loc[on.df["acquirer"] == "Nordic Capital", "acquirer_type"].iat[0] == "Financial Sponsor"


def test_schema_map_override_and_fuzzy_confidence():
    matches, _unused = map_headers(["Purchaser Name", "Deal Value USD", "Sectr", "Yr"], {"deal_year": "Yr"})
    assert matches["deal_year"].method == "override"
    assert matches["sector"].method == "fuzzy" and matches["sector"].confidence >= 0.8


def test_units():
    import pandas as pd

    v, failed, notes = units.parse_money(pd.Series(["$350M", "1.2B", "500K", "n/a", ""]), "deal_value")
    assert list(v.round(3).fillna(-1)) == [350.0, 1200.0, 0.5, -1, -1] and failed == 1
    v, _, notes = units.parse_money(pd.Series(["250000000", "90000000"]), "ev_usd")
    assert list(v) == [250.0, 90.0] and "raw dollars" in notes[0]
    y, q, failed, _ = units.parse_year_quarter(pd.Series(["2021-05-03", "03/15/2019", "bad"]), "date")
    assert list(y.fillna(0)) == [2021, 2019, 0] and list(q.fillna("")) == ["Q2", "Q1", ""] and failed == 1


def test_eda_adapts_and_reports_unchecked_items():
    ds = data.load(FIX / "missing_multiples_bidders.csv")
    out = eda.run(ds, TargetProfile())
    flags = {f["id"]: f for f in out["flags"]}
    assert flags["feature_off_competitive_process"]["observed"] is True
    assert set(out["charts"]) == {"size_distribution", "deals_per_year", "sector_mix", "multiple_vs_margin"}
    assert all(s.startswith("<svg") for s in out["charts"].values())


@pytest.mark.parametrize(
    "name,extra",
    [
        ("renamed_headers", []),
        ("tech_billions", ["--sector", "Enterprise Software"]),
        ("missing_multiples_bidders", []),
        ("messy", []),
    ],
)
def test_every_fixture_runs_eda_and_offline_end_to_end(tmp_path, name, extra):
    base = ["--data", str(FIX / f"{name}.csv"), "--out-dir", str(tmp_path), *extra]
    assert main([*base, "--eda-only"]) == 0
    assert main([*base, "--provider", "offline", "--no-cache", "--feedback-file", str(tmp_path / "f.json")]) == 0
    assert next(tmp_path.rglob("eda.md")).read_text().count("| ") > 20
    assert next(tmp_path.rglob("charts/*.svg"))


def test_missing_core_exits_with_data_code(tmp_path, capsys):
    assert main(["--data", str(FIX / "missing_core.csv"), "--provider", "offline", "--out-dir", str(tmp_path)]) == 3
    assert "Traceback" not in capsys.readouterr().err
