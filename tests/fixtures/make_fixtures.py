"""Regenerate the CSV fixtures for task K from the provided dataset (deterministic, seed-free).

    python tests/fixtures/make_fixtures.py

Each fixture exercises one way a real-world M&A export differs from the original file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "ma_transactions_500.csv"
QUARTER_DATE = {"Q1": "02-15", "Q2": "05-15", "Q3": "08-15", "Q4": "11-15"}
TECH = {
    "Healthcare Services": "Enterprise Software",
    "Physician Groups": "IT Services",
    "Home Health/Hospice": "Cybersecurity",
    "Behavioral Health": "Data & Analytics",
    "Dental": "Fintech",
    "Revenue Cycle": "Payments",
    "Health IT": "Cloud Infrastructure",
    "Health Insurance": "Insurtech",
    "Medical Devices": "Semiconductors",
    "Pharma/Biotech": "Hardware",
}


def renamed(df: pd.DataFrame) -> pd.DataFrame:
    """Same deals, different vendor export: renamed headers, a date instead of year+quarter, other status words."""
    out = df.copy()
    out["Announce Date"] = out["deal_year"].astype(str) + "-" + out["deal_quarter"].map(QUARTER_DATE)
    out["outcome"] = out["outcome"].replace({"Closed": "Completed", "Withdrawn": "Abandoned"})
    out["acquirer_type"] = out["acquirer_type"].replace({"Financial Sponsor": "PE", "Strategic": "Corporate"})
    out = out.drop(columns=["deal_year", "deal_quarter"]).rename(
        columns={
            "transaction_id": "Deal ID",
            "acquirer": "Buyer",
            "target_company": "Target Name",
            "sector": "Industry",
            "deal_size_mm": "Enterprise Value ($M)",
            "ev_ebitda_multiple": "EV/EBITDA",
            "ev_revenue_multiple": "EV/Revenue",
            "outcome": "Status",
            "acquirer_type": "Buyer Type",
            "ebitda_margin_pct": "EBITDA Margin",
        }
    )
    return out


def tech_billions(df: pd.DataFrame) -> pd.DataFrame:
    """A non-healthcare file: tech sectors, sizes in $ billions, percentages as fractions."""
    out = df.copy()
    out["sector"] = out["sector"].map(TECH)
    out["sub_sector"] = out["sector"]
    out["deal_value_bn"] = (out["deal_size_mm"] / 1000).round(4)
    for c in ("ebitda_margin_pct", "revenue_growth_pct", "synergy_pct_of_deal"):
        out[c] = (out[c] / 100).round(4)
    return out.drop(columns=["deal_size_mm"]).rename(
        columns={"ebitda_margin_pct": "ebitda_margin", "revenue_growth_pct": "revenue_growth"}
    )


def missing_multiples_bidders(df: pd.DataFrame) -> pd.DataFrame:
    """No multiples and no bidder counts (EBITDA and revenue remain, so multiples can be computed)."""
    return df.drop(columns=["ev_ebitda_multiple", "ev_revenue_multiple", "num_bidders"])


def missing_core(df: pd.DataFrame) -> pd.DataFrame:
    """No acquirer column: the engine cannot rank and must stop with a clear message."""
    return df.drop(columns=["acquirer"]).rename(columns={"acquirer_type": "buyer_category"}).head(50)


def messy(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicates, blank rows, unreadable numbers, money strings, percent signs, stray whitespace."""
    out = df.head(150).copy().astype(str)
    out.loc[0:9, "deal_size_mm"] = [f"${v}M" for v in out.loc[0:9, "deal_size_mm"]]
    out.loc[10, "deal_size_mm"] = "1.2B"
    out.loc[11, "deal_size_mm"] = "n/a"
    out.loc[12, "ev_ebitda_multiple"] = "12.x"
    out.loc[13:20, "ebitda_margin_pct"] = [f"{v}%" for v in out.loc[13:20, "ebitda_margin_pct"]]
    out.loc[21:25, "acquirer"] = ["  " + a.upper() + " " for a in out.loc[21:25, "acquirer"]]
    blank = pd.DataFrame([[""] * out.shape[1]] * 3, columns=out.columns)
    out = pd.concat([out, out.iloc[30:35], blank], ignore_index=True)  # 5 exact duplicates + 3 blank rows
    out.columns = [f"  {c.replace('_', ' ').title()} " for c in out.columns]  # messy headers
    return out


def main() -> None:
    df = pd.read_csv(SRC)
    for name, fn in [
        ("renamed_headers", renamed),
        ("tech_billions", tech_billions),
        ("missing_multiples_bidders", missing_multiples_bidders),
        ("missing_core", missing_core),
        ("messy", messy),
    ]:
        fn(df).to_csv(HERE / f"{name}.csv", index=False)
        print(f"wrote {name}.csv")


if __name__ == "__main__":
    main()
