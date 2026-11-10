"""Task C: every branch of the size role."""

import pandas as pd
import pytest

from acquirer_engine.evidence import size_position


def g(sizes, types=None):
    types = types or ["Leveraged Buyout"] * len(sizes)
    return pd.DataFrame({"deal_size_mm": sizes, "deal_type": types})


HIST = g([50, 80, 100, 150, 200, 250, 300, 400, 500, 600])  # p25 ~112, p35 ~167, p75 ~375, p90 ~510


@pytest.mark.parametrize(
    "ev,typ,role",
    [
        (1_000, "Financial Sponsor", "stretch"),  # above its max
        (90, "Financial Sponsor", "add-on"),  # below 35th percentile
        (250, "Financial Sponsor", "platform"),  # inside IQR
        (550, "Financial Sponsor", "upper end of range"),  # above p75, not a stretch
        (90, "Strategic", "tuck-in"),
        (250, "Strategic", "core-sized"),
        (550, "Strategic", "core-sized"),  # strategics have no separate upper band
        (1_000, "Strategic", "stretch"),
    ],
)
def test_percentile_branches(ev, typ, role):
    assert size_position(HIST, ev, typ)["size_role"] == role


def test_stretch_by_2x_p90_even_below_max():
    hist = g([10] * 9 + [1000])  # p90 ~109, so 2x p90 ~218 < max
    assert size_position(hist, 500, "Financial Sponsor")["size_role"] == "stretch"


def test_add_on_by_half_median_even_above_p35():
    hist = g([100, 100, 100, 400, 400, 400, 400, 400, 400, 400])  # p35 = 400? no: median 400
    out = size_position(hist, 150, "Financial Sponsor")
    assert out["size_role"] == "add-on" and "half" in out["size_role_basis"]


def test_platform_labelled_deals_take_precedence():
    hist = g([30, 40, 50, 300, 320], ["Bolt-on Acquisition"] * 3 + ["Platform Investment"] * 2)
    assert size_position(hist, 250, "Financial Sponsor")["size_role"] == "platform"  # ~ its platform deals
    assert size_position(hist, 100, "Financial Sponsor")["size_role"] == "add-on"  # < half platform median
    assert "Platform Investment" in size_position(hist, 250, "Financial Sponsor")["size_role_basis"]


def test_platform_labels_ignored_when_target_far_above_them():
    hist = g([30, 40, 150, 200, 250, 300], ["Platform Investment"] * 2 + ["Leveraged Buyout"] * 4)
    out = size_position(hist, 220, "Financial Sponsor")  # 6x its platform median of 35
    assert "Platform Investment" not in out["size_role_basis"] and out["size_role"] == "platform"
    assert out["platform_deals"]["median_mm"] == 35.0  # still reported as a fact


def test_single_platform_label_is_not_evidence():
    hist = g([30, 40, 50, 60, 300], ["Bolt-on Acquisition"] * 4 + ["Platform Investment"])
    assert "Platform Investment" not in size_position(hist, 45, "Financial Sponsor")["size_role_basis"]


def test_capacity_uses_closed_deals_only():
    hist = pd.DataFrame(
        {
            "deal_size_mm": [50, 80, 100, 150, 6931.9],
            "deal_type": ["Leveraged Buyout"] * 5,
            "outcome": ["Closed"] * 4 + ["Withdrawn"],
        }
    )
    out = size_position(hist, 400, "Financial Sponsor")
    assert out["largest_closed_deal_mm"] == 150.0 and out["capacity_basis"] == "closed deals"
    assert out["size_role"] == "stretch" and "largest closed deal ($150.0M)" in out["size_role_basis"]


def test_capacity_falls_back_when_nothing_closed():
    hist = pd.DataFrame({"deal_size_mm": [50, 80], "deal_type": ["Carve-out"] * 2, "outcome": ["Pending"] * 2})
    assert size_position(hist, 60, "Strategic")["capacity_basis"] == "all deals (none closed)"
