import pytest
from pydantic import ValidationError

from acquirer_engine import data
from acquirer_engine.config import DEFAULT_DATA
from acquirer_engine.profile import ProfileError, TargetProfile, load_yaml, validate_against


def _write(tmp_path, text, bom=False):
    p = tmp_path / "d.csv"
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode())
    return p


def _header_and_row():
    lines = DEFAULT_DATA.read_text(encoding="utf-8-sig").splitlines()
    return lines[0], lines[1]


def test_real_file_loads_clean(ds):
    assert len(ds.df) == 500 and ds.ref_year == 2024 and len(ds.sha256) == 64


def test_bom_and_messy_headers_are_normalized(tmp_path):
    header, row = _header_and_row()
    messy = ",".join(f" {h.upper()} " for h in header.split(","))
    loaded = data.load(_write(tmp_path, f"{messy}\n{row}\n", bom=True))
    assert "acquirer" in loaded.df.columns and len(loaded.df) == 1


def test_missing_column_is_reported(tmp_path):
    header, row = _header_and_row()
    cols = header.split(",")
    i = cols.index("deal_size_mm")
    bad_header = ",".join(c for j, c in enumerate(cols) if j != i)
    bad_row = ",".join(c for j, c in enumerate(row.split(",")) if j != i)
    with pytest.raises(data.DataError, match="deal_size_mm"):
        data.load(_write(tmp_path, f"{bad_header}\n{bad_row}\n"))


def test_bad_rows_dropped_and_counted_and_names_merged(tmp_path):
    header, row = _header_and_row()
    cols = header.split(",")
    vals = row.split(",")
    neg = list(vals)
    neg[cols.index("deal_size_mm")] = "-5"
    neg[cols.index("transaction_id")] = "X-1"
    case = list(vals)
    case[cols.index("acquirer")] = "  " + vals[cols.index("acquirer")].upper() + " "
    case[cols.index("transaction_id")] = "X-2"
    text = "\n".join([header, row, row, ",".join(neg), ",".join(case)]) + "\n"
    loaded = data.load(_write(tmp_path, text))
    assert len(loaded.df) == 2  # duplicate + negative size dropped
    assert loaded.df["acquirer"].nunique() == 1  # casing/whitespace variant merged
    assert any("duplicate" in n for n in loaded.notes) and any("non-positive" in n for n in loaded.notes)


def test_missing_and_empty_file(tmp_path):
    with pytest.raises(data.DataError):
        data.load(tmp_path / "nope.csv")
    with pytest.raises(data.DataError):
        data.load(_write(tmp_path, ""))


def test_profile_validation(ds):
    """Categorical values are validated against THIS file's vocabulary (case-insensitive, with suggestions)."""
    assert validate_against(TargetProfile(sector="healthcare services"), ds).sector == "Healthcare Services"
    with pytest.raises(ProfileError, match=r"Did you mean.*Healthcare Services"):
        validate_against(TargetProfile(sector="Helthcare Servics"), ds)
    with pytest.raises(ProfileError, match="geography"):
        validate_against(TargetProfile(geography="Atlantis"), ds)
    with pytest.raises(ValidationError):
        TargetProfile(ev_mm=-1)
    with pytest.raises(ProfileError):
        load_yaml("does-not-exist.yaml")
