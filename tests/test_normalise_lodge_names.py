from __future__ import annotations

import pandas as pd
import pytest

from lodge_classifier.io.normalise_lodge_names import normalise_lodge_names


def test_normalise_lodge_names_creates_expected_rows() -> None:
    """
    Ensure the normalisation produces one row per name event and preserves
    provenance fields and number/flag columns.
    """
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "LodgeName": "Alpha Lodge",
                "DateNamed": "1890-01-01",
                "OtherName1": "Alpha Lodge No. 1",
                "OtherDate1": "1900-01-01",
                "OtherName2": "",
                "OtherDate2": None,
                "LodgeNumber1894": 123,
                "Lapsed": 0,
                "Erased": 1,
                "LodgeNumber1729": 10,
                "ReNumber1729": 20,
            }
        ]
    )

    out = normalise_lodge_names(df)

    assert len(out) == 2
    assert set(out["name_type"].tolist()) == {"original", "alternate"}
    assert set(out["name_slot"].tolist()) == {"original", "other_1"}
    assert out["original_id"].nunique() == 1

    # Preserve flags and number columns
    assert out["Lapsed"].tolist() == [0, 0]
    assert out["Erased"].tolist() == [1, 1]
    assert out["LodgeNumber1729"].tolist() == [10, 10]
    assert out["ReNumber1729"].tolist() == [20, 20]


def test_normalise_lodge_names_raises_on_missing_required_cols() -> None:
    """
    Ensure required columns are enforced.
    """
    df = pd.DataFrame([{"ID": 1}])
    with pytest.raises(ValueError):
        normalise_lodge_names(df)
