"""Contested columns: where the sources of one entity disagree, the system does not repair automatically."""
import pandas as pd

from src.detectors.d5_entity_consensus import column_agreement, contested_columns


def several_sources():
    """Four flights, each reported by three sources. The scheduled time always agrees; the actual time agrees in
    two of three rows; the source never repeats within a flight."""
    rows = []
    for f in range(4):
        for s in range(3):
            rows.append({"flight": f"F{f}", "src": f"s{s}", "sched": f"{8 + f}:00",
                         "act": f"{8 + f}:05" if s < 2 else f"{8 + f}:20"})
    return pd.DataFrame(rows)


def test_agreement_per_column():
    agree = column_agreement(several_sources())
    assert round(agree["act"], 2) == 0.67
    assert agree["src"] < 0.5


def test_only_partly_agreeing_columns_are_contested():
    df = several_sources()
    assert contested_columns(df, 0.7) == {"act"}       # src is not an attribute of the flight at all
    assert contested_columns(df, 0.6) == set()


def test_one_row_per_object_has_no_contested_columns():
    df = pd.DataFrame({"name": [f"beer {i}" for i in range(10)], "abv": ["5%"] * 10})
    assert contested_columns(df, 0.7) == set()
