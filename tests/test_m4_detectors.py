"""M4 detectors: D7 (numeric columns written in several formats) and the D5 entity-consensus extension."""
import pandas as pd

from src.detectors import d5_entity_consensus, d7_format_mix


def multi_source_flights():
    """Four sources report the same three flights; source a has no time for AA-1, source b a wrong one."""
    rows = []
    for source in ["a", "b", "c", "d"]:
        for flight, dep in [("AA-1-JFK-LAX", "7:53 a.m."), ("UA-2-SFO-IAH", "1:05 p.m."), ("DL-3-ATL-BOS", "9:10 a.m.")]:
            rows.append({"src": source, "flight": flight, "dep": dep, "arr": "x"})
    df = pd.DataFrame(rows)
    df.loc[0, "dep"] = ""
    df.loc[3, "dep"] = "7:58 a.m."
    return df


def test_d7_flags_mixed_formats_and_mixed_units_only():
    df = pd.DataFrame({
        "abv": ["0.05", "0.06", "0.05%", "0.07", "0.04", "0.06%", "0.05", "0.05", "0.06", "0.05"],
        "ounces": ["12.0 oz", "12.0 oz.", "16.0 ounce", "12.0 oz", "12.0 oz.", "12.0 oz", "12.0 ounce", "12.0 oz.",
                   "12.0 oz", "16.0 oz"],
        "sample": [f"{i} patients" for i in range(10, 20)],  # one unit everywhere: consistent
        "name": list("abcdefghij"),
    })
    out = d7_format_mix.detect(df)
    flagged = set(zip(out["row"], out["column"]))
    assert (2, "abv") in flagged and (5, "abv") in flagged and (0, "abv") not in flagged
    assert all((i, "ounces") in flagged for i in range(10))
    assert not any(col in ("sample", "name") for _, col in flagged)
    assert set(out["detector"]) == {"d7"}


def test_d7_value_formats():
    assert d7_format_mix.value_format("12") == "plain"
    assert d7_format_mix.value_format("12.0 oz") == "decorated"
    assert d7_format_mix.value_format("N/A") == "placeholder"
    assert d7_format_mix.value_format("Lager") == "other"
    assert d7_format_mix.unit_text("12.0 OZ.") == "oz."


def test_entity_consensus_flags_the_sources_that_disagree():
    out = d5_entity_consensus.detect(multi_source_flights(), "flights")
    flagged = set(zip(out["row"], out["column"]))
    assert (0, "dep") in flagged and (3, "dep") in flagged  # empty, and the source with another time
    assert (6, "dep") not in flagged                        # agrees with the other sources


def test_entity_consensus_is_silent_with_one_row_per_object():
    one_row_per_object = pd.DataFrame({"id": list("abcdef"), "v": list("123456")})
    assert d5_entity_consensus.detect(one_row_per_object, "t").empty
