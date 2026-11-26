"""M5 (X7) detector options: D2 free-text gate and the D7 extension. Defaults must stay the M4 detectors."""
import pandas as pd

from src.detectors import d2_pattern, d7_format_mix


def flagged(out):
    return set(zip(out["row"], out["column"]))


def test_d2_gate_keeps_only_encoding_damage_in_free_text():
    df = pd.DataFrame({"title": ["w" * (i + 1) for i in range(40)]})  # every shape is different
    df.loc[5, "title"] = "Barrag�n"
    assert len(d2_pattern.detect(df)) == 40                                     # M4 behaviour: all flagged
    assert flagged(d2_pattern.detect(df, min_unique_share=0.9, fallback="damage")) == {(5, "title")}
    assert d2_pattern.detect(df, min_unique_share=0.9).empty


def test_d2_fire_rate_gate_keeps_a_rare_shape_in_a_unique_id_column():
    df = pd.DataFrame({"id": [f"tt{1000000 + i}" for i in range(60)]})
    df.loc[7, "id"] = "american_ultra"
    assert flagged(d2_pattern.detect(df, max_fire_rate=0.5)) == {(7, "id")}


def test_d7_extension_flags_separators_trailing_zeros_and_ambiguous_dates():
    df = pd.DataFrame({"count": ["3,341", "912", "12,004", "87"] * 5, "value": ["7.0", "6.5", "7.4", "8.0"] * 5,
                       "date": ["1/1/15", "5/1/00", "1/1/73", "1/1/01"] * 5})
    assert d7_format_mix.detect(df).empty
    out = flagged(d7_format_mix.detect(df, extended=True))
    assert {(0, "count"), (2, "count"), (0, "value"), (3, "value"), (0, "date")} <= out
    assert not {(1, "count"), (1, "value"), (1, "date"), (2, "date"), (3, "date")} & out


def test_d7_extension_hyphen_unit_but_not_street_addresses():
    df = pd.DataFrame({"oz": ["12", "16", "12.0 oz. Alumi-Tek", "12"] * 5,
                       "address": ["1300 south montgomery avenue", "315 w hickory st"] * 10})
    out = flagged(d7_format_mix.detect(df, extended=True))
    assert (2, "oz") in out and not any(c == "address" for _, c in out)


def test_m5_detector_set_is_the_chosen_arm():
    from src.detectors import m5_detectors
    from src.detectors.candidates import DEFAULT_DETECTORS, M5_D2

    assert M5_D2 == {"min_unique_share": 0.9, "fallback": "rare_char", "rare_char_share": 0.02}
    assert set(m5_detectors("t")) == set(DEFAULT_DETECTORS) | {"d7", "d5c"}
    assert DEFAULT_DETECTORS["d2"] is d2_pattern.detect  # the M4 set is unchanged


def test_composite_key_is_off_by_default_and_silent_when_one_column_is_the_entity():
    from src.detectors import m5_detectors
    from src.detectors.x7_options import composite_consensus

    assert "o1b" not in m5_detectors("t") and "o1b" in m5_detectors("t", composite_key=True)
    rows = [{"city": c, "year": y, "mayor": f"{c}{y}", "src": s} for c in "abcd" for y in "1234" for s in "wxyz"]
    df = pd.DataFrame(rows)
    df.loc[5, "mayor"] = "wrong"
    assert flagged(composite_consensus(df.drop(columns="src"), "t", 0.85, 0.70)) == {(5, "mayor")}
    one_row = pd.DataFrame({"id": [str(i) for i in range(30)], "v": ["x"] * 30})
    assert composite_consensus(one_row, "t").empty
