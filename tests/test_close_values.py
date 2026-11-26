"""M5: the prompt can name a frequent value one or two edits away from the cell (a likely misspelling)."""
import pandas as pd

from src.profiling.context import TableContext, close_frequent_value, edit_distance


def test_edit_distance_with_limit():
    assert edit_distance("camp_noju", "camp_nou") == 1
    assert edit_distance("juvnetus", "juventus") == 2
    assert edit_distance("barcelona", "madrid") == 3          # over the limit


def test_close_frequent_value():
    counts = pd.Series({"Camp_Nou": 40, "Anfield": 30, "Camp_Noju": 1})
    assert close_frequent_value("Camp_Noju", 1, counts) == ("Camp_Nou", 40, 1)
    assert close_frequent_value("Camp_Nou", 40, counts) is None     # the frequent value itself
    assert close_frequent_value("Old_Trafford", 1, counts) is None  # nothing close


def test_prompt_shows_the_hint_only_when_asked():
    df = pd.DataFrame({"stadium": ["Camp_Nou"] * 6 + ["Camp_Noju"], "team": ["Barca"] * 7})
    with_hint = TableContext("t", df, close_values=True).prompt(6, "stadium")
    assert "similar spelling: 'Camp_Nou'" in with_hint
    assert "similar spelling" not in TableContext("t", df).prompt(6, "stadium")


def test_values_with_digits_are_skipped():
    import pandas as pd
    from src.profiling.context import close_frequent_value
    counts = pd.Series({"6:03 a.m.": 20, "Camp_Nou": 20})
    assert close_frequent_value("6:08 a.m.", 1, counts) is None
    assert close_frequent_value("Camp_Noju", 1, counts)[0] == "Camp_Nou"
