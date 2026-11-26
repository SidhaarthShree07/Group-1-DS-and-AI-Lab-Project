"""M5 baselines: the label sets are nested, start with the M4 rows, and the M4 rows are the export_baselines draw."""
import random

from scripts.run_baselines_m5 import cell_records, label_order


def test_label_sets_are_nested_and_keep_the_m4_rows():
    for table, n in {"hospital": 159, "flights": 365, "beers": 361, "rayyan": 150, "movies": 1109}.items():
        order = label_order(table, n)
        assert len(order) == len(set(order)) == 50 and max(order) < n
        assert sorted(order[:20]) == sorted(random.Random(f"42:{table}").sample(range(n), 20))
        assert order == label_order(table, n)


def test_cell_records_follow_the_dirty_header_in_row_major_order():
    import pandas as pd
    dirty = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    cells = pd.DataFrame({"dirty_value": ["1", "x", "2", "y"], "clean_value": ["1", "z", "2", "y"], "error_type": ["", "p", "", ""]})
    recs = cell_records(dirty, cells)
    assert [(r["row"], r["column"]) for r in recs] == [(0, "a"), (0, "b"), (1, "a"), (1, "b")]
    assert recs[1]["clean_value"] == "z"
