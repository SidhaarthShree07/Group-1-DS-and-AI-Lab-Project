"""D2 fire-rate filter (M5): a D2-only cell in a column D2 flags mostly is dropped, any other cell is kept."""
import unittest

import pandas as pd

from src.detectors import d2_fire_gate


def frame(rows):
    return pd.DataFrame(rows, columns=["row", "column", "d1", "d2", "d5", "is_candidate"])


class D2FireGateTest(unittest.TestCase):
    def test_d2_only_cells_in_a_busy_column_are_dropped(self):
        rows = [(i, "name", 0.0, 0.9, 0.0, True) for i in range(8)] + [(i, "name", 0.0, 0.0, 0.0, False) for i in range(8, 10)]
        out = d2_fire_gate.apply(frame(rows), 0.5)
        self.assertFalse(out["is_candidate"].any())

    def test_cell_with_another_detector_is_kept(self):
        rows = [(i, "name", 0.0, 0.9, 0.0, True) for i in range(8)]
        rows[3] = (3, "name", 0.0, 0.9, 0.8, True)
        out = d2_fire_gate.apply(frame(rows), 0.5)
        self.assertEqual(out.loc[out["is_candidate"], "row"].tolist(), [3])

    def test_quiet_column_is_untouched(self):
        rows = [(0, "zip", 0.0, 0.9, 0.0, True)] + [(i, "zip", 0.0, 0.0, 0.0, False) for i in range(1, 10)]
        out = d2_fire_gate.apply(frame(rows), 0.5)
        self.assertTrue(out.loc[0, "is_candidate"])

    def test_input_is_not_modified(self):
        df = frame([(i, "name", 0.0, 0.9, 0.0, True) for i in range(4)])
        d2_fire_gate.apply(df, 0.5)
        self.assertTrue(df["is_candidate"].all())


if __name__ == "__main__":
    unittest.main()
