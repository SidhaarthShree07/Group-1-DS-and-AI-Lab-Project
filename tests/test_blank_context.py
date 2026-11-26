"""Blank-context lines (M5): only empty cells get them, the closed value set only for small columns."""
import unittest

import pandas as pd

from src.profiling.context import TableContext


def table():
    langs = ["english"] * 70 + ["spanish"] * 20 + [""] * 10
    return pd.DataFrame({"lang": langs, "age": [str(20 + i % 40) for i in range(100)], "sex": ["f", "m"] * 50})


class BlankContextTest(unittest.TestCase):
    def test_default_prompt_is_unchanged(self):
        d = table()
        self.assertEqual(TableContext("t", d).prompt(95, "lang"), TableContext("t", d, blank_context=False).prompt(95, "lang"))

    def test_non_empty_cells_are_unchanged_when_on(self):
        d = table()
        self.assertEqual(TableContext("t", d).prompt(3, "lang"), TableContext("t", d, blank_context=True).prompt(3, "lang"))

    def test_empty_cell_gets_the_two_lines(self):
        d = table()
        text = TableContext("t", d, blank_context=True).prompt(95, "lang")
        self.assertIn("an empty cell is not one of them", text)
        self.assertIn("in this row 2 of the other 2 cells are filled", text)
        self.assertIn("'english'", text)

    def test_open_column_gets_only_the_row_line(self):
        d = table()
        d.loc[5, "age"] = ""
        text = TableContext("t", d, blank_context=True).prompt(5, "age")
        self.assertNotIn("an empty cell is not one of them", text)
        self.assertIn("other 2 cells are filled", text)


if __name__ == "__main__":
    unittest.main()
