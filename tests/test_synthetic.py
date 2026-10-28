"""Tests for the ContextGuard synthetic-data generator (src/data/synthetic.py)."""

import random
import unittest
import pandas as pd
import numpy as np

from src.data.synthetic import (
    norm,
    is_number,
    numeric_columns,
    typo,
    column_stats,
    outlier_value,
    discover_dependencies,
    corrupt,
    RATES,
)


class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_norm_raha_rule(self):
        """Verifies Raha ground-truth normalisation: HTML unescape, whitespace collapse, strip."""
        self.assertEqual(norm("  A&nbsp; B  "), "A B")
        self.assertEqual(norm("test\t\n  value"), "test value")
        self.assertEqual(norm("clean"), "clean")

    def test_is_number(self):
        """Verifies identification of parseable numeric strings."""
        self.assertTrue(is_number("123"))
        self.assertTrue(is_number("-45.67"))
        self.assertTrue(is_number("0.0"))
        self.assertFalse(is_number("abc"))
        self.assertFalse(is_number(""))

    def test_numeric_columns(self):
        """Verifies detection of numeric columns."""
        df = pd.DataFrame({
            "num": [str(i) for i in range(50)],
            "text": [f"val_{i}" for i in range(50)],
        })
        num_cols = numeric_columns(df)
        self.assertIn("num", num_cols)
        self.assertNotIn("text", num_cols)

    def test_typo_modifies_value_under_raha_rule(self):
        """Verifies character-level corruption changes the value under norm()."""
        val = "birmingham"
        corrupted = typo(val, self.rng)
        self.assertNotEqual(norm(val), norm(corrupted))

    def test_column_stats_and_outliers(self):
        """Verifies column stats calculation and outlier value generation."""
        series = pd.Series([str(i) for i in range(1, 101)])
        stats = column_stats(series)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertTrue(stats["iqr"] > 0)
        self.assertTrue(stats["integer"])

        outlier = outlier_value("50", stats, self.rng)
        outlier_float = float(outlier)
        # Outlier must be pushed far beyond the distribution boundaries
        self.assertTrue(outlier_float > stats["max"] or outlier_float < stats["min"])

    def test_discover_dependencies(self):
        """Verifies functional dependency discovery on clean relational data."""
        df = pd.DataFrame({
            "zip": ["35233", "35233", "90210", "90210"] * 25,
            "city": ["birmingham", "birmingham", "beverly hills", "beverly hills"] * 25,
            "id": [str(i) for i in range(100)],
        })
        deps = discover_dependencies(df, max_deps=5)
        # zip -> city should be discovered as a strong dependency
        dep_pairs = [(a, b) for a, b in deps]
        self.assertIn(("zip", "city"), dep_pairs)

    def test_corrupt_reproducibility(self):
        """Verifies that the same seed produces identical corruptions."""
        df = pd.DataFrame({
            "cat": ["A", "B", "A", "B"] * 25,
            "subcat": ["X", "Y", "X", "Y"] * 25,
            "val": [str(i) for i in range(100)],
        })
        rng1 = random.Random("42:test_table")
        rng2 = random.Random("42:test_table")

        dirty1, recs1, deps1 = corrupt("test_table", df, rng1)
        dirty2, recs2, deps2 = corrupt("test_table", df, rng2)

        pd.testing.assert_frame_equal(dirty1, dirty2)
        self.assertEqual(recs1, recs2)
        self.assertEqual(deps1, deps2)

    def test_required_corruption_recipes(self):
        """Verifies all four error recipes are represented in the rates."""
        expected_recipes = {"missing_value", "pattern_violation", "rule_violation", "outliers"}
        self.assertEqual(set(RATES.keys()), expected_recipes)


if __name__ == "__main__":
    unittest.main()
