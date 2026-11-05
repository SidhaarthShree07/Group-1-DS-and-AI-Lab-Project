"""Unit tests for the model output parser (src/evaluation/parse.py).

Author: Sahilpreet

Tests verify all requirements from the Milestone 3 specification:
1. Valid JSON with error and repair
2. Valid JSON with no error
3. JSON within Markdown code fences (```json and ```)
4. Invalid JSON (syntax errors, truncated outputs)
5. Missing required fields
6. Invalid error_type
7. Wrong field types (e.g. non-boolean is_error, non-string evidence)
8. Contradictory fields (e.g. is_error=False with repair or error_type != 'none')
9. Empty string and whitespace-only inputs
10. None input
11. Raha normalisation of repair values via src.profiling.context.norm
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Gracefully mock optional heavy dependencies if run in an environment without them
for _mod in ("numpy", "pandas"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from src.evaluation.parse import (
    ALLOWED_ERROR_TYPES,
    normalise_repair,
    parse_prediction,
)


class TestOutputParser(unittest.TestCase):
    """Test suite for prediction output parsing and schema validation."""

    def test_valid_json_with_error_and_repair(self):
        """Test valid JSON representing a detected error with a repair."""
        raw = (
            '{"is_error": true, "error_type": "pattern", '
            '"evidence": "Typo in state abbreviation", "repair": "AL"}'
        )
        res = parse_prediction(raw)
        self.assertTrue(res["valid_json"])
        self.assertTrue(res["is_error"])
        self.assertEqual(res["error_type"], "pattern")
        self.assertEqual(res["evidence"], "Typo in state abbreviation")
        self.assertEqual(res["repair"], "AL")

    def test_valid_json_with_no_error(self):
        """Test valid JSON representing a correct cell with no error."""
        raw = (
            '{"is_error": false, "error_type": "none", '
            '"evidence": "Value matches column pattern and dependencies", "repair": null}'
        )
        res = parse_prediction(raw)
        self.assertTrue(res["valid_json"])
        self.assertFalse(res["is_error"])
        self.assertEqual(res["error_type"], "none")
        self.assertEqual(res["evidence"], "Value matches column pattern and dependencies")
        self.assertIsNone(res["repair"])

    def test_valid_json_error_without_repair(self):
        """Test valid JSON where an error is detected but repair is null."""
        raw = (
            '{"is_error": true, "error_type": "outlier", '
            '"evidence": "Value is outside normal bounds", "repair": null}'
        )
        res = parse_prediction(raw)
        self.assertTrue(res["valid_json"])
        self.assertTrue(res["is_error"])
        self.assertEqual(res["error_type"], "outlier")
        self.assertEqual(res["evidence"], "Value is outside normal bounds")
        self.assertIsNone(res["repair"])

    def test_markdown_code_fences_json(self):
        """Test extraction of JSON wrapped in ```json ... ```."""
        raw = (
            "Here is the result:\n"
            "```json\n"
            "{\n"
            '  "is_error": true,\n'
            '  "error_type": "rule",\n'
            '  "evidence": "Zip 35233 implies city Birmingham",\n'
            '  "repair": "birmingham"\n'
            "}\n"
            "```\n"
            "Hope this helps!"
        )
        res = parse_prediction(raw)
        self.assertTrue(res["valid_json"])
        self.assertTrue(res["is_error"])
        self.assertEqual(res["error_type"], "rule")
        self.assertEqual(res["repair"], "birmingham")

    def test_markdown_code_fences_generic(self):
        """Test extraction of JSON wrapped in generic ``` ... ```."""
        raw = (
            "```\n"
            '{"is_error": true, "error_type": "missing", '
            '"evidence": "Cell is empty", "repair": "N/A"}\n'
            "```"
        )
        res = parse_prediction(raw)
        self.assertTrue(res["valid_json"])
        self.assertTrue(res["is_error"])
        self.assertEqual(res["error_type"], "missing")
        self.assertEqual(res["repair"], "N/A")

    def test_invalid_json_syntax(self):
        """Test malformed and truncated JSON strings."""
        malformed_inputs = [
            '{"is_error": true, "error_type":',
            "not a json at all",
            '{broken_key: "value"}',
            '{"is_error": true, "error_type": "pattern"',
            "{",
            "}",
        ]
        for raw in malformed_inputs:
            with self.subTest(raw=raw):
                res = parse_prediction(raw)
                self.assertFalse(res["valid_json"])
                self.assertFalse(res["is_error"])
                self.assertEqual(res["error_type"], "none")
                self.assertEqual(res["evidence"], "")
                self.assertIsNone(res["repair"])

    def test_missing_required_fields(self):
        """Test JSON objects missing one of the required keys."""
        cases = [
            # missing repair
            '{"is_error": true, "error_type": "pattern", "evidence": "Typo"}',
            # missing is_error
            '{"error_type": "pattern", "evidence": "Typo", "repair": "clean"}',
            # missing error_type
            '{"is_error": true, "evidence": "Typo", "repair": "clean"}',
            # missing evidence
            '{"is_error": true, "error_type": "pattern", "repair": "clean"}',
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                res = parse_prediction(raw)
                self.assertFalse(res["valid_json"])
                self.assertFalse(res["is_error"])
                self.assertEqual(res["error_type"], "none")
                self.assertIsNone(res["repair"])

    def test_invalid_error_type(self):
        """Test disallowed error types."""
        cases = [
            '{"is_error": true, "error_type": "syntax_error", "evidence": "Bad", "repair": "ok"}',
            '{"is_error": true, "error_type": "typo", "evidence": "Bad", "repair": "ok"}',
            '{"is_error": true, "error_type": "random_type", "evidence": "Bad", "repair": "ok"}',
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                res = parse_prediction(raw)
                self.assertFalse(res["valid_json"])
                self.assertFalse(res["is_error"])
                self.assertEqual(res["error_type"], "none")

    def test_wrong_field_types(self):
        """Test non-conforming field types."""
        cases = [
            # is_error is string
            '{"is_error": "true", "error_type": "pattern", "evidence": "X", "repair": "Y"}',
            # is_error is integer
            '{"is_error": 1, "error_type": "pattern", "evidence": "X", "repair": "Y"}',
            # evidence is integer
            '{"is_error": true, "error_type": "pattern", "evidence": 1234, "repair": "Y"}',
            # repair is a list
            '{"is_error": true, "error_type": "pattern", "evidence": "X", "repair": ["Y"]}',
            # repair is a dict
            '{"is_error": true, "error_type": "pattern", "evidence": "X", "repair": {"val": "Y"}}',
            # JSON array instead of object
            '[{"is_error": true, "error_type": "pattern", "evidence": "X", "repair": "Y"}]',
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                res = parse_prediction(raw)
                self.assertFalse(res["valid_json"])
                self.assertFalse(res["is_error"])
                self.assertEqual(res["error_type"], "none")

    def test_contradictory_fields(self):
        """Test responses with contradictory logic."""
        # 1. is_error=False, but error_type != "none"
        raw1 = (
            '{"is_error": false, "error_type": "pattern", '
            '"evidence": "Looks okay", "repair": null}'
        )
        res1 = parse_prediction(raw1)
        self.assertFalse(res1["valid_json"])
        self.assertFalse(res1["is_error"])

        # 2. is_error=False, but non-empty repair given
        raw2 = (
            '{"is_error": false, "error_type": "none", '
            '"evidence": "Looks okay", "repair": "unexpected repair"}'
        )
        res2 = parse_prediction(raw2)
        self.assertFalse(res2["valid_json"])
        self.assertFalse(res2["is_error"])

        # 3. is_error=True, but error_type="none"
        raw3 = (
            '{"is_error": true, "error_type": "none", '
            '"evidence": "Something wrong", "repair": "clean"}'
        )
        res3 = parse_prediction(raw3)
        self.assertFalse(res3["valid_json"])
        self.assertFalse(res3["is_error"])

    def test_empty_and_whitespace_inputs(self):
        """Test empty string and whitespace-only inputs."""
        for raw in ("", "   ", "\t\n  \n"):
            with self.subTest(raw=raw):
                res = parse_prediction(raw)
                self.assertFalse(res["valid_json"])
                self.assertFalse(res["is_error"])
                self.assertEqual(res["error_type"], "none")
                self.assertIsNone(res["repair"])

    def test_none_input(self):
        """Test None input."""
        res = parse_prediction(None)
        self.assertFalse(res["valid_json"])
        self.assertFalse(res["is_error"])
        self.assertEqual(res["error_type"], "none")
        self.assertEqual(res["evidence"], "")
        self.assertIsNone(res["repair"])

    def test_normalise_repair_function(self):
        """Test Raha normalisation wrapper using canonical src.profiling.context.norm."""
        self.assertIsNone(normalise_repair(None))
        self.assertEqual(normalise_repair("  A&amp;  B  "), "A& B")
        self.assertEqual(normalise_repair("clean\t\n  value"), "clean value")
        self.assertEqual(normalise_repair("unaltered"), "unaltered")

    def test_parse_prediction_normalise_repair_flag(self):
        """Test normalise_repair_value flag in parse_prediction."""
        raw = (
            '{"is_error": true, "error_type": "pattern", '
            '"evidence": "Whitespace issue", "repair": "  fixed &amp;  clean  "}'
        )
        # Without normalisation (default)
        res_raw = parse_prediction(raw, normalise_repair_value=False)
        self.assertEqual(res_raw["repair"], "  fixed &amp;  clean  ")

        # With normalisation
        res_norm = parse_prediction(raw, normalise_repair_value=True)
        self.assertEqual(res_norm["repair"], "fixed & clean")

    def test_allowed_error_types_set(self):
        """Ensure allowed error types match canonical project specification."""
        self.assertEqual(
            ALLOWED_ERROR_TYPES,
            {"missing", "pattern", "rule", "outlier", "none"},
        )


if __name__ == "__main__":
    unittest.main()
