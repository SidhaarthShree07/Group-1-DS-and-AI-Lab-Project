"""Tests for the prompt builder: no clean values, context switches, truncation of long values."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.profiling.context import TableContext, norm, shape  # noqa: E402


def make_table():
    rows = []
    for zip_code, city in (("35233", "birmingham"), ("35660", "sheffield"), ("35801", "huntsville")):
        for _ in range(4):
            rows.append({"zip": zip_code, "city": city, "state": "al"})
    rows[0]["city"] = "birmingxam"  # the dirty value
    return pd.DataFrame(rows)


def test_norm_and_shape():
    assert norm("  a &amp;  b\t") == "a & b"
    assert shape("birmingham") == "a{10}" and shape("35233") == "9{5}" and shape("a1") == "a9"


def test_prompt_has_dependency_evidence_and_no_clean_value():
    ctx = TableContext("toy", make_table())
    text = ctx.prompt(0, "city")
    assert "<-- SUSPECT" in text and "birmingxam" in text
    assert "zip='35233' -> city is 'birmingham' in 3 of 4 rows" in text


def test_blocks_can_be_switched_off():
    ctx = TableContext("toy", make_table())
    row_only = ctx.prompt(0, "city", blocks=[])
    assert "Column profile" not in row_only and "Dependency evidence" not in row_only and "Similar rows" not in row_only
    assert "Row:" in row_only


def test_long_values_are_cut():
    df = pd.DataFrame({"a": ["x" * 500] * 12, "b": ["y"] * 12})
    ctx = TableContext("toy", df)
    assert len(ctx.prompt(0, "a")) < 3000


def test_evidence_sentence_uses_dependency():
    ctx = TableContext("toy", make_table())
    sentence = ctx.evidence_sentence(0, "city", True, "rule")
    assert "birmingham" in sentence and "3 of 4" in sentence
