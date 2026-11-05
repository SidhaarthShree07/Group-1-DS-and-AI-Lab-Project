"""Tests for stage 0 (D7): rules are parsed strictly, checked on the dirty table, and only kept rules flag cells."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detectors.llm_rules import build_prompt, check_rules, detect_with, parse_rules  # noqa: E402


def table():
    rows = [{"zip": "35233", "city": "birmingham", "state": "al", "score": "90"} for _ in range(20)]
    rows += [{"zip": "35801", "city": "huntsville", "state": "al", "score": "85"} for _ in range(20)]
    rows[3]["zip"] = "3523x"           # breaks the zip pattern
    rows[5]["city"] = "birmingxam"     # breaks zip -> city
    rows[7]["score"] = "900"           # out of range
    return pd.DataFrame(rows)


def test_prompt_uses_dirty_profile_only():
    text = build_prompt("toy", table())
    assert "zip" in text and "birmingham" in text and "JSON list" in text


def test_parse_keeps_only_known_kinds():
    raw = ('Here are rules: [{"kind": "pattern", "column": "zip", "regex": "^[0-9]{5}$"},'
           ' {"kind": "sql", "query": "drop"}, {"kind": "range", "column": "score", "min": 0, "max": 100},'
           ' {"kind": "determines", "columns": ["zip", "city"]}, {"kind": "range", "column": "x", "min": 5, "max": 1}]')
    rules = parse_rules(raw)
    assert [r["kind"] for r in rules] == ["pattern", "range", "determines"]
    assert parse_rules("no json here") == []


def test_check_keeps_true_rules_and_drops_hallucinated_ones():
    rules = parse_rules('[{"kind": "pattern", "column": "zip", "regex": "^[0-9]{5}$"},'
                        ' {"kind": "pattern", "column": "city", "regex": "^[0-9]+$"},'
                        ' {"kind": "allowed_values", "column": "state", "values": ["ca"]},'
                        ' {"kind": "range", "column": "score", "min": 0, "max": 100},'
                        ' {"kind": "determines", "columns": ["zip", "city"]},'
                        ' {"kind": "pattern", "column": "nope", "regex": ".*"}]')
    checked = check_rules(rules, table())
    kept = [(r["kind"], r.get("column") or tuple(r["columns"])) for r in checked if r["kept"]]
    assert ("pattern", "zip") in kept and ("range", "score") in kept and ("determines", ("zip", "city")) in kept
    assert ("pattern", "city") not in kept and ("allowed_values", "state") not in kept
    assert all(r["reason"] for r in checked)


def test_only_kept_rules_flag_cells():
    rules = check_rules(parse_rules('[{"kind": "pattern", "column": "zip", "regex": "^[0-9]{5}$"},'
                                    ' {"kind": "range", "column": "score", "min": 0, "max": 100},'
                                    ' {"kind": "pattern", "column": "city", "regex": "^[0-9]+$"}]'), table())
    flags = set(zip(detect_with(rules, table())["row"], detect_with(rules, table())["column"]))
    assert (3, "zip") in flags and (7, "score") in flags
    assert not any(col == "city" for _, col in flags)  # the hallucinated city rule was dropped
