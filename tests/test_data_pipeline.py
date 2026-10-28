"""Tests for the data pipeline pieces that do not need the raw downloads."""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.export_baselines import hc_constraint  # noqa: E402
from src.data.load_external import classify, is_artifact, split_groups  # noqa: E402
from src.data.synthetic import corrupt, discover_dependencies, typo  # noqa: E402
from src.profiling.dependencies import scan_dependencies  # noqa: E402


def test_format_only_changes_are_artifacts():
    assert is_artifact("3", "3.0") and is_artifact("1907", "01907")
    assert not is_artifact("3", "4")
    assert classify("3", "3.0", {"3"}) == "format_artifact"
    assert classify("birmingham", "", {"birmingham"}) == "missing_value"
    assert classify("birmingham", "sheffield", {"birmingham", "sheffield"}) == "rule_violation"
    assert classify("birmingham", "birmingxam", {"birmingham"}) == "pattern_violation"


def test_group_split_keeps_groups_together():
    groups = {k: list(range(k * 10, k * 10 + 10)) for k in range(30)}
    parts = split_groups(groups, 300)
    assert sum(len(v) for v in parts.values()) == 300
    owner = {}
    for split, rows in parts.items():
        for r in rows:
            assert owner.setdefault(r // 10, split) == split  # every group sits in one split


def test_dependency_scan_drops_keys_and_constant_columns():
    n = 200
    df = pd.DataFrame({"id": [str(i) for i in range(n)], "zip": [str(i % 10) for i in range(n)],
                       "city": [f"c{i % 10}" for i in range(n)], "const": ["x"] * n})
    found = scan_dependencies(df, pairs=False)
    pairs = set(zip(found.lhs, found.rhs))
    assert ("zip", "city") in pairs
    assert not any(a == "id" for a, _ in pairs)      # a key determines everything: filtered
    assert not any(b == "const" for _, b in pairs)   # a constant column is determined by anything: filtered


def test_holoclean_constraint_syntax():
    assert hc_constraint("zip", "city") == "t1&t2&EQ(t1.zip,t2.zip)&IQ(t1.city,t2.city)"
    assert hc_constraint("a, b", "c") == "t1&t2&EQ(t1.a,t2.a)&EQ(t1.b,t2.b)&IQ(t1.c,t2.c)"


def test_synthetic_is_seeded_and_finds_dependencies():
    n = 400
    clean = pd.DataFrame({"edu": [f"e{i % 8}" for i in range(n)], "edu_num": [str(i % 8) for i in range(n)],
                          "age": [str(20 + (i * 7) % 60) for i in range(n)], "note": [f"word{i % 50}" for i in range(n)]})
    assert ("edu", "edu_num") in discover_dependencies(clean)
    d1, r1, _ = corrupt("toy", clean, random.Random("1:toy"))
    d2, r2, _ = corrupt("toy", clean, random.Random("1:toy"))
    assert r1 == r2 and d1.equals(d2)
    assert {kind for _, _, kind, _, _ in r1} == {"missing_value", "pattern_violation", "rule_violation", "outliers"}
    assert typo("birmingham", random.Random(0)) != "birmingham"
