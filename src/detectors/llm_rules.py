#!/usr/bin/env python3
"""Stage 0, detector D7: data rules proposed by the language model, checked on the dirty table (decision D21).

Author: Sidhaarth Shree

The base model reads the column names and a profile of the DIRTY table once per table and proposes rules of four
fixed kinds. No clean value is ever shown. Every rule is then checked on the dirty table and kept only if
  - it holds for at least MIN_SUPPORT of the non-missing values it applies to, and
  - it flags at most MAX_FLAGGED of the column's cells, and
  - for `determines`, the dependency has strength at least MIN_STRENGTH (the g3 measure of D5).
Kept rules become detector D7. Its flags only ADD candidates: the model and the confidence gate still decide every
cell, so a wrong rule can cost extra model calls but cannot change data.

Rule kinds (JSON):
  {"kind": "pattern",        "column": c, "regex": "^[0-9]{5}$"}
  {"kind": "allowed_values", "column": c, "values": ["al", "ak", ...]}
  {"kind": "range",          "column": c, "min": 0, "max": 100}
  {"kind": "determines",     "columns": [a, b]}            # a determines b
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling.context import MISSING, TableContext, norm, to_float  # noqa: E402

MIN_SUPPORT = 0.95
MAX_FLAGGED = 0.10
MIN_STRENGTH = 0.90
MAX_RULES = 15
KINDS = {"pattern", "allowed_values", "range", "determines"}

PROMPT = """You are given the columns of a data table and a profile of its values. Some values may be wrong.
Propose at most {max_rules} rules that correct values of this table must follow. Use only these four kinds:
- {{"kind": "pattern", "column": "<column>", "regex": "<Python regular expression for the whole value>"}}
- {{"kind": "allowed_values", "column": "<column>", "values": ["<value>", ...]}}  (only for columns with few distinct values)
- {{"kind": "range", "column": "<column>", "min": <number>, "max": <number>}}  (only for numeric columns)
- {{"kind": "determines", "columns": ["<column A>", "<column B>"]}}  (equal A values must have equal B values)
Propose a rule only if you are confident it holds for almost all correct rows. Answer with a JSON list only.

Table: {table} ({rows} rows)
Columns:
{columns}
"""


def column_summary(ctx: TableContext, col: str) -> str:
    p = ctx.profile(col)
    parts = [f"{'numeric' if p['numeric'] else 'text'}", f"{p['distinct']} distinct", f"{p['missing'] * 100:.0f}% missing"]
    top = ", ".join(f"{ctx.short(k, 24)!r}" for k in list(p["counts"].index[:5]))
    parts.append(f"frequent: {top}")
    if p["shapes"]:
        parts.append("patterns: " + ", ".join(ctx.short(s, 30) for s, _ in p["shapes"]))
    if p["numeric"]:
        parts.append(f"1st-99th percentile {p['p1']:g} to {p['p99']:g}")
    return f"- {col}: " + "; ".join(parts)


def build_prompt(table: str, dirty: pd.DataFrame) -> str:
    ctx = TableContext(table, dirty)
    cols = "\n".join(column_summary(ctx, c) for c in ctx.cols)
    return PROMPT.format(max_rules=MAX_RULES, table=table, rows=ctx.n, columns=cols)


def parse_rules(text: str) -> List[Dict]:
    """The JSON list from the model's answer; malformed entries are dropped, never repaired."""
    m = re.search(r"\[[\s\S]*\]", text or "")
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    rules = []
    for r in items if isinstance(items, list) else []:
        if not isinstance(r, dict) or r.get("kind") not in KINDS:
            continue
        if r["kind"] == "determines":
            cols = r.get("columns")
            if isinstance(cols, list) and len(cols) == 2 and all(isinstance(c, str) for c in cols) and cols[0] != cols[1]:
                rules.append({"kind": "determines", "columns": cols})
        elif isinstance(r.get("column"), str):
            if r["kind"] == "pattern" and isinstance(r.get("regex"), str) and len(r["regex"]) <= 200:
                rules.append({"kind": "pattern", "column": r["column"], "regex": r["regex"]})
            elif r["kind"] == "allowed_values" and isinstance(r.get("values"), list) and r["values"]:
                rules.append({"kind": "allowed_values", "column": r["column"], "values": [str(v) for v in r["values"]]})
            elif r["kind"] == "range" and all(isinstance(r.get(k), (int, float)) for k in ("min", "max")) and r["min"] <= r["max"]:
                rules.append({"kind": "range", "column": r["column"], "min": float(r["min"]), "max": float(r["max"])})
    return rules[:MAX_RULES]


def _violations(rule: Dict, df: pd.DataFrame) -> Optional[tuple]:
    """(column, applicable row indices, violating row indices) for a single-column rule, or None if it cannot apply."""
    col = rule["column"]
    if col not in df.columns:
        return None
    vals = df[col]
    rows = [i for i, v in enumerate(vals) if v.lower() not in MISSING]
    if rule["kind"] == "pattern":
        try:
            rx = re.compile(rule["regex"])
        except re.error:
            return None
        bad = [i for i in rows if not rx.fullmatch(vals.iat[i])]
    elif rule["kind"] == "allowed_values":
        allowed = {norm(v).lower() for v in rule["values"]}
        bad = [i for i in rows if vals.iat[i].lower() not in allowed]
    else:  # range
        nums = {i: to_float(vals.iat[i]) for i in rows}
        rows = [i for i in rows if nums[i] is not None]
        bad = [i for i in rows if not rule["min"] <= nums[i] <= rule["max"]]
    return col, rows, bad


def check_rules(rules: List[Dict], dirty: pd.DataFrame) -> List[Dict]:
    """Each rule with its support, flagged share, decision and reason (kept and dropped rules are both returned)."""
    df = dirty.map(norm).reset_index(drop=True)
    n = len(df)
    out = []
    for r in rules:
        rec = dict(r, support=None, flagged_share=None, kept=False, reason="")
        if r["kind"] == "determines":
            a, b = r["columns"]
            if a not in df.columns or b not in df.columns or n == 0:
                rec["reason"] = "unknown column"
            else:
                g = df.groupby([a, b]).size().reset_index(name="k")
                strength = float(g.groupby(a)["k"].max().sum() / n)
                flagged = len(_dependency_flags(df, a, b)) / n
                rec.update(support=round(strength, 4), flagged_share=round(flagged, 4))
                if df[a].nunique() > 0.6 * n:
                    rec["reason"] = "determinant is a key"
                elif strength < MIN_STRENGTH:
                    rec["reason"] = f"strength {strength:.2f} below {MIN_STRENGTH}"
                elif flagged > MAX_FLAGGED:
                    rec["reason"] = f"flags {flagged:.1%} of cells"
                else:
                    rec.update(kept=True, reason="holds on the dirty table")
        else:
            v = _violations(r, df)
            if v is None:
                rec["reason"] = "unknown column or invalid rule"
            else:
                col, rows, bad = v
                support = 1 - len(bad) / len(rows) if rows else 0.0
                flagged = len(bad) / n if n else 0.0
                rec.update(support=round(support, 4), flagged_share=round(flagged, 4))
                if not rows:
                    rec["reason"] = "no values to check"
                elif support < MIN_SUPPORT:
                    rec["reason"] = f"holds for {support:.1%} of values, below {MIN_SUPPORT:.0%}"
                elif flagged > MAX_FLAGGED:
                    rec["reason"] = f"flags {flagged:.1%} of cells"
                else:
                    rec.update(kept=True, reason="holds on the dirty table")
        out.append(rec)
    return out


def _dependency_flags(df: pd.DataFrame, a: str, b: str) -> List[int]:
    """Rows whose b disagrees with the majority b of their a group (groups of at least two rows)."""
    flags = []
    for _, idx in df.groupby(a).indices.items():
        if len(idx) < 2:
            continue
        maj = df[b].iloc[idx].value_counts().index[0]
        flags += [int(i) for i in idx if df[b].iat[i] != maj]
    return flags


def detect_with(rules: List[Dict], dirty: pd.DataFrame) -> pd.DataFrame:
    """D7 flags from KEPT rules, in the shape of D1 to D6: row, column, detector, score."""
    df = dirty.map(norm).reset_index(drop=True)
    recs = []
    for r in rules:
        if not r.get("kept"):
            continue
        if r["kind"] == "determines":
            a, b = r["columns"]
            recs += [{"row": i, "column": b, "detector": "d7", "score": 1.0} for i in _dependency_flags(df, a, b)]
        else:
            v = _violations(r, df)
            if v:
                col, _, bad = v
                recs += [{"row": i, "column": col, "detector": "d7", "score": 1.0} for i in bad]
    if not recs:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(recs).drop_duplicates(["row", "column"])


def make_detector(rules: List[Dict]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda dirty: detect_with(rules, dirty)
