#!/usr/bin/env python3
"""Table context for one cell: row, column profile, dependency evidence and similar rows.

Author: Sidhaarth Shree

Everything here is computed from the DIRTY table only; no clean value is read. The same code
builds the prompts for fine-tuning (src/llm/build_examples.py) and, later, for the app, so
training and use look the same to the model. It is the packaged form of the prompt builder
used in the model comparison (scripts/model_bakeoff/prep_prompts.py), with switches for the
three context blocks so that the "with and without context" experiment (E2) needs no new code.
"""
from __future__ import annotations

import collections
import html
import re
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

MISSING = {"", "nan", "null", "none", "n/a", "na", "-", "?", "not available"}  # same set as the model comparison
ALL_BLOCKS = ("profile", "dependencies", "similar")

TASK = (
    "Task: decide whether the marked cell is an error. Use the row, the column profile and the dependency evidence.\n"
    "Error types: missing (empty, null or placeholder value), pattern (wrong format, typo or inconsistent representation), "
    "rule (violates a dependency with other columns), outlier (implausible value for this column).\n\n"
)
ANSWER_FORMAT = (
    'Answer with JSON only, in this format: {"is_error": true or false, "error_type": "missing|pattern|rule|outlier|none", '
    '"evidence": "<one short sentence>", "repair": "<corrected value, or null if not an error>"}'
)


def norm(value) -> str:
    """The Raha comparison rule: unescape HTML, collapse tab/newline/space runs, strip the ends."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub("[\t\n ]+", " ", text)
    return text.strip("\t\n ")


def shape(value: str) -> str:
    """Letters become a/A, digits 9, other characters stay; runs longer than 4 are written c{n}."""
    out, prev, run = [], None, 0
    for ch in value:
        key = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch
        if key == prev:
            run += 1
        else:
            if prev is not None:
                out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
            prev, run = key, 1
    if prev is not None:
        out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
    return "".join(out)


def to_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


class TableContext:
    """Statistics of one dirty table (a split of it, or an uploaded file)."""

    def __init__(self, name: str, dirty: pd.DataFrame, key_share: float = 0.6):
        self.name = name
        self.df = dirty.map(norm).reset_index(drop=True)
        self.cols: List[str] = list(self.df.columns)
        self.n = len(self.df)
        self.key_share = key_share  # a determinant with more distinct values than this share of rows is a key
        self._profile: Dict[str, dict] = {}
        self._index: Dict[str, dict] = {}
        self._fd: Optional[Dict[tuple, float]] = None

    # ---- statistics ------------------------------------------------------------------------
    def profile(self, col: str) -> dict:
        if col in self._profile:
            return self._profile[col]
        s = self.df[col]
        counts = s.value_counts()
        values = [v for v in s if v.lower() not in MISSING]
        nums = [to_float(v) for v in values]
        nums_ok = [x for x in nums if x is not None]
        numeric = len(values) >= 10 and len(nums_ok) / max(1, len(values)) >= 0.9
        shapes = collections.Counter(shape(v) for v in values)
        p = {
            "distinct": int(s.nunique()),
            "counts": counts,
            "missing": float(np.mean([v.lower() in MISSING for v in s])) if self.n else 0.0,
            "numeric": numeric,
            "shapes": shapes.most_common(3),
            "shape_total": sum(shapes.values()),
            "shape_counts": shapes,
        }
        if numeric:
            a = np.array(nums_ok)
            p.update(median=float(np.median(a)), p1=float(np.percentile(a, 1)), p99=float(np.percentile(a, 99)),
                     min=float(a.min()), max=float(a.max()))
        self._profile[col] = p
        return p

    def rows_with(self, col: str, value: str) -> List[int]:
        if col not in self._index:
            self._index[col] = {k: list(v) for k, v in self.df.groupby(col).indices.items()}
        return self._index[col].get(value, [])

    def dependencies(self) -> Dict[tuple, float]:
        """strength[(A, B)] = share of rows whose B equals the most common B of their A group (1 - g3)."""
        if self._fd is None:
            self._fd = {}
            for a in self.cols:
                if self.n == 0 or self.df[a].nunique() > self.key_share * self.n:
                    continue
                for b in self.cols:
                    if a == b:
                        continue
                    g = self.df.groupby([a, b]).size().reset_index(name="k")
                    self._fd[(a, b)] = float(g.groupby(a)["k"].max().sum() / self.n)
        return self._fd

    @staticmethod
    def short(value: str, limit: int = 40) -> str:
        return value if len(value) <= limit else value[: limit - 1] + "…"

    # ---- facts about one cell ---------------------------------------------------------------
    def facts(self, i: int, col: str, max_chars: int = 40, n_similar: int = 3) -> dict:
        """Everything the prompt (and the evidence sentence) needs about cell (i, col)."""
        val = self.df.at[i, col]
        prof = self.profile(col)
        fd = self.dependencies()
        determined_by, determines = [], []
        best = None
        for s, a in sorted(((s, a) for (a, b), s in fd.items() if b == col and s >= 0.9), reverse=True)[:2]:
            rows = self.rows_with(a, self.df.at[i, a])
            if len(rows) >= 2:
                cnt = collections.Counter(self.df.at[r, col] for r in rows)
                majority, k = cnt.most_common(1)[0]
                determined_by.append({"col": a, "key": self.df.at[i, a], "majority": majority, "k": k, "n": len(rows)})
                best = best or (a, rows)
        rows_same = self.rows_with(col, val)
        if len(rows_same) >= 2:
            for s, b in sorted(((s, b) for (a, b), s in fd.items() if a == col and s >= 0.9), reverse=True)[:2]:
                cnt = collections.Counter(self.df.at[r, b] for r in rows_same)
                majority, k = cnt.most_common(1)[0]
                if self.df.at[i, b] != majority:
                    determines.append({"col": b, "majority": majority, "k": k, "n": len(rows_same), "here": self.df.at[i, b]})
        similar = []
        if best:
            a, rows = best
            others = [r for r in rows if r != i][:n_similar]
            show = [a, col] + [c for c in self.cols if c not in (a, col)][:3]
            for r in others:
                similar.append({c: self.df.at[r, c] for c in show})
        return {"value": val, "profile": prof, "determined_by": determined_by, "determines": determines,
                "similar": similar, "shape": shape(val), "occurs": int(prof["counts"].get(val, 0)),
                "max_chars": max_chars}

    # ---- prompt -----------------------------------------------------------------------------
    def prompt(self, i: int, col: str, blocks: Sequence[str] = ALL_BLOCKS, max_chars: int = 40,
               n_similar: int = 3, top_values: bool = True) -> str:
        f = self.facts(i, col, max_chars, n_similar)
        sh = lambda v, m=max_chars: self.short(v, m)  # noqa: E731
        prof, val = f["profile"], f["value"]
        row = "\n".join(
            f"- {c}: {sh(self.df.at[i, c])!r}" + ("   <-- SUSPECT" if c == col else "") for c in self.cols
        )
        text = TASK + f"Table: {self.name} | Column: {col} | Cell value: {sh(val, 80)!r}\n\nRow:\n{row}\n\n"
        if "profile" in blocks:
            lines = [f"type: {'numeric' if prof['numeric'] else 'text/categorical'}, {prof['distinct']} distinct values, "
                     f"{prof['missing'] * 100:.1f}% missing"]
            if top_values:
                top = ", ".join(f"{sh(k, 24)!r} ({v})" for k, v in prof["counts"].head(5).items())
                lines.append(f"most frequent values: {top}")
            if prof["shapes"]:
                lines.append("common patterns: " + ", ".join(
                    f"{self.short(s, 40)} ({c / max(1, prof['shape_total']) * 100:.0f}%)" for s, c in prof["shapes"]))
            if prof["numeric"]:
                lines.append(f"median {prof['median']:g}, 1st-99th percentile {prof['p1']:g} to {prof['p99']:g}, "
                             f"min {prof['min']:g}, max {prof['max']:g}")
            lines.append(f"this value's pattern: {self.short(f['shape'], 60)}; occurs {f['occurs']} time(s) in the column")
            text += "Column profile:\n" + "\n".join("- " + x for x in lines) + "\n\n"
        if "dependencies" in blocks:
            ev = [f"{d['col']}={sh(d['key'])!r} -> {col} is {sh(d['majority'])!r} in {d['k']} of {d['n']} rows "
                  f"(this row has {sh(val)!r})" for d in f["determined_by"]]
            ev += [f"{col}={sh(val)!r} -> {d['col']} is {sh(d['majority'])!r} in {d['k']} of {d['n']} rows "
                   f"(this row has {sh(d['here'])!r})" for d in f["determines"]]
            text += "Dependency evidence:\n" + ("\n".join("- " + x for x in ev) if ev else "- no strong dependency found") + "\n\n"
        if "similar" in blocks:
            sim = ["  " + "; ".join(f"{c}={sh(v, 24)!r}" for c, v in r.items()) for r in f["similar"]]
            text += "Similar rows (same value of the strongest determinant):\n" + ("\n".join(sim) if sim else "  none") + "\n\n"
        return text + ANSWER_FORMAT

    # ---- training answer ----------------------------------------------------------------------
    def evidence_sentence(self, i: int, col: str, is_error: bool, error_type: str) -> str:
        """A short, template-built justification. It states only facts that are in the prompt."""
        f = self.facts(i, col)
        val, prof = f["value"], f["profile"]
        dep = f["determined_by"][0] if f["determined_by"] else None
        if is_error and error_type == "missing":
            return f"The value {self.short(val, 24)!r} is empty or a placeholder."
        if dep and dep["majority"] != val and is_error:
            return (f"Rows with {dep['col']}={self.short(dep['key'], 24)!r} have {self.short(dep['majority'], 24)!r} "
                    f"in {dep['k']} of {dep['n']} rows, but this row has {self.short(val, 24)!r}.")
        if is_error:
            share = prof["shape_counts"].get(f["shape"], 0) / max(1, prof["shape_total"]) * 100
            return (f"The value {self.short(val, 24)!r} occurs {f['occurs']} time(s) and its pattern {self.short(f['shape'], 40)} "
                    f"covers {share:.0f}% of the column.")
        if dep and dep["majority"] == val:
            return (f"Matches the usual value for {dep['col']}={self.short(dep['key'], 24)!r} "
                    f"({dep['k']} of {dep['n']} rows).")
        return f"The value {self.short(val, 24)!r} fits the common values and patterns of the column."
