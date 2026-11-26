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
import itertools
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


def differ_only_in_format(a: str, b: str) -> bool:
    """True for two different strings that hold the same numbers, e.g. '12.0 oz.' and '12.0 oz'."""
    numbers_a = re.findall(r"\d+(?:\.\d+)?", a)
    numbers_b = re.findall(r"\d+(?:\.\d+)?", b)
    return bool(numbers_a) and [float(x) for x in numbers_a] == [float(x) for x in numbers_b] and a != b


def edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Levenshtein distance, stopping early once it exceeds `limit` (returns limit + 1)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def close_frequent_value(value: str, occurs: int, counts, top: int = 200):
    """A much more frequent value of the column within two edits of `value`, as (value, count, distance), or None.
    Typos such as 'Camp_Noju' for 'Camp_Nou' look like ordinary names to the model unless it is shown the neighbour.
    Values with digits are skipped: '6:08' and '6:03' are one edit apart but are different times, not a misspelling."""
    if len(value) < 4 or value.lower() in MISSING or any(ch.isdigit() for ch in value):
        return None
    best = None
    for other, n in counts.head(top).items():
        if other == value or n < max(3, 3 * occurs) or len(other) < 4 or any(ch.isdigit() for ch in str(other)):
            continue
        d = edit_distance(value.lower(), str(other).lower())
        if d <= 2 and (best is None or (d, -n) < (best[2], -best[1])):
            best = (other, int(n), d)
    return best


class TableContext:
    """Statistics of one dirty table (a split of it, or an uploaded file)."""

    def __init__(self, name: str, dirty: pd.DataFrame, key_share: float = 0.6, evidence_threshold: float = 0.9,
                 skip_missing_keys: bool = False, weighted_majority: bool = False, close_values: bool = False,
                 blank_context: bool = False):
        """With the defaults the prompts are exactly the ones the model was trained on. The three options were added
        in M4 for inference only:
          evidence_threshold  lowest dependency strength shown as evidence for the entity column (training: 0.9).
                              In Flights the sources disagree, so flight -> time holds for only 56 to 77% of the
                              rows and the flight's other rows were never shown to the model.
          skip_missing_keys   rows that share an empty value are not related, so an empty key never forms a group.
          weighted_majority   the group majority counts each row by its reliability (see row_reliability), and
                              similar rows are listed most reliable first."""
        self.close_values = close_values  # M5: name a frequent value one or two edits away (likely misspelling)
        self.blank_context = blank_context  # M5: extra profile lines, only for a cell that is itself empty
        self.evidence_threshold = evidence_threshold
        self.skip_missing_keys = skip_missing_keys
        self.weighted_majority = weighted_majority
        self.name = name
        self.df = dirty.map(norm).reset_index(drop=True)
        self.cols: List[str] = list(self.df.columns)
        self.n = len(self.df)
        self.key_share = key_share  # a determinant with more distinct values than this share of rows is a key
        self._profile: Dict[str, dict] = {}
        self._index: Dict[str, dict] = {}
        self._fd: Optional[Dict[tuple, float]] = None
        self._entity_key: Optional[str] = None
        self._entity_key_checked = False

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
        determinants = sorted(((s, a) for (a, b), s in fd.items() if b == col and s >= 0.9), reverse=True)
        entity_col = self.entity_key() if self.evidence_threshold < 0.9 else None
        if entity_col and entity_col != col and fd.get((entity_col, col), 0.0) >= self.evidence_threshold:
            # the entity column goes first, even when conflicting sources weaken its dependency
            determinants = [(fd[(entity_col, col)], entity_col)] + [(s, a) for s, a in determinants if a != entity_col]
        if self.skip_missing_keys:
            determinants = [(s, a) for s, a in determinants if self.df.at[i, a].lower() not in MISSING]
        for s, a in determinants[:2]:
            rows = self.rows_with(a, self.df.at[i, a])
            if len(rows) >= 2:
                cnt = collections.Counter(self.df.at[r, col] for r in rows)
                majority, k = cnt.most_common(1)[0]
                if self.weighted_majority:
                    weights = self.row_reliability(rows, exclude=(a, col))
                    votes = collections.defaultdict(float)
                    for r in rows:
                        if self.df.at[r, col].lower() not in MISSING:
                            votes[self.df.at[r, col]] += weights[r]
                    if votes:
                        majority = max(votes, key=votes.get)
                        k = cnt[majority]
                    rows = sorted(rows, key=lambda r: -weights[r])
                weak_entity_evidence = a == entity_col and fd.get((a, col), 0.0) < 0.9
                if weak_entity_evidence and differ_only_in_format(majority, self.df.at[i, col]):
                    continue  # the other rows only write the same number differently: nothing new to show
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

    def entity_key(self) -> Optional[str]:
        """The column naming the real-world object that several rows describe (`flight` in Flights), or None.

        No column names are used. Two columns that are each repeated (2 to 50% distinct values) but together
        identify almost every row form a (source, entity) pair, e.g. (src, flight); this is a unique column
        combination as in HyUCC (Papenbrock & Naumann 2017). Of the columns found in such pairs, the entity is the
        one that determines the other columns best: a flight fixes its times, a source does not. A table with one
        row per object has no such pair."""
        if self._entity_key_checked:
            return self._entity_key
        strength, n = self.dependencies(), self.n
        repeated_cols = [c for c in self.cols
                         if n and 2 <= self.df[c].nunique() <= 0.5 * n
                         and (self.df[c].str.lower().isin(MISSING)).mean() < 0.05]
        pair_members = set()
        for a, b in itertools.combinations(repeated_cols, 2):
            if len(self.df[[a, b]].drop_duplicates()) >= 0.98 * n:
                pair_members.update((a, b))

        def determines_others(col: str) -> float:
            return sum(strength.get((col, other), 0.0) for other in self.cols if other != col)

        self._entity_key = max(sorted(pair_members), key=determines_others) if pair_members else None
        self._entity_key_checked = True
        return self._entity_key

    def row_reliability(self, rows: List[int], exclude=()) -> Dict[int, float]:
        """How much to trust each row of a group: the share of its columns (other than `exclude`) that hold the
        group's most common value, with add-one smoothing. Rows that agree with the group elsewhere count more."""
        cols = [c for c in self.cols if c not in exclude]
        group_majority = {c: collections.Counter(self.df.at[r, c] for r in rows).most_common(1)[0][0] for c in cols}
        return {r: (sum(self.df.at[r, c] == group_majority[c] for c in cols) + 1) / (len(cols) + 2) for r in rows}

    def blank_lines(self, i: int, col: str, prof: dict) -> List[str]:
        """Extra facts about an empty cell, shown only for empty cells (every other prompt is unchanged): the whole
        value set of the column when it has at most 5 distinct values, and how many other cells of the row are filled.
        Both can be read back from the ordinary prompt, so the training prompts can be rewritten by
        scripts/add_blank_context.py without the table."""
        lines = []
        filled = [k for k in prof["counts"].index if k.lower() not in MISSING]
        if filled and prof["distinct"] <= 5:
            lines.append(f"the {len(filled)} non-empty values of this column are all of: "
                         + ", ".join(f"{self.short(k, 24)!r}" for k in filled) + "; an empty cell is not one of them")
        others = [c for c in self.cols if c != col]
        n_filled = sum(self.df.at[i, c].lower() not in MISSING for c in others)
        lines.append(f"in this row {n_filled} of the other {len(others)} cells are filled")
        return lines

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
            if self.blank_context and val.lower() in MISSING:
                lines += self.blank_lines(i, col, prof)
            if self.close_values:
                near = close_frequent_value(val, f["occurs"], prof["counts"])
                if near:
                    lines.append(f"similar spelling: {sh(near[0], 40)!r} occurs {near[1]} time(s) and differs by "
                                 f"{near[2]} character(s)")
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
