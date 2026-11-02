#!/usr/bin/env python3
"""Audit the error TYPES that were assigned by a rule (heuristic) and draw a sample for hand review.

Author: Sidhaarth Shree

Whether a cell is an error never depends on a heuristic: it is the Raha comparison of the dirty and the clean value.
Only the error TYPE of the external tables (Tax, Inpatient, Facilities) comes from the rule in
src/data/load_external.py::classify:  placeholder -> missing;  same number -> format artifact;  rule: the dirty value is
another valid value of the column, or the clean value of another column of the same row (a swap), or a well-formed
value that is a different value;  anything else -> pattern.  That rule can still be wrong: a "rule" value need not break
any dependency of the table, and a "pattern" value need not be a typo.

This script checks every such cell against evidence the heuristic did not use and sorts it into SUPPORTED or UNCERTAIN:
  missing   always supported (a placeholder replaced a real value).
  pattern   supported when the dirty value is a small edit of the clean value (similarity >= 0.6, a typo or a dropped,
            added or changed character); uncertain when it is a different string (a replacement or a misplaced value).
  rule      supported when (a) a dependency found in the dirty table of the same split (strength >= 0.90, the same scan
            as the prompts) says that rows with the same determinant value have the CLEAN value as majority and this
            row has another one, or (b) the cell is half of an exact swap with another column of the row (this cell's
            dirty value is the other cell's clean value and the other way round); uncertain otherwise (the value is
            merely valid elsewhere, or well formed but different).
Output: docs/tables/m2_label_audit_summary.csv (counts per table, split and type) and
        docs/tables/m2_label_audit_sample.csv (a seeded sample per table and type, with a verdict and a comment for each
        of two reviewers: correct, wrong_type or not_an_error) and m2_label_audit_holdout.csv (a second sample of other
        cells, for measuring the error rate after the rule was corrected) and m2_label_audit_random.csv (a simple random
        sample of pattern and rule cells, from which the share of wrong types in the whole group can be read). A sample is drawn once and then frozen: later
        runs keep the same cells and the verdicts and only refresh the current type and audit status; `reviewed_type`
        is the type the reviewers saw, so a verdict stays tied to what was judged.

Usage: python scripts/audit_labels.py [--root .] [--per-group 10] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.build_examples import TYPE_MAP, load_split  # noqa: E402
from src.profiling.context import TableContext, norm  # noqa: E402

TABLES = ["tax", "inpatient", "facilities"]  # the tables whose types come from the rule
SPLITS = ["train", "val", "test"]
SIM_MIN = 0.6


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def swap_partner(cells, r: int, c: int, n_cols: int):
    """Column of the same row whose cell is the other half of an exact swap with cell (r, c), or None."""
    a, b = norm(cells.at[r * n_cols + c, "clean_value"]), norm(cells.at[r * n_cols + c, "dirty_value"])
    for k in range(n_cols):
        if k != c and norm(cells.at[r * n_cols + k, "dirty_value"]) == a and norm(cells.at[r * n_cols + k, "clean_value"]) == b:
            return k
    return None


def judge(ctx: TableContext, r: int, col: str, etype: str, clean: str, dirty: str, swap_col: str = None):
    """Return (status, reason) for one error cell."""
    if etype == "missing":
        return "supported", "placeholder replaced a real value"
    if etype == "pattern":
        s = similarity(clean, dirty)
        if s >= SIM_MIN:
            return "supported", f"small edit of the clean value (similarity {s:.2f})"
        return "uncertain", f"different string (similarity {s:.2f}): replacement or misplaced value, not a typo"
    if etype == "rule":
        for d in ctx.facts(r, col)["determined_by"]:
            if d["majority"] == clean and dirty != clean:
                return "supported", f"{d['col']}={d['key']!r} usually has {clean!r} ({d['k']} of {d['n']} rows)"
        if swap_col:
            return "supported", f"exact swap with column {swap_col} of the same row"
        return "uncertain", "no dependency backs it: the value is only valid elsewhere or well formed but different"
    return "uncertain", "type not covered"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", nargs="*", default=["data/splits/external"])
    ap.add_argument("--tables-dir", default="docs/tables")
    ap.add_argument("--per-group", type=int, default=10, help="sampled cells per table and type for hand review")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-size", type=int, default=150, help="size of the simple random sample of pattern and rule cells")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    rows, pool = [], {}
    for table in TABLES:
        for split in SPLITS:
            dirty, cells = load_split(root, args.splits, table, split)
            ctx = TableContext(table, dirty)
            n_cols = len(ctx.cols)
            counts = {}
            for i in cells.index[cells["is_error"]]:
                etype = TYPE_MAP.get(cells.at[i, "error_type"], cells.at[i, "error_type"])
                if cells.at[i, "error_type"] == "format_artifact" or etype not in ("missing", "pattern", "rule"):
                    continue
                r, c = divmod(int(i), n_cols)
                col = ctx.cols[c]
                clean, dirty_v = cells.at[i, "clean_value"], cells.at[i, "dirty_value"]
                partner = swap_partner(cells, r, c, n_cols) if etype == "rule" else None
                status, why = judge(ctx, r, col, etype, clean, dirty_v, ctx.cols[partner] if partner is not None else None)
                counts[(etype, status)] = counts.get((etype, status), 0) + 1
                pool.setdefault((table, etype), []).append({
                    "table": table, "split": split, "row": r, "column": col, "dirty_value": dirty_v, "clean_value": clean,
                    "assigned_type": etype, "audit_status": status, "audit_reason": why})
            for etype in ("missing", "pattern", "rule"):
                ok, unc = counts.get((etype, "supported"), 0), counts.get((etype, "uncertain"), 0)
                rows.append({"table": table, "split": split, "assigned_type": etype, "cells": ok + unc, "supported": ok,
                             "uncertain": unc, "uncertain_share": f"{unc / (ok + unc):.1%}" if ok + unc else ""})
    out = root / args.tables_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "m2_label_audit_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_key = {(x["table"], x["split"], str(x["row"]), x["column"]): x for items in pool.values() for x in items}
    review_cols = ["reviewer_1_verdict", "reviewer_1_comment", "reviewer_2_verdict", "reviewer_2_comment"]
    fields = ["table", "split", "row", "column", "dirty_value", "clean_value", "assigned_type", "reviewed_type", "audit_status",
              "audit_reason"] + review_cols

    def draw(seed: int, exclude: set, size: int = 0):
        """Seeded sample. By default per table and type, half from the uncertain cells (where mistakes are expected);
        with `size` a simple random sample of that many pattern and rule cells (no stratification), so that its error
        rate can be read as the rate of the whole group."""
        rng, sample = random.Random(seed), []
        if size:
            every = sorted((x for k, items in pool.items() if k[1] != "missing" for x in items
                            if (x["table"], x["split"], str(x["row"]), x["column"]) not in exclude),
                           key=lambda x: (x["table"], x["split"], x["row"], x["column"]))
            return [dict(x, reviewed_type=x["assigned_type"], **{c: "" for c in review_cols}) for x in rng.sample(every, size)]
        for key in sorted(pool):
            items = [x for x in pool[key] if (x["table"], x["split"], str(x["row"]), x["column"]) not in exclude]
            unc = [x for x in items if x["audit_status"] == "uncertain"]
            sup = [x for x in items if x["audit_status"] == "supported"]
            take_u = rng.sample(unc, min(len(unc), args.per_group // 2)) if unc else []
            take_s = rng.sample(sup, min(len(sup), args.per_group - len(take_u))) if sup else []
            sample += take_u + take_s
        return [dict(x, reviewed_type=x["assigned_type"], **{c: "" for c in review_cols}) for x in sample]

    def sample_file(path: Path, seed: int, exclude: set, size: int = 0):
        """The sample is drawn once and then frozen: a later run keeps the same cells and the reviewers' verdicts, and only
        refreshes the current type and audit status. `reviewed_type` is the type the reviewers saw."""
        if path.exists():
            with open(path, encoding="utf-8", newline="") as f:
                old = list(csv.DictReader(f))
            sample, dropped = [], 0
            for o in old:
                cur = by_key.get((o["table"], o["split"], o["row"], o["column"]))
                if cur is None:
                    dropped += 1
                    continue
                sample.append(dict(cur, reviewed_type=o.get("reviewed_type") or o["assigned_type"], **{c: o.get(c, "") for c in review_cols}))
            note = f"kept from {path.name}" + (f", {dropped} cells no longer errors" if dropped else "")
        else:
            sample, note = draw(seed, exclude, size), f"drawn (seed {seed})"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sample)
        changed = sum(x["assigned_type"] != x["reviewed_type"] for x in sample)
        print(f"{path.name}: {len(sample)} cells {note}; {sum(bool(x['reviewer_1_verdict']) for x in sample)} with a verdict; "
              f"{changed} whose type changed since the review")
        return {(x["table"], x["split"], str(x["row"]), x["column"]) for x in sample}

    first = sample_file(out / "m2_label_audit_sample.csv", args.seed, set())
    second = sample_file(out / "m2_label_audit_holdout.csv", args.seed + 1, first)  # a second sample of cells nobody has seen
    sample_file(out / "m2_label_audit_random.csv", args.seed + 2, first | second, args.random_size)  # simple random sample
    print(f"{'table':11s} {'type':8s} {'cells':>6s} {'supported':>9s} {'uncertain':>9s}")
    for t in TABLES:
        for e in ("missing", "pattern", "rule"):
            sub = [r for r in rows if r["table"] == t and r["assigned_type"] == e]
            n, u = sum(r["cells"] for r in sub), sum(r["uncertain"] for r in sub)
            print(f"{t:11s} {e:8s} {n:6d} {n - u:9d} {u:9d} ({u / n:.1%})" if n else f"{t:11s} {e:8s} {0:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
