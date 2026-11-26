#!/usr/bin/env python3
"""Rewrite saved prompts so that an empty cell gets the M5 blank-context lines (see TableContext.blank_lines).

Author: Sidhaarth Shree

The lines are built from the prompt itself: the column's whole value set when it has at most 5 distinct values (read from
the "most frequent values" line) and the filled share of the row (read from the "Row:" block). Only prompts whose cell
is empty change. Used to bring the M4 training and validation examples to the new prompt without the original tables.

Usage: python scripts/add_blank_context.py <in.jsonl> <out.jsonl>
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.profiling.context import MISSING  # noqa: E402

CELL = re.compile(r"\| Cell value: (.*)\n")
DISTINCT = re.compile(r"type: [^,]+, (\d+) distinct values")
TOP = re.compile(r"- most frequent values: (.*)\n")
ITEM = re.compile(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\") \((\d+)\)")


def short(v: str, m: int = 24) -> str:
    return v if len(v) <= m else v[: m - 1] + "…"


def is_missing(v: str) -> bool:
    return v.strip().lower() in MISSING


def rewrite(prompt: str) -> str:
    m = CELL.search(prompt)
    if not m:
        return prompt
    try:
        if not is_missing(ast.literal_eval(m.group(1))):
            return prompt
    except (ValueError, SyntaxError):
        return prompt
    lines = []
    d, top = DISTINCT.search(prompt), TOP.search(prompt)
    if d and top and int(d.group(1)) <= 5:
        filled = [ast.literal_eval(k) for k, _ in ITEM.findall(top.group(1))]
        filled = [k for k in filled if not is_missing(k)]
        if filled:
            lines.append(f"the {len(filled)} non-empty values of this column are all of: "
                         + ", ".join(repr(short(k)) for k in filled) + "; an empty cell is not one of them")
    row = prompt.split("Row:\n", 1)[1].split("\n\n", 1)[0].split("\n")
    others = [r for r in row if "<-- SUSPECT" not in r]
    n_filled = 0
    for r in others:
        val = r.split(": ", 1)[1]
        n_filled += not is_missing(ast.literal_eval(val))
    lines.append(f"in this row {n_filled} of the other {len(others)} cells are filled")
    marker = "- this value's pattern:"
    i = prompt.index(marker)
    j = prompt.index("\n", i) + 1
    return prompt[:j] + "".join("- " + x + "\n" for x in lines) + prompt[j:]


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    n = changed = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            new = rewrite(r["prompt"])
            changed += new != r["prompt"]
            r["prompt"] = new
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} examples, {changed} prompts rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
