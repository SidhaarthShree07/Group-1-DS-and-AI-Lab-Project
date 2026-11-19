#!/usr/bin/env python3
"""Re-score Milestone 3 pipeline run with score_decisions (Milestone 4 Task 5).

Author: Sahilpreet

Answers the 140 against 128 repairs question of the M3 review:
- Calls score_decisions on the real m4_e1_base run directory.
- The gate column (gate_decision field) gives the pre-validation gate output.
- Real M3 gate numbers:
    Beers:    11 right / 15 wrong / 5 no-change
    Flights:  10 right /  1 wrong / 1 no-change
    Hospital: 41 right / 41 wrong / 3 no-change
    Total:    62 right / 57 wrong / 9 no-change (128 gate proposals)
- With validation:
    Total:    59 right /  1 wrong / 236 to review
- Outputs docs/tables/m3_rescored_gate.csv with the full 7-bin outcome table.

Usage:
    python scripts/rescore_m3_gate.py [--run-dir m4_e1_base] [--out docs/tables/m3_rescored_gate.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from src.evaluation.metrics import OUTCOME_COLUMNS, score_decisions


def rescore_m3_gate(
    run_dir: Path,
    out_path: Path,
) -> Dict[str, Any]:
    """Execute score_decisions on real m4_e1_base run files."""
    cells_file = run_dir / "cells.csv"
    preds_file = run_dir / "predictions.jsonl"

    if not cells_file.exists():
        raise FileNotFoundError(f"cells.csv not found in {run_dir}")
    if not preds_file.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")

    cells_df = pd.read_csv(cells_file, dtype=str, keep_default_na=False)
    preds = [json.loads(line) for line in preds_file.read_text().splitlines() if line.strip()]

    res = score_decisions(preds, cells_df)
    tot = res["total"]

    print(f"[*] Rescored M3 Baseline ({run_dir.name}):")
    print(f"  Gate Only : {tot.get('gate_repaired_right')} right / "
          f"{tot.get('gate_repaired_wrong')} wrong / "
          f"{tot.get('gate_repaired_no_change')} no-change")
    w = tot.get("repaired_wrong_error", 0) + tot.get("repaired_wrong_correct_cell", 0)
    rev = tot.get("review_error", 0) + tot.get("review_correct_cell", 0)
    print(f"  Post-Valid: {tot.get('repaired_right')} right / {w} wrong / {rev} review")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w_csv = csv.DictWriter(f, fieldnames=OUTCOME_COLUMNS)
        w_csv.writeheader()
        w_csv.writerows(res["outcome_table"])
    print(f"[SUCCESS] Wrote real M3 rescored outcome table -> {out_path}")

    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_REPO_ROOT / "m4_e1_base",
        help="Path to real m4_e1_base run directory containing cells.csv and predictions.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "docs" / "tables" / "m3_rescored_gate.csv",
        help="Output CSV path for rescored gate outcome table.",
    )
    args = parser.parse_args()

    rescore_m3_gate(args.run_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
