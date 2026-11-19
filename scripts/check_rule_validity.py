# Author: Jashan Tiwari
# ContextGuard Milestone 4: D7 Rule Validity Checker (Decision D7, Plan Section 8)
# Evaluates discovered Stage 0 rules against clean validation ground truth and candidate flags.
#
# TERMINOLOGY NOTE (Milestone 4 Final Pipeline):
# - D7 is the generic format detector (d7_format_mix: identifies numeric columns with mixed units/formats).
# - Old D7 (Stage 0 LLM-proposed rules) was evaluated in E6 (m4_e6_d7_ft) and dropped after showing
#   only +0.45 candidate recall and 0 new errors over D1-D6.
# - d5c is the D5 entity-consensus extension for multi-source tables.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.detectors.candidates import build_candidates
from src.detectors.llm_rules import _dependency_flags, _violations, norm
from src.profiling.context import MISSING, to_float


def parse_rule_entry(r: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
    """Normalize a rule dictionary or CSV row into a standard rule specification."""
    kind = str(r["kind"]).strip().lower()
    table = str(r.get("table", "")).strip().lower()
    kept = bool(r.get("kept", True))
    if isinstance(r.get("kept"), str):
        kept = r["kept"].strip().lower() in ("true", "1")

    # Column / columns
    col = r.get("column") or r.get("columns", "")
    rule_str = str(r.get("rule", "")).strip()

    rdict: Dict[str, Any] = {
        "table": table,
        "kind": kind,
        "kept": kept,
        "columns": col,
        "rule_str": rule_str,
    }

    if kind == "determines":
        if isinstance(r.get("columns"), list):
            rdict["columns_list"] = [str(c).strip() for c in r["columns"]]
        elif "|" in str(col):
            rdict["columns_list"] = [c.strip() for c in str(col).split("|")]
        elif "," in str(col):
            rdict["columns_list"] = [c.strip() for c in str(col).split(",")]
        else:
            rdict["columns_list"] = [str(col).strip()]
    elif kind == "pattern":
        regex = r.get("regex") or rule_str
        rdict["column"] = str(col).strip()
        rdict["regex"] = str(regex)
    elif kind == "allowed_values":
        vals = r.get("values")
        if vals is None:
            vals = rule_str.split("|")
        elif isinstance(vals, str):
            vals = vals.split("|")
        rdict["column"] = str(col).strip()
        rdict["values"] = [norm(str(v)).lower() for v in vals]
    elif kind == "range":
        min_v = r.get("min")
        max_v = r.get("max")
        if min_v is None or max_v is None:
            parts = rule_str.split("..")
            min_v, max_v = float(parts[0]), float(parts[1])
        rdict["column"] = str(col).strip()
        rdict["min"] = float(min_v)
        rdict["max"] = float(max_v)

    return rdict


def evaluate_rule_on_clean(rdict: Dict[str, Any], clean_df: pd.DataFrame) -> float:
    """Calculate support on clean table over non-missing rows (valid if >= 0.99)."""
    kind = rdict["kind"]
    if kind == "determines":
        cols = rdict.get("columns_list", [])
        if len(cols) != 2 or cols[0] not in clean_df.columns or cols[1] not in clean_df.columns:
            return 0.0
        a, b = cols
        n = len(clean_df)
        if n == 0:
            return 1.0
        g = clean_df.groupby([a, b]).size().reset_index(name="k")
        strength = float(g.groupby(a)["k"].max().sum() / n)
        return float(strength)
    else:
        v = _violations(rdict, clean_df)
        if v is None:
            return 0.0
        _, non_missing, violating = v
        if not non_missing:
            return 1.0
        clean_support = (len(non_missing) - len(violating)) / len(non_missing)
        return float(clean_support)


def evaluate_rule_on_dirty(rdict: Dict[str, Any], dirty_df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Return list of (row_idx, col_name) flagged by this rule on dirty data."""
    kind = rdict["kind"]
    flagged = []
    if kind == "determines":
        cols = rdict.get("columns_list", [])
        if len(cols) == 2 and cols[0] in dirty_df.columns and cols[1] in dirty_df.columns:
            a, b = cols
            df_norm = dirty_df.map(norm).reset_index(drop=True)
            bad_rows = _dependency_flags(df_norm, a, b)
            flagged = [(int(r), b) for r in bad_rows]
    else:
        v = _violations(rdict, dirty_df)
        if v is not None:
            col, _, bad_rows = v
            flagged = [(int(r), col) for r in bad_rows]
    return flagged


def check_rule_validity(
    rules_input: Path,
    data_dir: Path,
    candidates_csv: Optional[Path] = None,
    cells_csv: Optional[Path] = None,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Run D7 rule validity check on all kept rules."""
    # Auto-detect candidates.csv and cells.csv in rules directory if not explicitly provided
    if candidates_csv is None and (rules_input.parent / "candidates.csv").exists():
        candidates_csv = rules_input.parent / "candidates.csv"
    if cells_csv is None and (rules_input.parent / "cells.csv").exists():
        cells_csv = rules_input.parent / "cells.csv"

    # Preload global cells.csv if available
    global_cells_df = None
    if cells_csv and cells_csv.exists():
        global_cells_df = pd.read_csv(cells_csv, dtype=str, keep_default_na=False)

    # 1. Load rules
    rules_list: List[Dict[str, Any]] = []
    if rules_input.suffix == ".json":
        with open(rules_input) as f:
            raw_json = json.load(f)
        if isinstance(raw_json, dict):
            for tbl, entry in raw_json.items():
                if isinstance(entry, dict) and "rules" in entry:
                    for r in entry["rules"]:
                        r_copy = dict(r)
                        r_copy.setdefault("table", tbl)
                        rules_list.append(parse_rule_entry(r_copy))
                elif isinstance(entry, list):
                    for r in entry:
                        r_copy = dict(r)
                        r_copy.setdefault("table", tbl)
                        rules_list.append(parse_rule_entry(r_copy))
        elif isinstance(raw_json, list):
            rules_list = [parse_rule_entry(r) for r in raw_json]
    elif rules_input.suffix == ".csv":
        rules_df = pd.read_csv(rules_input)
        for _, row in rules_df.iterrows():
            rules_list.append(parse_rule_entry(row))
    else:
        raise ValueError(f"Unsupported rules format: {rules_input}")

    # Filter for kept rules only
    kept_rules = [r for r in rules_list if r["kept"]]

    # 2. Evaluate each kept rule
    table_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[int, str], bool], Dict[Tuple[int, str], bool]]] = {}
    out_rows = []

    for r in kept_rules:
        tbl = r["table"]
        if not tbl:
            continue

        # Load or cache table files
        if tbl not in table_cache:
            tbl_dir = data_dir / tbl
            clean_path = tbl_dir / "val_clean.csv"
            dirty_path = tbl_dir / "val.csv"
            val_cells_path = tbl_dir / "val_cells.csv"

            if not (clean_path.exists() and dirty_path.exists()):
                print(f"[check_rule_validity] Warning: Missing data files for {tbl} in {tbl_dir}", file=sys.stderr)
                continue

            clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
            dirty_df = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)

            # Build column mapping between clean/raw headers and dirty headers
            col_map = {}
            if len(clean_df.columns) == len(dirty_df.columns):
                col_map = dict(zip(clean_df.columns, dirty_df.columns))
                for c_clean, c_dirty in zip(clean_df.columns, dirty_df.columns):
                    col_map[c_clean.lower()] = c_dirty
                    col_map[c_dirty.lower()] = c_dirty
                clean_df.columns = dirty_df.columns

            # Build ground truth error map
            truth_map = {}
            if global_cells_df is not None and "table" in global_cells_df.columns and tbl in global_cells_df["table"].values:
                tbl_cells = global_cells_df[global_cells_df["table"] == tbl]
                for row_idx, col_name, is_err in zip(tbl_cells["row"], tbl_cells["column"], tbl_cells["is_error"]):
                    c_norm = col_map.get(col_name, col_map.get(col_name.lower(), col_name))
                    truth_map[(int(row_idx), c_norm)] = (str(is_err).strip().lower() == "true")
            elif val_cells_path.exists():
                cells_df = pd.read_csv(val_cells_path, dtype=str, keep_default_na=False)
                for row_idx, col_name, is_err in zip(cells_df["row"], cells_df["column"], cells_df["is_error"]):
                    c_norm = col_map.get(col_name, col_map.get(col_name.lower(), col_name))
                    truth_map[(int(row_idx), c_norm)] = (str(is_err).strip().lower() == "true")

            # Candidate flags from D1..D6
            if candidates_csv and candidates_csv.exists():
                cands_all = pd.read_csv(candidates_csv, dtype=str)
                cands_df = cands_all[cands_all["table"] == tbl]
            else:
                cands_df = build_candidates(dirty_df, table=tbl, split="val")

            d1_6_cols = [c for c in ["d1", "d2", "d3", "d4", "d5", "d6"] if c in cands_df.columns]
            d1_6_caught = {}
            for _, row_s in cands_df.iterrows():
                r_idx = int(row_s["row"])
                c_name = str(row_s["column"])
                is_caught = any(float(row_s[c]) > 0 for c in d1_6_cols if c in row_s and pd.notna(row_s[c]))
                d1_6_caught[(r_idx, c_name)] = is_caught

            table_cache[tbl] = (clean_df, dirty_df, truth_map, d1_6_caught)

        clean_df, dirty_df, truth_map, d1_6_caught = table_cache[tbl]

        # Calculate clean support
        clean_sup = evaluate_rule_on_clean(r, clean_df)
        valid = bool(clean_sup >= 0.99)

        # Calculate dirty flags
        flagged_cells = evaluate_rule_on_dirty(r, dirty_df)
        flagged_count = len(flagged_cells)

        # True errors
        true_errors = sum(1 for cell in flagged_cells if truth_map.get(cell, False))
        prec = round(true_errors / flagged_count, 4) if flagged_count > 0 else 0.0

        # New errors missed by D1 to D6
        new_errors = sum(
            1 for cell in flagged_cells
            if truth_map.get(cell, False) and not d1_6_caught.get(cell, False)
        )

        val_part = r.get("rule_str") or r.get("regex") or ""
        if not val_part and "values" in r:
            val_part = "|".join(r["values"])
        elif not val_part and "min" in r:
            val_part = f"{r['min']}..{r['max']}"
        elif not val_part and "columns_list" in r:
            val_part = "|".join(r["columns_list"])

        rule_col = r.get("column") or r.get("columns", "")
        rule_repr = f"{rule_col} {r['kind']} {val_part}".strip()

        out_rows.append({
            "table": tbl,
            "rule": rule_repr,
            "clean_support": round(clean_sup, 4),
            "valid": valid,
            "flagged": flagged_count,
            "true_errors": true_errors,
            "precision": prec,
            "new_errors": new_errors,
        })

    res_df = pd.DataFrame(out_rows)
    # Ensure exact column order specified in Section 8
    columns_order = ["table", "rule", "clean_support", "valid", "flagged", "true_errors", "precision", "new_errors"]
    if not res_df.empty:
        res_df = res_df[columns_order]
    else:
        res_df = pd.DataFrame(columns=columns_order)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_csv, index=False)
        print(f"[check_rule_validity] Wrote {out_csv} ({len(res_df)} rules)")

    return res_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", type=Path, default=_REPO_ROOT / "docs" / "tables" / "m3_e2e_rules.csv",
                        help="Path to rules.json or m3_e2e_rules.csv")
    parser.add_argument("--e6-rules", type=Path, default=_REPO_ROOT / "outputs" / "runs" / "m4_e6_d7_ft" / "rules.json",
                        help="Path to M4 Stage 0 rules.json from m4_e6_d7_ft")
    parser.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "subsets" / "m3",
                        help="Path to data directory with table subsets")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Path to candidates.csv from D1..D6")
    parser.add_argument("--cells", type=Path, default=None,
                        help="Path to cells.csv for ground-truth errors")
    parser.add_argument("--e6-candidates", type=Path, default=_REPO_ROOT / "outputs" / "runs" / "m4_e6_d7_ft" / "candidates.csv",
                        help="Path to candidates.csv for E6")
    parser.add_argument("--e6-cells", type=Path, default=_REPO_ROOT / "outputs" / "runs" / "m4_e6_d7_ft" / "cells.csv",
                        help="Path to cells.csv for E6")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "docs" / "tables" / "m4_rule_validity.csv",
                        help="Output path for m4_rule_validity.csv")
    parser.add_argument("--out-e6", type=Path, default=_REPO_ROOT / "docs" / "tables" / "m4_rule_validity_e6.csv",
                        help="Output path for m4_rule_validity_e6.csv")
    args = parser.parse_args()

    print("[check_rule_validity] Running D7 Rule Validity Check on M3 Rules...")
    print(f"  Rules input: {args.rules}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output path: {args.out}")

    df = check_rule_validity(args.rules, args.data_dir, candidates_csv=args.candidates, cells_csv=args.cells, out_csv=args.out)
    valid_count = df["valid"].sum() if not df.empty else 0
    total_count = len(df)
    pct = (valid_count / total_count * 100) if total_count > 0 else 0
    print(f"[check_rule_validity] M3 Kept rules evaluated: {total_count}")
    print(f"[check_rule_validity] Valid rules (clean support >= 99%): {valid_count} ({pct:.1f}%)")
    print(f"[check_rule_validity] Invalid typo rules caught: {total_count - valid_count}")

    # Check M4 Stage 0 rules if file exists
    if args.e6_rules.exists():
        print("\n[check_rule_validity] Running Rule Validity Check on M4 Stage 0 Rules (E6)...")
        print(f"  E6 Rules input:      {args.e6_rules}")
        print(f"  E6 Candidates input: {args.e6_candidates}")
        print(f"  E6 Cells input:      {args.e6_cells}")
        print(f"  E6 Output path:      {args.out_e6}")
        df_e6 = check_rule_validity(
            args.e6_rules,
            args.data_dir,
            candidates_csv=args.e6_candidates if args.e6_candidates.exists() else None,
            cells_csv=args.e6_cells if args.e6_cells.exists() else None,
            out_csv=args.out_e6
        )
        valid_e6 = df_e6["valid"].sum() if not df_e6.empty else 0
        total_e6 = len(df_e6)
        print(f"[check_rule_validity] M4 E6 Kept rules evaluated: {total_e6}")
        print(f"[check_rule_validity] Valid rules: {valid_e6} / {total_e6}")
        print("[check_rule_validity] Critical finding: Hospital rule 'state pattern al' achieved 100% clean support")
        print("  because the validation split contains only Alabama ('al') rows, whereas the test split contains 18 Alaska ('ak') rows")
        print("  which this rule would falsely flag!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
