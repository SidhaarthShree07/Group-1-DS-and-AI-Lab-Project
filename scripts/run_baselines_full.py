#!/usr/bin/env python3
"""Run baseline error detection and repair methods for ContextGuard Milestone 4.

Author: Aaradhya (Group 1 · DSAI Lab, IIT Madras BS Degree)

Evaluates three baseline systems on the held-out test splits:
- B1: HoloClean (simplified denial-constraint check + modal repair voting)
- B2: dBoost (simplified unsupervised outlier detection with train grid search)
- B3: Raha + Baran (simplified value/vicinity transfer under 20-tuple budget)

All scoring leaves out the 20 labelled tuples recorded in
data/processed/baselines/B3_Raha_Baran/<table_name>/scoring_mask.csv.

Note on Implementation Parity:
As noted in the M4 specification and lead guidance, these runs represent
simplified baseline pipelines (heuristic constraint violation voting, reduced
feature strategies, and value transfer) and are clearly designated as
'simplified' across all reporting tables and documentation.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import random
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm.build_examples import load_split
from src.profiling.context import norm

TABLES = ["hospital", "flights", "beers", "rayyan", "movies"]
SPLITS_DIRS = ["data/splits/core"]
BASELINES_DIR = Path("data/processed/baselines")


RAHA_SEED = 0

def score_flags(
    flagged_cells: Iterable[Union[Tuple[int, str], Dict[str, Any]]],
    cells: Union[Sequence[Dict[str, Any]], str, Path, Any],
    exclude_rows: Optional[Iterable[int]] = None,
    repairs: Optional[Dict[Tuple[int, str], Any]] = None,
) -> Dict[str, Any]:
    """Score baseline flagged cells and repairs according to Sahilpreet's shared contract.

    Follows ContextGuard evaluation standards:
    - Same Raha normalisation: norm(dirty) != norm(clean)
    - Excludes format artifacts (error_type == 'format_artifact')
    - Excludes rows in exclude_rows (the 20 labelled rows, left out for every method)
    - Evaluates detection PRF, repair accuracy, and false repair rate.
    """
    exclude_set = set(exclude_rows) if exclude_rows is not None else set()
    flagged_set: Set[Tuple[int, str]] = set()
    for item in flagged_cells:
        if isinstance(item, tuple):
            flagged_set.add((int(item[0]), str(item[1])))
        elif isinstance(item, dict):
            flagged_set.add((int(item["row"]), str(item["column"])))

    repair_dict: Dict[Tuple[int, str], str] = {}
    if repairs is not None:
        for k, v in repairs.items():
            if isinstance(k, tuple):
                repair_dict[(int(k[0]), str(k[1]))] = str(v)
            elif isinstance(k, dict):
                repair_dict[(int(k["row"]), str(k["column"]))] = str(v)

    tp = fp = tn = fn = 0
    total_true_errors = 0
    successful_repairs = executed_repairs = incorrect_repairs = 0
    format_artifacts_skipped = 0
    scored_cells = 0

    for c in cells:
        if str(c.get("error_type", "")).strip() == "format_artifact":
            format_artifacts_skipped += 1
            continue

        r = int(c["row"])
        col = str(c["column"])
        if r in exclude_set:
            continue

        scored_cells += 1
        d_val = c.get("dirty_value", "")
        c_val = c.get("clean_value", "")
        gold_is_error = norm(d_val) != norm(c_val)
        if gold_is_error:
            total_true_errors += 1

        is_flagged = (r, col) in flagged_set
        if is_flagged:
            if gold_is_error:
                tp += 1
            else:
                fp += 1
            if (r, col) in repair_dict:
                executed_repairs += 1
                rep_val = repair_dict[(r, col)]
                if gold_is_error and norm(rep_val) == norm(c_val):
                    successful_repairs += 1
                else:
                    incorrect_repairs += 1
        else:
            if gold_is_error:
                fn += 1
            else:
                tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    rep_acc = successful_repairs / total_true_errors if total_true_errors > 0 else 0.0
    false_rep_rate = incorrect_repairs / executed_repairs if executed_repairs > 0 else 0.0

    return {
        "n_scored_cells": scored_cells,
        "total_true_errors": total_true_errors,
        "detected_cells": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "executed_repairs": executed_repairs,
        "successful_repairs": successful_repairs,
        "incorrect_repairs": incorrect_repairs,
        "repair_accuracy": round(rep_acc, 4),
        "false_repair_rate": round(false_rep_rate, 4),
        "format_artifacts_skipped": format_artifacts_skipped,
    }


def load_table_data(root: Path, table: str):
    dirty_df, cells_df = load_split(root, SPLITS_DIRS, table, "test")
    cols = list(dirty_df.columns)
    n_cols = len(cols)

    # the 20 labelled rows are left out of scoring
    mask_path = root / BASELINES_DIR / "B3_Raha_Baran" / table / "scoring_mask.csv"
    if mask_path.exists():
        mask_df = pd.read_csv(mask_path)
        valid_rows = set(mask_df[mask_df["include_in_scoring"]]["row_id"].tolist())
    else:
        valid_rows = set(range(len(dirty_df)))
    excluded_rows = set(range(len(dirty_df))) - valid_rows

    cell_records = []
    for idx, r in cells_df.reset_index(drop=True).iterrows():
        cell_records.append({
            "row": idx // n_cols,
            "column": cols[idx % n_cols],
            "dirty_value": r.get("dirty_value", ""),
            "clean_value": r.get("clean_value", ""),
            "is_error": r.get("is_error", False),
            "error_type": r.get("error_type", ""),
        })

    return dirty_df, cols, excluded_rows, cell_records


def run_b1_holoclean_repair(root: Path, table: str, dirty_df: pd.DataFrame, cols: list, excluded_rows: set, cell_records: list):
    t0 = time.time()
    c_file = root / BASELINES_DIR / "B1_HoloClean" / table / "constraints.txt"
    if not c_file.exists() or c_file.stat().st_size == 0:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B1_HoloClean_simplified",
            "table": table,
            "runtime_sec": elapsed,
            "status": "completed (0 constraints)",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "executed_repairs": 0,
            "successful_repairs": 0,
            "repair_accuracy": 0.0,
            "false_repair_rate": 0.0,
            "notes": "No denial constraints mined"
        }

    constraints = [line.strip() for line in c_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    flagged = set()
    repairs = {}

    for c in constraints:
        eq_parts = []
        iq_part = None
        for token in c.split("&"):
            if token.startswith("EQ("):
                col = token[3:-1].split(",")[0].replace("t1.", "")
                eq_parts.append(col)
            elif token.startswith("IQ("):
                iq_part = token[3:-1].split(",")[0].replace("t1.", "")

        if eq_parts and iq_part and iq_part in cols and all(col in cols for col in eq_parts):
            grouped = dirty_df.groupby(eq_parts, dropna=False)
            for _, group in grouped:
                if len(group) > 1 and group[iq_part].nunique(dropna=False) > 1:
                    val_counts = group[iq_part].value_counts(dropna=False)
                    maj_val = str(val_counts.index[0]) if len(val_counts) > 0 else ""
                    # Fix: do NOT repair cells to their own value; only flag and repair deviating cells
                    for r_idx in group.index:
                        curr_val = str(dirty_df.loc[r_idx, iq_part])
                        if curr_val != maj_val:
                            flagged.add((r_idx, iq_part))
                            repairs[(r_idx, iq_part)] = maj_val

    elapsed = round(time.time() - t0, 3)
    res = score_flags(flagged, cell_records, exclude_rows=excluded_rows, repairs=repairs)
    return {
        "baseline": "B1_HoloClean_simplified",
        "table": table,
        "runtime_sec": elapsed,
        "status": "completed",
        "detected_cells": res["detected_cells"],
        "tp": res["tp"],
        "fp": res["fp"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "executed_repairs": res["executed_repairs"],
        "successful_repairs": res["successful_repairs"],
        "repair_accuracy": res["repair_accuracy"],
        "false_repair_rate": res["false_repair_rate"],
        "notes": f"Simplified DC violation check with modal voting across {len(constraints)} DCs; scored via score_flags"
    }


def run_b2_dboost_grid(root: Path, table: str, dirty_df: pd.DataFrame, cols: list, excluded_rows: set, cell_records: list):
    t0 = time.time()
    try:
        # Lazy import of dBoost to prevent dependency crashes
        import raha.tools.dBoost.dboost.imported_dboost as db
    except ImportError as e:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B2_dBoost_simplified",
            "table": table,
            "runtime_sec": elapsed,
            "status": "failed: dBoost library not found",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "executed_repairs": 0,
            "successful_repairs": 0,
            "repair_accuracy": 0.0,
            "false_repair_rate": 0.0,
            "notes": "dBoost dependency not available; reported as failed per specification"
        }

    try:
        # 1. Load train split for hyperparameter tuning
        train_dirty, train_cells = load_split(root, SPLITS_DIRS, table, "train")
        train_cols = list(train_dirty.columns)
        n_t_cols = len(train_cols)
        train_records = []
        for idx, r in train_cells.reset_index(drop=True).iterrows():
            train_records.append({
                "row": idx // n_t_cols,
                "column": train_cols[idx % n_t_cols],
                "dirty_value": r.get("dirty_value", ""),
                "clean_value": r.get("clean_value", ""),
                "is_error": r.get("is_error", False),
                "error_type": r.get("error_type", ""),
            })

        execution_success = False

        def run_db(df, c_list, params_args):
            nonlocal execution_success
            temp_dir = tempfile.mkdtemp()
            temp_csv = os.path.join(temp_dir, f"{table}_dirty.csv")
            df.to_csv(temp_csv, index=False, encoding="utf-8")
            temp_out = temp_csv + "-dboost_output.csv"
            cmd = ["-F", ","] + params_args + [temp_csv]
            cmd_error = None
            try:
                db.run(cmd)
                execution_success = True
            except Exception as ex:
                cmd_error = str(ex)

            flagged = set()
            if os.path.exists(temp_out):
                try:
                    ocdf = pd.read_csv(temp_out, sep=",", header=None, dtype=str, encoding="utf-8", on_bad_lines="skip")
                    for val in ocdf.values:
                        try:
                            r_idx, c_idx = int(val[0]), int(val[1])
                            if r_idx > 0 and (r_idx - 1) < len(df) and c_idx < len(c_list):
                                flagged.add((r_idx - 1, c_list[c_idx]))
                        except (ValueError, IndexError):
                            continue
                    os.remove(temp_out)
                except Exception:
                    pass
            try:
                os.remove(temp_csv)
                os.rmdir(temp_dir)
            except Exception:
                pass

            if cmd_error and not flagged:
                raise RuntimeError(cmd_error)
            return flagged

        grid = [
            ["--statistical", "0.2", "--histogram", "0.2", "0.2"],
            ["--statistical", "0.5", "--histogram", "0.5", "0.5"],
            ["--statistical", "0.8", "--histogram", "0.8", "0.8"],
            ["--statistical", "0.5", "--gaussian", "3.0"],
        ]

        best_cfg = grid[0]
        best_train_f1 = -1.0
        for cfg in grid:
            try:
                fl = run_db(train_dirty, train_cols, cfg)
                s = score_flags(fl, train_records)
                if s["f1"] > best_train_f1:
                    best_train_f1 = s["f1"]
                    best_cfg = cfg
            except Exception:
                continue

        if not execution_success:
            elapsed = round(time.time() - t0, 3)
            return {
                "baseline": "B2_dBoost_simplified",
                "table": table,
                "runtime_sec": elapsed,
                "status": "failed: dBoost tool crashed during execution",
                "detected_cells": 0,
                "tp": 0,
                "fp": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "executed_repairs": 0,
                "successful_repairs": 0,
                "repair_accuracy": 0.0,
                "false_repair_rate": 0.0,
                "notes": "dBoost failed to execute across grid parameters"
            }

        test_flagged = run_db(dirty_df, cols, best_cfg)
        elapsed = round(time.time() - t0, 3)
        res = score_flags(test_flagged, cell_records, exclude_rows=excluded_rows)
        cfg_str = " ".join(best_cfg)
        return {
            "baseline": "B2_dBoost_simplified",
            "table": table,
            "runtime_sec": elapsed,
            "status": "completed",
            "detected_cells": res["detected_cells"],
            "tp": res["tp"],
            "fp": res["fp"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "executed_repairs": 0,
            "successful_repairs": 0,
            "repair_accuracy": 0.0,
            "false_repair_rate": 0.0,
            "notes": f"Simplified dBoost best config tuned on train split ({cfg_str}); scored via score_flags"
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B2_dBoost_simplified",
            "table": table,
            "runtime_sec": elapsed,
            "status": f"failed: {str(e)[:40]}",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "executed_repairs": 0,
            "successful_repairs": 0,
            "repair_accuracy": 0.0,
            "false_repair_rate": 0.0,
            "notes": f"Execution error: {str(e)}"
        }


def run_b3_raha_baran(root: Path, table: str, dirty_df: pd.DataFrame, cols: list, excluded_rows: set, cell_records: list):
    t0 = time.time()
    try:
        # Lazy import of raha
        import raha
        dirty_path = str((root / BASELINES_DIR / "B3_Raha_Baran" / table / "dirty.csv").resolve())
        clean_path = str((root / BASELINES_DIR / "B3_Raha_Baran" / table / "clean.csv").resolve())
        lbl_path = root / BASELINES_DIR / "B3_Raha_Baran" / table / "labelled_tuples_20.json"

        lbl_indices = []
        if lbl_path.exists():
            lbl_indices = json.loads(lbl_path.read_text(encoding="utf-8")).get("labelled_row_indices", [])

        # Ensure labelled row indices strictly match holdout rows in scoring_mask
        assert set(lbl_indices) == excluded_rows, "Raha 20 labelled tuples must match scoring mask holdout!"

        app = raha.Detection()
        app.LABELING_BUDGET = 20
        app.VERBOSE = False
        app.SAVE_RESULTS = False

        # Raha labels exactly our 20 rows (the ones left out of scoring for every method), in a fixed order, and
        # its random choices are seeded, so the baseline is the same on every run
        fixed_rows = sorted(lbl_indices)
        raha_sample_tuple = raha.detection.Detection.sample_tuple

        def sample_our_rows(det_self, d):
            raha_sample_tuple(det_self, d)  # keeps Raha's cluster bookkeeping
            d.sampled_tuple = fixed_rows[len(d.labeled_tuples)]

        app.sample_tuple = types.MethodType(sample_our_rows, app)
        random.seed(RAHA_SEED)
        np.random.seed(RAHA_SEED)

        # Simplified strategy runner for fast execution across environments
        def safe_runner(d, algorithm, configuration):
            out_cells = {}
            if algorithm == "PVD":
                attr, ch = configuration
                if attr in d.dataframe.columns:
                    j = d.dataframe.columns.get_loc(attr)
                    for i, val in d.dataframe[attr].items():
                        try:
                            if ch in str(val):
                                out_cells[(i, j)] = ""
                        except Exception:
                            continue
            elif algorithm == "RVD":
                l_attr, r_attr = configuration
                if l_attr in d.dataframe.columns and r_attr in d.dataframe.columns:
                    l_j = d.dataframe.columns.get_loc(l_attr)
                    r_j = d.dataframe.columns.get_loc(r_attr)
                    val_map = {}
                    for i, r_val in d.dataframe.iterrows():
                        lv = r_val[l_attr]
                        rv = r_val[r_attr]
                        if lv:
                            if lv not in val_map:
                                val_map[lv] = set()
                            if rv:
                                val_map[lv].add(rv)
                    for i, r_val in d.dataframe.iterrows():
                        lv = r_val[l_attr]
                        if lv in val_map and len(val_map[lv]) > 1:
                            out_cells[(i, l_j)] = ""
                            out_cells[(i, r_j)] = ""
            return {"name": json.dumps([algorithm, configuration]), "output": list(out_cells.keys()), "runtime": 0.01}

        def safe_run_strategies(det_self, d):
            algorithm_and_configurations = []
            for attr in d.dataframe.columns[:8]:
                col_vals = "".join(d.dataframe[attr].dropna().astype(str).tolist()[:50])
                for ch in set(col_vals[:15]):
                    algorithm_and_configurations.append(("PVD", [attr, ch]))
            c_list = d.dataframe.columns.tolist()[:6]
            for a in c_list:
                for b in c_list:
                    if a != b:
                        algorithm_and_configurations.append(("RVD", [a, b]))
            profiles = []
            for alg, cfg in algorithm_and_configurations:
                profiles.append(safe_runner(d, alg, cfg))
            d.strategy_profiles = profiles

        raha.detection.Detection.run_strategies = safe_run_strategies

        dd = {"name": table, "path": dirty_path, "clean_path": clean_path}
        res_det = app.run(dd)
        flagged = set()
        for (r_idx, c_idx) in res_det:
            if r_idx < len(dirty_df) and c_idx < len(cols):
                flagged.add((r_idx, cols[c_idx]))

        # Baran simplified repair: value lookup from the 20 labelled tuples
        repairs = {}
        clean_df = pd.read_csv(clean_path, dtype=str)
        value_transforms = {}
        for r in lbl_indices:
            for c_name in cols:
                d_v = str(dirty_df.loc[r, c_name])
                c_v = str(clean_df.loc[r, c_name])
                if d_v != c_v and d_v:
                    value_transforms[(c_name, d_v)] = c_v

        for (r_idx, c_name) in flagged:
            d_v = str(dirty_df.loc[r_idx, c_name])
            if (c_name, d_v) in value_transforms:
                repairs[(r_idx, c_name)] = value_transforms[(c_name, d_v)]
            else:
                repairs[(r_idx, c_name)] = str(dirty_df[c_name].mode().iloc[0]) if len(dirty_df[c_name].mode()) > 0 else d_v

        elapsed = round(time.time() - t0, 3)
        res = score_flags(flagged, cell_records, exclude_rows=excluded_rows, repairs=repairs)
        return {
            "baseline": "B3_Raha_Baran_simplified",
            "table": table,
            "runtime_sec": elapsed,
            "status": "completed",
            "detected_cells": res["detected_cells"],
            "tp": res["tp"],
            "fp": res["fp"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "executed_repairs": res["executed_repairs"],
            "successful_repairs": res["successful_repairs"],
            "repair_accuracy": res["repair_accuracy"],
            "false_repair_rate": res["false_repair_rate"],
            "notes": "Simplified Raha detection with 20-tuple Baran value transfer; scored via score_flags"
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B3_Raha_Baran_simplified",
            "table": table,
            "runtime_sec": elapsed,
            "status": f"failed: {str(e)[:40]}",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "executed_repairs": 0,
            "successful_repairs": 0,
            "repair_accuracy": 0.0,
            "false_repair_rate": 0.0,
            "notes": f"Execution error: {str(e)}"
        }


def main():
    records = []
    print("=" * 75)
    print("Executing Milestone 4 Simplified Baselines (B1 HoloClean, B2 dBoost, B3 Baran)...")
    print("=" * 75)

    for table in TABLES:
        print(f"\nProcessing Table: {table.upper()}...")
        dirty_df, cols, excluded_rows, cell_records = load_table_data(REPO_ROOT, table)
        n_rows = len(dirty_df)
        n_scored_rows = n_rows - len(excluded_rows)
        print(f"  Rows: {n_rows} total, {n_scored_rows} scored ({len(excluded_rows)} labelled rows left out)")

        # B1 HoloClean
        res_b1 = run_b1_holoclean_repair(REPO_ROOT, table, dirty_df, cols, excluded_rows, cell_records)
        res_b1["total_test_rows"] = n_rows
        res_b1["scored_rows"] = n_scored_rows
        records.append(res_b1)
        print(f"  [B1 HoloClean] Status: {res_b1['status']}, F1: {res_b1['f1']:.4f}, RepAcc: {res_b1['repair_accuracy']:.4f}, Time: {res_b1['runtime_sec']}s")

        # B2 dBoost
        res_b2 = run_b2_dboost_grid(REPO_ROOT, table, dirty_df, cols, excluded_rows, cell_records)
        res_b2["total_test_rows"] = n_rows
        res_b2["scored_rows"] = n_scored_rows
        records.append(res_b2)
        print(f"  [B2 dBoost   ] Status: {res_b2['status']}, F1: {res_b2['f1']:.4f}, Time: {res_b2['runtime_sec']}s")

        # B3 Raha / Baran
        res_b3 = run_b3_raha_baran(REPO_ROOT, table, dirty_df, cols, excluded_rows, cell_records)
        res_b3["total_test_rows"] = n_rows
        res_b3["scored_rows"] = n_scored_rows
        records.append(res_b3)
        print(f"  [B3 Baran    ] Status: {res_b3['status']}, F1: {res_b3['f1']:.4f}, RepAcc: {res_b3['repair_accuracy']:.4f}, Time: {res_b3['runtime_sec']}s")

    out_csv = REPO_ROOT / "docs" / "tables" / "m4_baselines_full.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(records)
    col_order = [
        "baseline", "table", "total_test_rows", "scored_rows", "runtime_sec",
        "status", "detected_cells", "tp", "fp", "precision", "recall", "f1",
        "executed_repairs", "successful_repairs", "repair_accuracy", "false_repair_rate", "notes"
    ]
    df_res = df_res[col_order]
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 75)
    print(f"Successfully generated baseline summary table at: {out_csv}")
    print("=" * 75)


if __name__ == "__main__":
    main()
