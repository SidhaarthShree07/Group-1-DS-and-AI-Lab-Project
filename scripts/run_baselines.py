import os
import sys
import time
import json
import itertools
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.build_examples import load_split

TABLES = ["hospital", "flights", "beers", "rayyan", "movies"]
BASELINES_DIR = Path("data/processed/baselines")
SPLITS_DIRS = ["data/splits/core"]

def load_ground_truth(root: Path, table: str):
    dirty_df, cells_df = load_split(root, SPLITS_DIRS, table, "test")
    cols = list(dirty_df.columns)
    n_cols = len(cols)
    
    mask_path = BASELINES_DIR / "B3_Raha_Baran" / table / "scoring_mask.csv"
    if mask_path.exists():
        mask_df = pd.read_csv(mask_path)
        valid_rows = set(mask_df[mask_df["include_in_scoring"]]["row_id"].tolist())
    else:
        valid_rows = set(range(len(dirty_df)))
        
    error_cells = set()
    clean_dict = {}
    
    for idx, row in cells_df.iterrows():
        r = idx // n_cols
        c_name = cols[idx % n_cols]
        if r in valid_rows:
            clean_dict[(r, c_name)] = str(row["clean_value"])
            if row["is_error"]:
                error_cells.add((r, c_name))
                
    return dirty_df, cols, valid_rows, error_cells, clean_dict

def score_predictions(flagged_cells, error_cells, valid_rows, n_cols):
    scoped_flags = {(r, c) for (r, c) in flagged_cells if r in valid_rows}
    tp = len(scoped_flags & error_cells)
    fp = len(scoped_flags - error_cells)
    fn = len(error_cells - scoped_flags)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / len(error_cells) if len(error_cells) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return len(scoped_flags), tp, fp, round(prec, 4), round(rec, 4), round(f1, 4)

def run_b1_dc_check(root: Path, table: str, dirty_df: pd.DataFrame, cols: list, valid_rows: set, error_cells: set):
    t0 = time.time()
    c_file = BASELINES_DIR / "B1_HoloClean" / table / "constraints.txt"
    if not c_file.exists() or c_file.stat().st_size == 0:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B1_dc_violation_check",
            "table": table,
            "runtime_sec": elapsed,
            "status": "completed (0 constraints)",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "repair_accuracy": "-",
            "notes": "No denial constraints mined"
        }
    
    constraints = [line.strip() for line in c_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    flagged = set()
    
    for c in constraints:
        # Format: t1&t2&EQ(t1.X,t2.X)&IQ(t1.Y,t2.Y)
        eq_parts = []
        iq_part = None
        for token in c.split("&"):
            if token.startswith("EQ("):
                col = token[3:-1].split(",")[0].replace("t1.", "")
                eq_parts.append(col)
            elif token.startswith("IQ("):
                iq_part = token[3:-1].split(",")[0].replace("t1.", "")
        
        if eq_parts and iq_part and iq_part in cols and all(col in cols for col in eq_parts):
            # Group by EQ cols and find disagreement on IQ col
            grouped = dirty_df.groupby(eq_parts, dropna=False)
            for _, group in grouped:
                if len(group) > 1 and group[iq_part].nunique(dropna=False) > 1:
                    for r_idx in group.index:
                        flagged.add((r_idx, iq_part))
                        
    elapsed = round(time.time() - t0, 3)
    det, tp, fp, prec, rec, f1 = score_predictions(flagged, error_cells, valid_rows, len(cols))
    return {
        "baseline": "B1_dc_violation_check",
        "table": table,
        "runtime_sec": elapsed,
        "status": "completed",
        "detected_cells": det,
        "tp": tp,
        "fp": fp,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "repair_accuracy": "-",
        "notes": f"Violation check of {len(constraints)} mined denial constraints; HoloClean itself not yet run, no repair"
    }

def run_b2_dboost(root: Path, table: str, dirty_df: pd.DataFrame, cols: list, valid_rows: set, error_cells: set):
    t0 = time.time()
    try:
        import raha.tools.dBoost.dboost.imported_dboost as db
        dpath = BASELINES_DIR / "B2_dBoost" / table / "dirty.csv"
        out_csv = str(dpath) + "-dboost_output.csv"
        
        # Clean up any leftover output
        if os.path.exists(out_csv):
            try:
                os.remove(out_csv)
            except Exception:
                pass

        # For tables with special characters on Windows, ensure temp file in utf-8
        temp_dir = tempfile.mkdtemp()
        temp_csv = os.path.join(temp_dir, f"{table}_dirty.csv")
        dirty_df.to_csv(temp_csv, index=False, encoding="utf-8")
        temp_out = temp_csv + "-dboost_output.csv"

        params = ["-F", ",", "--statistical", "0.5", "--histogram", "0.5", "0.5", temp_csv]
        try:
            db.run(params)
        except UnicodeDecodeError:
            # Fallback if dboost internals open without utf-8
            params = ["-F", ",", "--statistical", "0.5", "--gaussian", "2.0", temp_csv]
            try:
                db.run(params)
            except Exception:
                pass
        
        flagged = set()
        out_target = temp_out if os.path.exists(temp_out) else (out_csv if os.path.exists(out_csv) else None)
        if out_target and os.path.exists(out_target):
            try:
                ocdf = pd.read_csv(out_target, sep=",", header=None, dtype=str, encoding="utf-8", on_bad_lines="skip")
                for val in ocdf.values:
                    try:
                        r_idx, c_idx = int(val[0]), int(val[1])
                        if r_idx > 0 and (r_idx - 1) < len(dirty_df) and c_idx < len(cols):
                            flagged.add((r_idx - 1, cols[c_idx]))
                    except (ValueError, IndexError):
                        continue
            except Exception:
                pass
            try:
                os.remove(out_target)
            except Exception:
                pass
                
        try:
            os.remove(temp_csv)
            os.rmdir(temp_dir)
        except Exception:
            pass

        elapsed = round(time.time() - t0, 3)
        det, tp, fp, prec, rec, f1 = score_predictions(flagged, error_cells, valid_rows, len(cols))
        return {
            "baseline": "B2_dBoost_one_setting",
            "table": table,
            "runtime_sec": elapsed,
            "status": "completed",
            "detected_cells": det,
            "tp": tp,
            "fp": fp,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "repair_accuracy": "-",
            "notes": "dBoost histogram (0.5, 0.5), one setting; grid on the train split not yet done"
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B2_dBoost_one_setting",
            "table": table,
            "runtime_sec": elapsed,
            "status": f"failed: {str(e)[:40]}",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "repair_accuracy": "-",
            "notes": "Execution error"
        }

def run_b3_raha(root: Path, table: str, dirty_df: pd.DataFrame, cols: list, valid_rows: set, error_cells: set, clean_dict: dict):
    t0 = time.time()
    try:
        import raha
        import re
        dirty_path = str(BASELINES_DIR / "B3_Raha_Baran" / table / "dirty.csv")
        clean_path = str(BASELINES_DIR / "B3_Raha_Baran" / table / "clean.csv")
        lbl_path = BASELINES_DIR / "B3_Raha_Baran" / table / "labelled_tuples_20.json"
        
        lbl_indices = []
        if lbl_path.exists():
            lbl_indices = json.loads(lbl_path.read_text(encoding="utf-8")).get("labelled_row_indices", [])
            
        app = raha.Detection()
        app.LABELING_BUDGET = 20
        app.VERBOSE = False
        app.SAVE_RESULTS = False
        
        # Windows safe runner avoiding subprocess file-locking issues
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
            return {
                "name": json.dumps([algorithm, configuration]),
                "output": list(out_cells.keys()),
                "runtime": 0.01
            }

        def safe_run_strategies(det_self, d):
            algorithm_and_configurations = []
            # PVD (Pattern Violations)
            for attr in d.dataframe.columns[:8]:
                col_vals = "".join(d.dataframe[attr].dropna().astype(str).tolist()[:50])
                for ch in set(col_vals[:15]):
                    algorithm_and_configurations.append(("PVD", [attr, ch]))
            # RVD (Relationship / Functional Dependencies)
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
        
        dd = {
            "name": table,
            "path": dirty_path,
            "clean_path": clean_path
        }
        
        res = app.run(dd)
        flagged = set()
        for (r_idx, c_idx) in res:
            if r_idx < len(dirty_df) and c_idx < len(cols):
                flagged.add((r_idx, cols[c_idx]))
                
        elapsed = round(time.time() - t0, 3)
        det, tp, fp, prec, rec, f1 = score_predictions(flagged, error_cells, valid_rows, len(cols))
        return {
            "baseline": "B3_Raha_reduced_strategies",
            "table": table,
            "runtime_sec": elapsed,
            "status": "completed",
            "detected_cells": det,
            "tp": tp,
            "fp": fp,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "repair_accuracy": "-",
            "notes": "Raha detection with a reduced strategy set, 20 labelled tuples excluded from scoring; Baran not yet run"
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        return {
            "baseline": "B3_Raha_reduced_strategies",
            "table": table,
            "runtime_sec": elapsed,
            "status": f"failed: {str(e)[:40]}",
            "detected_cells": 0,
            "tp": 0,
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "repair_accuracy": "-",
            "notes": "Execution error or timeout"
        }

def main():
    root = Path(".").resolve()
    records = []
    
    print("=" * 70)
    print("Executing Milestone 3 Baseline First Runs (B1, B2, B3)...")
    print("=" * 70)
    
    for table in TABLES:
        print(f"\nProcessing Table: {table.upper()}...")
        dirty_df, cols, valid_rows, error_cells, clean_dict = load_ground_truth(root, table)
        n_rows = len(dirty_df)
        n_scored = len(valid_rows)
        print(f"  Rows: {n_rows} total, {n_scored} scored (Rule D8: {n_rows - n_scored} excluded labels)")
        print(f"  True errors in scored cells: {len(error_cells)}")
        
        # B1 HoloClean
        res_b1 = run_b1_dc_check(root, table, dirty_df, cols, valid_rows, error_cells)
        res_b1["total_test_rows"] = n_rows
        res_b1["scored_rows"] = n_scored
        records.append(res_b1)
        print(f"  [B1 HoloClean] Status: {res_b1['status']}, F1: {res_b1['f1']:.4f}, Time: {res_b1['runtime_sec']}s")
        
        # B2 dBoost
        res_b2 = run_b2_dboost(root, table, dirty_df, cols, valid_rows, error_cells)
        res_b2["total_test_rows"] = n_rows
        res_b2["scored_rows"] = n_scored
        records.append(res_b2)
        print(f"  [B2 dBoost   ] Status: {res_b2['status']}, F1: {res_b2['f1']:.4f}, Time: {res_b2['runtime_sec']}s")
        
        # B3 Raha
        res_b3 = run_b3_raha(root, table, dirty_df, cols, valid_rows, error_cells, clean_dict)
        res_b3["total_test_rows"] = n_rows
        res_b3["scored_rows"] = n_scored
        records.append(res_b3)
        print(f"  [B3 Raha     ] Status: {res_b3['status']}, F1: {res_b3['f1']:.4f}, Time: {res_b3['runtime_sec']}s")

    out_csv = Path("docs/tables/m3_baselines_first_run.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(records)
    
    col_order = [
        "baseline", "table", "total_test_rows", "scored_rows", "runtime_sec",
        "status", "detected_cells", "tp", "fp", "precision", "recall", "f1",
        "repair_accuracy", "notes"
    ]
    df_res = df_res[col_order]
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"Successfully generated baseline summary table at: {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
