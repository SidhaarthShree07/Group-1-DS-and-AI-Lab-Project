import sys
import time
import tracemalloc
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.build_examples import load_split, TYPE_MAP
from src.detectors import (
    d1_missing,
    d2_pattern,
    d3_robust_z,
    d4_multivariate,
    d5_fd,
    d6_typo,
    build_candidates
)

TABLES = [
    "hospital",
    "flights",
    "beers",
    "rayyan",
    "movies",
    "tax",
    "inpatient",
    "facilities",
]

SPLITS_DIRS = ["data/splits/core", "data/splits/external"]

def evaluate_table(root: Path, table: str):
    dirty_df, cells_df = load_split(root, SPLITS_DIRS, table, "val")
    cols = list(dirty_df.columns)
    n_cols = len(cols)
    
    cands_df = build_candidates(dirty_df, table=table, split="val")
    cand_lookup = set(cands_df[cands_df["is_candidate"]][["row", "column"]].itertuples(index=False, name=None))
    
    total_cells = len(cells_df)
    cand_count = len(cand_lookup)
    pass_through = (cand_count / total_cells) * 100 if total_cells > 0 else 0.0
    
    tp_count = 0
    true_err_count = 0
    type_counts = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}
    type_hits = {"missing": 0, "pattern": 0, "rule": 0, "outlier": 0}
    
    for idx, row in cells_df.iterrows():
        if row["error_type"] == "format_artifact":
            continue  # format-only differences are not labels (M1 decision, M2 Section 6.2)
        if row["is_error"]:
            true_err_count += 1
            r = idx // n_cols
            c = cols[idx % n_cols]
            etype = TYPE_MAP.get(row["error_type"], row["error_type"]) if row["error_type"] else "none"
            if etype in type_counts:
                type_counts[etype] += 1
                
            if (r, c) in cand_lookup:
                tp_count += 1
                if etype in type_hits:
                    type_hits[etype] += 1
                    
    recall_all = (tp_count / true_err_count) * 100 if true_err_count > 0 else 100.0
    
    res = {
        "table": table,
        "rows": len(dirty_df),
        "columns": n_cols,
        "total_cells": total_cells,
        "candidate_cells": cand_count,
        "pass_through_pct": round(pass_through, 2),
        "true_errors": true_err_count,
        "candidates_tp": tp_count,
        "candidate_recall_pct": round(recall_all, 2),
        "recall_missing_pct": round((type_hits["missing"] / type_counts["missing"] * 100), 2) if type_counts["missing"] > 0 else "-",
        "recall_pattern_pct": round((type_hits["pattern"] / type_counts["pattern"] * 100), 2) if type_counts["pattern"] > 0 else "-",
        "recall_rule_pct": round((type_hits["rule"] / type_counts["rule"] * 100), 2) if type_counts["rule"] > 0 else "-",
        "recall_outlier_pct": round((type_hits["outlier"] / type_counts["outlier"] * 100), 2) if type_counts["outlier"] > 0 else "-",
        "missing_errors": type_counts["missing"],
        "pattern_errors": type_counts["pattern"],
        "rule_errors": type_counts["rule"],
        "outlier_errors": type_counts["outlier"],
    }
    return res

def measure_detector_costs(root: Path):
    detectors = {
        "D1 (Missing)": d1_missing.detect,
        "D2 (Pattern)": d2_pattern.detect,
        "D3 (Robust z)": d3_robust_z.detect,
        "D4 (Isolation Forest)": d4_multivariate.detect,
        "D5 (Dependencies)": d5_fd.detect,
        "D6 (Typo / Distance)": d6_typo.detect,
    }
    
    records = []
    total_val_rows = 0
    val_dfs = []
    for table in TABLES:
        dirty, _ = load_split(root, SPLITS_DIRS, table, "val")
        total_val_rows += len(dirty)
        val_dfs.append(dirty)
        
    for det_name, det_fn in detectors.items():
        tracemalloc.start()
        start_time = time.perf_counter()
        
        for df in val_dfs:
            _ = det_fn(df)
            
        elapsed_sec = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        sec_per_1k = (elapsed_sec / total_val_rows) * 1000
        peak_mb = peak_mem / (1024 * 1024)
        
        records.append({
            "detector": det_name,
            "total_runtime_sec": round(elapsed_sec, 3),
            "sec_per_1k_rows": round(sec_per_1k, 4),
            "peak_memory_mb": round(peak_mb, 2)
        })
        
    return pd.DataFrame(records)

def main():
    root = Path(__file__).resolve().parents[1]
    results = []
    print("Evaluating candidate recall on validation splits:")
    for table in TABLES:
        r = evaluate_table(root, table)
        results.append(r)
        print(f"  {table:12s}: recall={r['candidate_recall_pct']}% | pass-through={r['pass_through_pct']}% | errors={r['candidates_tp']}/{r['true_errors']}")
        
    df_results = pd.DataFrame(results)
    out_recall = root / "docs" / "tables" / "m3_candidate_recall.csv"
    out_recall.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_recall, index=False)
    print(f"\nSaved {out_recall}")
    
    print("\nMeasuring detector computational costs (CPU):")
    df_costs = measure_detector_costs(root)
    out_costs = root / "docs" / "tables" / "m3_detector_cost.csv"
    df_costs.to_csv(out_costs, index=False)
    print(f"Saved {out_costs}")
    print(df_costs.to_string(index=False))

if __name__ == "__main__":
    main()
