import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.build_examples import load_split, TYPE_MAP
from src.detectors import DEFAULT_DETECTORS, build_candidates, d5_entity_consensus, d7_format_mix

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

def evaluate_table_m4(root: Path, table: str, m3_df: pd.DataFrame):
    dirty_df, cells_df = load_split(root, SPLITS_DIRS, table, "val")
    cols = list(dirty_df.columns)
    n_cols = len(cols)
    
    # final M4 detector set: D1-D6, D7 (format mix) and the D5 entity-consensus extension
    detectors = dict(DEFAULT_DETECTORS)
    detectors["d7"] = d7_format_mix.detect
    detectors["d5c"] = lambda df, name=table: d5_entity_consensus.detect(df, name)
    cands_df = build_candidates(dirty_df, table=table, split="val", detectors=detectors)
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
            continue
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
    
    # Lookup M3 metrics for comparison
    m3_row = m3_df[m3_df["table"] == table].iloc[0] if not m3_df.empty and table in m3_df["table"].values else None
    m3_recall = float(m3_row["candidate_recall_pct"]) if m3_row is not None else 0.0
    m3_pt = float(m3_row["pass_through_pct"]) if m3_row is not None else 0.0
    m3_cands = int(m3_row["candidate_cells"]) if m3_row is not None else 0
    m3_tp = int(m3_row["candidates_tp"]) if m3_row is not None else 0

    return {
        "table": table,
        "rows": len(dirty_df),
        "columns": n_cols,
        "total_cells": total_cells,
        "true_errors": true_err_count,
        "m3_candidate_cells": m3_cands,
        "m3_pass_through_pct": m3_pt,
        "m3_candidates_tp": m3_tp,
        "m3_candidate_recall_pct": m3_recall,
        "m4_candidate_cells": cand_count,
        "m4_pass_through_pct": round(pass_through, 2),
        "m4_candidates_tp": tp_count,
        "m4_candidate_recall_pct": round(recall_all, 2),
        "delta_recall_pct": round(recall_all - m3_recall, 2),
        "delta_pass_through_pct": round(pass_through - m3_pt, 2),
        "recall_missing_pct": round((type_hits["missing"] / type_counts["missing"] * 100), 2) if type_counts["missing"] > 0 else "-",
        "recall_pattern_pct": round((type_hits["pattern"] / type_counts["pattern"] * 100), 2) if type_counts["pattern"] > 0 else "-",
        "recall_rule_pct": round((type_hits["rule"] / type_counts["rule"] * 100), 2) if type_counts["rule"] > 0 else "-",
        "recall_outlier_pct": round((type_hits["outlier"] / type_counts["outlier"] * 100), 2) if type_counts["outlier"] > 0 else "-",
        "missing_errors": type_counts["missing"],
        "pattern_errors": type_counts["pattern"],
        "rule_errors": type_counts["rule"],
        "outlier_errors": type_counts["outlier"],
    }

def main():
    root = Path(__file__).resolve().parents[1]
    m3_path = root / "docs" / "tables" / "m3_candidate_recall.csv"
    m3_df = pd.read_csv(m3_path) if m3_path.exists() else pd.DataFrame()
    
    results = []
    print("Evaluating M4 upgraded candidate recall on validation splits:")
    for table in TABLES:
        try:
            r = evaluate_table_m4(root, table, m3_df)
        except FileNotFoundError:
            print(f"  {table:12s}: validation split not found here, skipped")
            continue
        results.append(r)
        print(f"  {table:12s}: Recall: {r['m3_candidate_recall_pct']}% -> {r['m4_candidate_recall_pct']}% (+{r['delta_recall_pct']}%) | PT: {r['m3_pass_through_pct']}% -> {r['m4_pass_through_pct']}% (+{r['delta_pass_through_pct']}%)")
        
    df_results = pd.DataFrame(results)
    out_recall = root / "docs" / "tables" / "m4_candidate_recall.csv"
    out_recall.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_recall, index=False)
    print(f"\nSaved {out_recall}")

if __name__ == "__main__":
    main()
