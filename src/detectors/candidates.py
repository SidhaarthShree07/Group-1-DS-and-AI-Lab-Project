from typing import Dict, Any, Optional, Callable
import pandas as pd
from . import d1_missing, d2_pattern, d3_robust_z, d4_multivariate, d5_fd, d6_typo

DEFAULT_DETECTORS = {
    "d1": d1_missing.detect,
    "d2": d2_pattern.detect,
    "d3": d3_robust_z.detect,
    "d4": d4_multivariate.detect,
    "d5": d5_fd.detect,
    "d6": d6_typo.detect,
}

def build_candidates(
    dirty_df: pd.DataFrame,
    table: str = "",
    split: str = "",
    detectors: Optional[Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
    d7_detector: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> pd.DataFrame:
    if detectors is None:
        active_detectors = dict(DEFAULT_DETECTORS)
    else:
        active_detectors = dict(detectors)
        
    if d7_detector is not None:
        active_detectors["d7"] = d7_detector

    scores_by_cell = {}
    for r in range(len(dirty_df)):
        for col in dirty_df.columns:
            scores_by_cell[(r, col)] = {d: 0.0 for d in active_detectors}

    for det_name, det_fn in active_detectors.items():
        res_df = det_fn(dirty_df)
        if res_df is not None and not res_df.empty:
            for item in res_df.itertuples(index=False):
                r = int(item.row)
                c = str(item.column)
                score = float(item.score)
                if (r, c) in scores_by_cell:
                    scores_by_cell[(r, c)][det_name] = max(scores_by_cell[(r, c)][det_name], score)

    rows = []
    det_keys = sorted(active_detectors.keys())
    
    for (r, col), scores in scores_by_cell.items():
        is_cand = any(scores[d] > 0.0 for d in det_keys)
        row_dict = {
            "table": table,
            "split": split,
            "row": r,
            "column": col,
        }
        for d in det_keys:
            row_dict[d] = scores[d]
        row_dict["is_candidate"] = is_cand
        rows.append(row_dict)

    out_df = pd.DataFrame(rows)
    cols_order = ["table", "split", "row", "column"] + det_keys + ["is_candidate"]
    return out_df[cols_order]
