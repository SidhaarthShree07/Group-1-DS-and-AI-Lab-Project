import re
import numpy as np
import pandas as pd

ID_PATTERN = re.compile(r"(^id$|_id$|^index$|zip|phone|code|npi|ccn|number|numbr)", re.I)

def to_num(series: pd.Series) -> pd.Series:
    clean_series = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(clean_series, errors="coerce")

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        std = np.std(x)
        if std == 0:
            return np.zeros_like(x)
        return np.abs(x - med) / std
    return 0.6745 * np.abs(x - med) / mad

def detect(dirty_df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    records = []
    
    for col in dirty_df.columns:
        if ID_PATTERN.search(col):
            continue
            
        s = dirty_df[col]
        nums = to_num(s)
        valid_mask = nums.notna()
        n_valid = int(valid_mask.sum())
        
        if n_valid < 10 or (n_valid / max(1, len(s))) < 0.6:
            continue
            
        indices = np.where(valid_mask)[0]
        vals = nums.iloc[indices].to_numpy()
        
        z_scores = robust_z(vals)
        outlier_pos = np.where(z_scores > threshold)[0]
        
        for pos in outlier_pos:
            row_idx = int(indices[pos])
            z_val = float(z_scores[pos])
            score = round(min(1.0, z_val / 10.0), 4)
            records.append({
                "row": row_idx,
                "column": col,
                "detector": "d3",
                "score": score
            })
            
    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)
