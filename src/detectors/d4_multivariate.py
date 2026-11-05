import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ID_PATTERN = re.compile(r"(^id$|_id$|^index$|zip|phone|code|npi|ccn|number|numbr)", re.I)

def to_num(series: pd.Series) -> pd.Series:
    clean_series = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(clean_series, errors="coerce")

def detect(dirty_df: pd.DataFrame, contamination: float = 0.05, seed: int = 42) -> pd.DataFrame:
    records = []
    n_rows = len(dirty_df)
    if n_rows < 20:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
        
    num_cols = []
    num_data = {}
    
    for col in dirty_df.columns:
        if ID_PATTERN.search(col):
            continue
        nums = to_num(dirty_df[col])
        if nums.notna().mean() >= 0.70 and nums.nunique() >= 10:
            num_cols.append(col)
            med = nums.median()
            num_data[col] = nums.fillna(med if not np.isnan(med) else 0.0).to_numpy()
            
    if not num_cols:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
        
    X = np.column_stack([num_data[c] for c in num_cols])
    
    clf = IsolationForest(contamination=contamination, random_state=seed, n_estimators=100)
    preds = clf.fit_predict(X)
    scores = -clf.decision_function(X)
    
    outlier_rows = np.where(preds == -1)[0]
    
    for r in outlier_rows:
        raw_score = float(scores[r])
        norm_score = round(float(np.clip((raw_score - 0.0) / 0.3, 0.1, 1.0)), 4)
        for col in num_cols:
            records.append({
                "row": int(r),
                "column": col,
                "detector": "d4",
                "score": norm_score
            })
            
    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)
