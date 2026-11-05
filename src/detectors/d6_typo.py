import html
import re
import pandas as pd
from collections import Counter

MISSING_TOKENS = {
    "", "nan", "null", "none", "n/a", "na", "-", "?", 
    "not available", "undefined", "empty"
}

def normalize(val):
    if val is None or pd.isna(val):
        return ""
    text = html.unescape(str(val))
    text = re.sub(r"[\t\n ]+", " ", text)
    return text.strip("\t\n ")

def lev_dist(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return lev_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            replace = prev[j] + (c1 != c2)
            curr[j + 1] = min(insert, delete, replace)
        prev = curr
    return prev[-1]

def detect(dirty_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    
    for col in dirty_df.columns:
        series = dirty_df[col].map(normalize)
        valid_vals = [v for v in series if v.lower() not in MISSING_TOKENS and len(v) >= 3]
        if len(valid_vals) < 20:
            continue
            
        counts = Counter(valid_vals)
        frequent = [v for v, cnt in counts.items() if cnt >= 4]
        if not frequent:
            continue
            
        rare = {v for v, cnt in counts.items() if cnt <= 2}
        typo_map = {}
        
        for r_val in rare:
            r_len = len(r_val)
            for f_val in frequent:
                if abs(len(f_val) - r_len) > 2:
                    continue
                d = lev_dist(r_val.lower(), f_val.lower())
                if d == 1 or (d == 2 and r_len >= 7):
                    score = 1.0 if d == 1 else 0.75
                    typo_map[r_val] = score
                    break
                    
        if typo_map:
            for r_idx, val in enumerate(series):
                if val in typo_map:
                    records.append({
                        "row": r_idx,
                        "column": col,
                        "detector": "d6",
                        "score": typo_map[val]
                    })
                    
    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)
