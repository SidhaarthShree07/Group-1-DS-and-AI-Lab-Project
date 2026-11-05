import collections
import html
import re
import pandas as pd

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

def shape(value: str) -> str:
    out, prev, run = [], None, 0
    for ch in value:
        key = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch
        if key == prev:
            run += 1
        else:
            if prev is not None:
                out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
            prev, run = key, 1
    if prev is not None:
        out.append(prev * run if run <= 4 else f"{prev}{{{run}}}")
    return "".join(out)

def detect(dirty_df: pd.DataFrame, rarity_threshold: float = 0.02) -> pd.DataFrame:
    records = []
    columns = list(dirty_df.columns)
    
    for col in columns:
        series = dirty_df[col]
        norm_vals = [normalize(v) for v in series]
        valid_indices = [i for i, v in enumerate(norm_vals) if v.lower() not in MISSING_TOKENS and v]
        
        if len(valid_indices) < 10:
            continue
            
        shapes = [shape(norm_vals[i]) for i in valid_indices]
        counts = collections.Counter(shapes)
        total = len(valid_indices)
        
        for idx in valid_indices:
            s = shape(norm_vals[idx])
            freq = counts[s] / total
            if freq <= rarity_threshold or (counts[s] <= 2 and total >= 30):
                score = round(1.0 - freq, 4)
                records.append({
                    "row": idx,
                    "column": col,
                    "detector": "d2",
                    "score": score
                })
                
    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)
