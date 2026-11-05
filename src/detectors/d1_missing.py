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

def detect(dirty_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    columns = list(dirty_df.columns)
    
    for r_idx, row in enumerate(dirty_df.itertuples(index=False)):
        for c_idx, val in enumerate(row):
            norm_val = normalize(val).lower()
            if norm_val in MISSING_TOKENS or not norm_val:
                records.append({
                    "row": r_idx,
                    "column": columns[c_idx],
                    "detector": "d1",
                    "score": 1.0
                })
                
    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)
