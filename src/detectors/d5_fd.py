import html
import re
import pandas as pd
from collections import defaultdict

def normalize(val):
    if val is None or pd.isna(val):
        return ""
    text = html.unescape(str(val))
    text = re.sub(r"[\t\n ]+", " ", text)
    return text.strip("\t\n ")

def _g3(df: pd.DataFrame, lhs: list[str], rhs: str) -> float:
    g = df.groupby(lhs + [rhs]).size().reset_index(name="k")
    return 1.0 - g.groupby(lhs)["k"].max().sum() / len(df)

def _support(df: pd.DataFrame, lhs: list[str]) -> float:
    sizes = df.groupby(lhs).size()
    return float(sizes[sizes >= 2].sum() / len(df))

def find_single_dependencies(df: pd.DataFrame, max_g3: float = 0.15, min_support: float = 0.2, min_gain: float = 0.03):
    n = len(df)
    nuniq = df.nunique()
    maj = {c: df[c].value_counts().iloc[0] / n for c in df.columns}
    cand = [c for c in df.columns if 2 <= nuniq[c] <= 0.7 * n]
    deps = []
    
    for a in cand:
        for b in df.columns:
            if a == b or nuniq[b] < 2:
                continue
            g3 = _g3(df, [a], b)
            strength = 1.0 - g3
            if g3 <= max_g3 and (strength - maj[b]) >= min_gain and _support(df, [a]) >= min_support:
                deps.append((a, b, strength))
    return deps

def detect(dirty_df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(dirty_df)
    if n_rows < 10:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
        
    df_norm = dirty_df.map(normalize)
    deps = find_single_dependencies(df_norm)
    
    records = []
    seen = set()
    
    for lhs, rhs, strength in deps:
        groups = defaultdict(list)
        for r_idx, val in enumerate(df_norm[lhs]):
            groups[val].append(r_idx)
            
        for l_val, r_indices in groups.items():
            if len(r_indices) < 2 or not l_val:
                continue
                
            r_vals = [df_norm[rhs].iloc[i] for i in r_indices]
            val_counts = pd.Series(r_vals).value_counts()
            maj_val = val_counts.index[0]
            maj_cnt = val_counts.iloc[0]
            maj_share = maj_cnt / len(r_indices)
            
            if maj_share >= 0.70 and maj_val:
                for idx, r_val in zip(r_indices, r_vals):
                    if r_val != maj_val and r_val:
                        key = (idx, rhs)
                        if key not in seen:
                            seen.add(key)
                            records.append({
                                "row": idx,
                                "column": rhs,
                                "detector": "d5",
                                "score": round(float(maj_share), 4)
                            })
                            
    if not records:
        return pd.DataFrame(columns=["row", "column", "detector", "score"])
    return pd.DataFrame(records)
