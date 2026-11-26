#!/usr/bin/env python3
"""Check the D2 fire-rate candidate filter on saved runs (no model calls).

Author: Sidhaarth Shree

Usage: python scripts/d2_fire_gate_check.py <max_fire_rate> <run dir> [<run dir> ...]
Prints detection counts before and after the filter, the number of model calls it saves and the flags it withdraws.
The model answers for the remaining cells are unchanged, because a prompt depends on the dirty table and the cell only.
"""
import sys, json
import pandas as pd
FIRE=float(sys.argv[1]); runs=sys.argv[2:]
for run in runs:
    cand=pd.read_csv(f"{run}/candidates.csv")
    if "table" not in cand: cand["table"]="t"
    cells=pd.read_csv(f"{run}/cells.csv",dtype=str,keep_default_na=False); cells["row"]=cells.row.astype(int)
    if "table" not in cells: cells["table"]="t"
    cells=cells[~cells.error_type.eq("format_artifact")]
    truth={(t,r,c):e=="True" for t,r,c,e in zip(cells.table,cells.row,cells.column,cells.is_error)}
    others=[d for d in ["d1","d3","d4","d5","d6","d7","d5c"] if d in cand]
    cand["d2only"]=(cand.d2>0)&(cand[others].sum(axis=1)==0)
    rate=cand.assign(f=cand.d2>0).groupby(["table","column"]).f.mean()
    cand["blocked"]=cand.d2only & cand.set_index(["table","column"]).index.map(rate).values.__gt__(FIRE)
    blocked={(t,int(r),c) for t,r,c in zip(cand.table[cand.blocked],cand.row[cand.blocked],cand.column[cand.blocked])}
    flags=[]
    for p in map(json.loads,open(f"{run}/predictions.jsonl")):
        k=(p.get("table","t"),p["row"],p["column"])
        if k in truth and p.get("valid_json") and p.get("is_error"): flags.append(k)
    keep=[k for k in flags if k not in blocked]
    tot=sum(truth.values())
    def prf(fl):
        tp=sum(truth[k] for k in fl); fp=len(fl)-tp; pr=tp/max(1,len(fl)); rc=tp/tot
        return tp,fp,round(pr,3),round(rc,3),round(2*pr*rc/max(1e-9,pr+rc),3)
    lost=[k for k in flags if k in blocked]
    print(f"{run.split('/')[-1]:32s} calls saved {len(blocked)} | before {prf(flags)} | after {prf(keep)} | withdrew {len(lost)} flags ({sum(truth[k] for k in lost)} true)")
