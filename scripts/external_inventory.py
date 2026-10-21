#!/usr/bin/env python3
"""Audit of the external clean/dirty table pairs (Tax, Billionaire, Inpatient, Facilities, Clean Me If You Can DE slice).

Read-only. Error cells follow Raha's rule (html.unescape, collapse [\\t\\n ]+, strip) so the counts are comparable with
scripts/data_inventory.py. Changed cells are typed with a documented heuristic (see classify()), because these sources
ship no error-type annotations. Writes three tables to docs/tables/ and prints leakage-overlap checks.

Usage: python scripts/external_inventory.py --raw data/raw [--out docs/tables]
Expected layout under --raw: raha/datasets/{tax,hospital}, ZeroED/data, BClean/dataset,
AddressTable/de_slice/named (from https://zenodo.org/records/20841898).
"""
import argparse, collections, csv, html, random, re
from pathlib import Path

csv.field_size_limit(10**9)
MISSING = {"", "nan", "null", "none", "n/a", "na", "?", "-", "not available"}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
DE_SAMPLE, DE_SEED = 5000, 42

PAIRS = [  # name, source, clean, dirty, dirty has a leading row-index column
    ("Tax", "Raha", "raha/datasets/tax/clean.csv", "raha/datasets/tax/dirty.csv", False),
    ("Billionaire", "ZeroED", "ZeroED/data/billionaire_clean.csv", "ZeroED/data/billionaire_error-01.csv", False),
    ("Inpatient", "BClean", "BClean/dataset/Inpatient/Inpatient_clean.csv", "BClean/dataset/Inpatient/Inpatient_dirty_10.csv", True),
    ("Facilities", "BClean", "BClean/dataset/facilities/facilities_clean.csv", "BClean/dataset/facilities/facilities_dirty10.csv", True),
]


def norm(v):
    return re.sub("[\t\n ]+", " ", html.unescape(v)).strip("\t\n ")


def read(path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.reader(f))


def is_artifact(a, b):
    """Same number written differently (3 vs 3.0) or a leading zero added or dropped (1203 vs 01203)."""
    if NUM.match(a) and NUM.match(b):
        return float(a) == float(b) or a.lstrip("0") == b.lstrip("0")
    return False


def classify(a, b, clean_values):
    """a = normalised clean value, b = normalised dirty value (a != b)."""
    if b.lower() in MISSING and a.lower() not in MISSING:
        return "missing_value"
    if is_artifact(a, b):
        return "format_artifact"
    if b in clean_values:
        return "rule_violation"      # a valid value of this column in the wrong place: breaks dependencies on other columns
    return "pattern_violation"       # new string: typo, distortion or different format


def clean_quality(rows):
    body = rows[1:]
    blank = sum(1 for r in body for v in r if v.strip() == "")
    placeholder = sum(1 for r in body for v in r if v.strip() != "" and v.strip().lower() in MISSING)
    nan_text = sum(1 for r in body for v in r if v.strip().lower() == "nan" or v.strip().lower().startswith("nan ("))
    dup = len(body) - len({tuple(r) for r in body})
    return blank, placeholder, nan_text, dup


def audit(name, source, header, clean, dirty, ids=None):
    cols = range(len(header))
    clean_vals = [set(norm(r[j]) for r in clean) for j in cols]
    errors = rows_with = 0
    types, per_col = collections.Counter(), collections.Counter()
    raw_changed = 0
    for i in range(len(clean)):
        hit = False
        for j in cols:
            if clean[i][j] != dirty[i][j]:
                raw_changed += 1
            a, b = norm(clean[i][j]), norm(dirty[i][j])
            if a != b:
                hit = True
                errors += 1
                per_col[header[j]] += 1
                types[classify(a, b, clean_vals[j])] += 1
        rows_with += hit
    return dict(dataset=name, source=source, rows=len(clean), data_columns=len(header), cells=len(clean) * len(header),
                error_cells=errors, error_rate_pct=round(100 * errors / (len(clean) * len(header)), 2),
                rows_with_error=rows_with, raw_changed_cells=raw_changed,
                **{t: types.get(t, 0) for t in ("missing_value", "pattern_violation", "rule_violation", "format_artifact")}), per_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inv, col_rows, quality, loaded = [], [], [], {}

    for name, source, c_rel, d_rel, has_index in PAIRS:
        C, D = read(raw / c_rel), read(raw / d_rel)
        if has_index:  # BClean's dirty files carry a leading, unnamed 0..n-1 row-index column
            assert D[0][0].strip() in ("", "Unnamed: 0") and all(r[0].strip() == str(i) for i, r in enumerate(D[1:6])), name
            D = [r[1:] for r in D]
        assert C[0] == D[0] and len(C) == len(D), f"{name}: header or row-count mismatch"
        res, per_col = audit(name, source, C[0], C[1:], D[1:])
        inv.append(res)
        col_rows += [(name, source, c, per_col.get(c, 0), res["rows"]) for c in C[0]]
        quality.append((name, *clean_quality(C)))
        loaded[name] = C

    # Clean Me If You Can: German slice, aligned by id, 5,000 rows drawn with seed 42
    de = raw / "AddressTable/de_slice/named"
    C, D = read(de / "slice_de_clean.csv"), read(de / "slice_de_dirty.csv")
    assert C[0] == D[0]
    cd, dd = {r[0]: r for r in C[1:] if r}, {r[0]: r for r in D[1:] if r}
    common = sorted(set(cd) & set(dd))
    sample = random.Random(DE_SEED).sample(common, DE_SAMPLE)
    header = C[0][1:]  # drop the id column
    cl = [cd[i][1:] for i in sample]
    di = [dd[i][1:] for i in sample]
    res, per_col = audit("CleanMe_DE_slice", "Clean Me If You Can", header, cl, di)
    res["notes"] = f"{len(common):,} rows in the full slice; {DE_SAMPLE} drawn with random.Random({DE_SEED}).sample over sorted ids"
    inv.append(res)
    col_rows += [("CleanMe_DE_slice", "Clean Me If You Can", c, per_col.get(c, 0), DE_SAMPLE) for c in header]
    quality.append(("CleanMe_DE_slice", *clean_quality([header] + cl)))

    fields = ["dataset", "source", "rows", "data_columns", "cells", "error_cells", "error_rate_pct", "rows_with_error",
              "raw_changed_cells", "missing_value", "pattern_violation", "rule_violation", "format_artifact", "notes"]
    with open(out / "external_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in inv:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(out / "external_column_differences.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "source", "column", "error_cells", "rows", "error_rate_pct"])
        for n, s, c, k, rows in col_rows:
            w.writerow([n, s, c, k, rows, round(100 * k / rows, 4)])
    with open(out / "external_clean_quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "blank_cells_in_clean", "placeholder_token_cells_in_clean", "nan_text_cells_in_clean", "duplicate_rows_in_clean"])
        w.writerows(quality)

    for r in inv:
        print(f"{r['dataset']:18} rows={r['rows']:>7,} cols={r['data_columns']:>2} errors={r['error_cells']:>7,} ({r['error_rate_pct']}%) "
              f"missing={r['missing_value']:,} pattern={r['pattern_violation']:,} rule={r['rule_violation']:,} artifact={r['format_artifact']:,}")

    # Leakage checks: Hospital (Raha) against the other healthcare tables, by identifier
    hosp = read(raw / "raha/datasets/hospital/clean.csv")
    hh = [h.lower() for h in hosp[0]]
    def col(rows, name):
        j = [h.lower() for h in rows[0]].index(name.lower())
        return {norm(r[j]) for r in rows[1:] if norm(r[j])}
    h_id, h_phone = col(hosp, "ProviderNumber"), col(hosp, "PhoneNumber")
    fac, inp = loaded["Facilities"], loaded["Inpatient"]
    print("\nOverlap with Hospital (clean files, exact identifier match):")
    print(f"  Facilities CCN in Hospital ProviderNumber: {len(col(fac, 'CCN') & h_id)} of {len(col(fac, 'CCN'))} CCNs; Hospital providers covered: {len(h_id & col(fac, 'CCN'))} of {len(h_id)}")
    print(f"  Facilities PhoneNumber in Hospital PhoneNumber: {len(col(fac, 'PhoneNumber') & h_phone)} shared of {len(h_phone)} Hospital phones")
    print(f"  Inpatient phn_numbr in Hospital PhoneNumber: {len(col(inp, 'phn_numbr') & h_phone)} shared; prvdr_id in ProviderNumber: {len(col(inp, 'prvdr_id') & h_id)} shared")


if __name__ == "__main__":
    main()
