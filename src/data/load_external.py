#!/usr/bin/env python3
"""Load, verify and audit the external tables; build cell files and splits for the training tables.

Author: Sahilpreet (file prepared with Sidhaarth's review fixes)

Tables: Tax (Raha, sample), Inpatient and Facilities (BClean) for training; Billionaire (ZeroED) and the Germany
slice of Clean Me If You Can for unseen-table testing. Raw files come from scripts/download_data.py (data/raw/).

For every table the script
  1. checks the SHA-256 of the raw files against the values in data/download_log.md,
  2. aligns clean and dirty columns and counts error cells with the Raha rule (the same rule as scripts/data_inventory.py),
  3. types each changed cell with the documented heuristic of scripts/external_inventory.py
     (missing_value, format_artifact, rule_violation, pattern_violation),
  4. writes data/processed/external/<table>_cells.csv (one row per cell: dirty value, clean value, is_error, type),
  5. for the training tables only: splits 70 / 15 / 15 (seed 42, by entity where the table repeats an entity) into
     data/splits/external/<table>/{train,val,test}{,_clean,_cells}.csv, the same layout as the core tables,
and writes docs/tables/m2_external_sources.csv, m2_external_dictionary.csv, m2_external_splits.csv and
m2_germany_sample_metadata.json.

Unseen tables (Billionaire, Germany) are only loaded and counted; they never enter a split or a generator.
Usage: python src/data/load_external.py [--raw data/raw] [--out docs/tables]
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(10**9)
SEED = 42
MISSING = {"", "nan", "null", "none", "n/a", "na", "?", "-", "not available"}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
TAX_SAMPLE = 20000
DE_SAMPLE = 5000
APPROVED = "approved for research use and sharing with attribution (team licence review, 28 Oct 2026)"
INHERITED = "no licence file in the repository; terms inherited from the benchmarks it bundles"

SOURCES = [
    {"name": "tax", "title": "Tax", "role": "training", "source": "Raha", "url": "https://github.com/BigDaMa/raha",
     "version": "commit 7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c (5 Jun 2025)", "licence": "Apache-2.0",
     "clean": "raha/datasets/tax/clean.csv", "dirty": "raha/datasets/tax/dirty.csv", "index": False, "sample": TAX_SAMPLE,
     "entity": None,
     "sha": ("201290927ae92e65b3940d776b3df5b4d953c5dfd9abb231715a2e65ecca87b0",
             "8dd3429ec4791b2ed1a688c308a57a9f3d1a94f77d1f4e98294a67273270b973")},
    {"name": "inpatient", "title": "Inpatient", "role": "training", "source": "BClean",
     "url": "https://github.com/thethe-github/BClean", "version": "commit bb8e09e3119dcc85475184592a311901a6b4c290 (15 Apr 2024)",
     "licence": INHERITED, "clean": "BClean/dataset/Inpatient/Inpatient_clean.csv",
     "dirty": "BClean/dataset/Inpatient/Inpatient_dirty_10.csv", "index": True, "sample": None, "entity": "NPI",
     "sha": ("75bd12ef098de0b7c6f4a5153890ed3ec5af731dc7f5404b741c9509320337c6",
             "680da3f6c11ec267dbf47397138568224deede0aeef1fc54a028a87527b9cfab")},
    {"name": "facilities", "title": "Facilities", "role": "training", "source": "BClean",
     "url": "https://github.com/thethe-github/BClean", "version": "commit bb8e09e3119dcc85475184592a311901a6b4c290 (15 Apr 2024)",
     "licence": INHERITED, "clean": "BClean/dataset/facilities/facilities_clean.csv",
     "dirty": "BClean/dataset/facilities/facilities_dirty10.csv", "index": True, "sample": None, "entity": "CCN",
     "sha": ("07fa1a78b6fd3a87cc867f093cbb96feea4770cc5f3d5be7b8d1469adaf53198",
             "3d5fadd2cf96b9e95712e3c839a235cf7989ab9865400a52b64c86adbb59ab39")},
    {"name": "billionaire", "title": "Billionaire", "role": "unseen test", "source": "ZeroED",
     "url": "https://github.com/ZJU-DataVIP-Lab/ZeroED", "version": "commit 2b84266e1d8b2014d36643d343429d9a20bd769e (28 Feb 2025)",
     "licence": INHERITED, "clean": "ZeroED/data/billionaire_clean.csv", "dirty": "ZeroED/data/billionaire_error-01.csv",
     "index": False, "sample": None, "entity": None,
     "sha": ("9103811393f2c192094f339963c2b0537adf1384ddb2803a5783da41a326b8cf",
             "c1c3f3a20b1778603b0655027c64b1e74153d350edf181e164bdf4ae016940d7")},
    {"name": "germany", "title": "Germany (Clean Me If You Can)", "role": "unseen test", "source": "Clean Me If You Can",
     "url": "https://zenodo.org/records/20841898", "version": "AddressTable v1.0-2026-06-26, de_slice/named",
     "licence": "data: ODbL 1.0 (OpenStreetMap-derived ground truth; Zenodo record lists CC BY 4.0); code: MIT",
     "clean": "germany/germany_clean.csv", "dirty": "germany/germany_dirty.csv", "index": False, "sample": DE_SAMPLE,
     "entity": None, "id_column": True,
     "sha": ("78c852e3a5680a7460732b67ab87d82484a718297190fa8852af73a80817a192",
             "2ef8ab1424c8c357af1aeaecef12b376cb7f20616437035f2753d0b2f2ac86ef")},
]

MEANINGS = {
    "tax": {"f_name": "First name", "l_name": "Last name", "gender": "Gender code", "area_code": "Telephone area code",
            "phone": "Telephone number", "city": "City", "state": "US state code", "zip": "ZIP code",
            "marital_status": "Marital status code", "has_child": "Has a child (Y/N)", "salary": "Annual salary",
            "rate": "Tax rate", "single_exemp": "Single exemption amount", "married_exemp": "Married exemption amount",
            "child_exemp": "Child exemption amount"},
    "inpatient": {"prvdr_id": "Provider identifier", "frst_nm": "Provider first name", "lst_nm": "Provider last name",
                  "adrs_id": "Address identifier", "adr_ln_1": "Street address line", "cty": "City", "st": "State code",
                  "zip": "ZIP code", "org_pac_id": "Organisation identifier", "phn_numbr": "Telephone number",
                  "NPI": "National Provider Identifier"},
    "facilities": {"CCN": "CMS Certification Number of the facility", "Facility": "Facility name", "Address1": "Street address",
                   "City": "City", "State": "State code", "ZipCode": "ZIP code", "County": "County", "PhoneNumber": "Telephone number",
                   "CMSRegion": "CMS region number", "Condition": "Condition group the record refers to"},
    "billionaire": {"Name": "Name of the person", "Rank": "Rank on the list", "Year": "Year of the list",
                    "Company Founded": "Year the company was founded", "Company Name": "Company name",
                    "Company Relationship": "Relation of the person to the company", "Company Sector": "Company sector",
                    "Company Type": "Company type", "Demographics Age": "Age", "Demographics Gender": "Gender",
                    "Location Citizenship": "Country of citizenship", "Location Country Code": "Country code",
                    "Location GDP": "GDP of the country", "Location Region": "World region", "Wealth Type": "Type of wealth",
                    "Wealth Worth In Billions": "Net worth in billions", "Wealth How Category": "How the wealth was made: category",
                    "Wealth How From Emerging": "From an emerging market (TRUE/FALSE)", "Wealth How Industry": "Industry of the wealth",
                    "Wealth How Inherited": "Inherited or not", "Wealth How Was Founder": "Was the founder (TRUE/FALSE)",
                    "Wealth How Was Political": "Wealth linked to politics (TRUE/FALSE)"},
    "germany": {"name": "Name of the organisation or place", "road": "Street", "house_number": "House number",
                "postcode": "Postal code", "locality": "Town or city", "region": "Federal state", "country": "Country name",
                "country_code": "Country code", "provenance_graph_iri": "URL of the web page the record was taken from"},
}


# ---- the shared rules -------------------------------------------------------------------------
def norm(v: str) -> str:
    """The Raha comparison rule."""
    return re.sub("[\t\n ]+", " ", html.unescape(v)).strip("\t\n ")


def is_artifact(a: str, b: str) -> bool:
    """Same number written differently (3 and 3.0) or a leading zero added or dropped (1203 and 01203)."""
    if NUM.match(a) and NUM.match(b):
        return float(a) == float(b) or a.lstrip("0") == b.lstrip("0")
    return False


def classify(a: str, b: str, clean_values: set) -> str:
    """a = normalised clean value, b = normalised dirty value (a != b). Heuristic of scripts/external_inventory.py."""
    if b.lower() in MISSING and a.lower() not in MISSING:
        return "missing_value"
    if is_artifact(a, b):
        return "format_artifact"
    if b in clean_values:
        return "rule_violation"  # a valid value of this column in the wrong place
    return "pattern_violation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.reader(f))


# ---- loading ------------------------------------------------------------------------------------
def load_pair(src: dict, raw: Path):
    """Return header, clean rows, dirty rows (aligned, same columns) and the original row numbers."""
    cp, dp = raw / src["clean"], raw / src["dirty"]
    for p, expected in zip((cp, dp), src["sha"]):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run scripts/download_data.py")
        got = sha256(p)
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {p}: {got} != {expected}")
    C, D = read(cp), read(dp)
    if src["index"]:  # BClean's dirty files carry a leading unnamed row-index column 0..n-1
        assert D[0][0].strip() in ("", "Unnamed: 0") and all(r[0].strip() == str(i) for i, r in enumerate(D[1:6])), src["name"]
        D = [r[1:] for r in D]
    assert C[0] == D[0], f"{src['name']}: headers differ"
    header, clean, dirty = C[0], C[1:], D[1:]
    ids = list(range(len(clean)))
    if src.get("id_column"):  # Germany: align by id, then draw the M1 sample
        cd, dd = {r[0]: r for r in clean if r}, {r[0]: r for r in dirty if r}
        common = sorted(set(cd) & set(dd))
        drawn = random.Random(SEED).sample(common, src["sample"])
        clean, dirty, ids = [cd[i] for i in drawn], [dd[i] for i in drawn], drawn
        return header, clean, dirty, ids
    assert len(clean) == len(dirty), f"{src['name']}: row counts differ"
    if src["sample"] and src["sample"] < len(clean):
        pick = sorted(random.Random(SEED).sample(range(len(clean)), src["sample"]))
        clean, dirty, ids = [clean[i] for i in pick], [dirty[i] for i in pick], pick
    return header, clean, dirty, ids


def build_cells(header, clean, dirty, skip=()):
    """One record per cell. `skip` lists column positions that are not data (the Germany id)."""
    cols = [j for j in range(len(header)) if j not in skip]
    clean_values = {j: {norm(r[j]) for r in clean} for j in cols}
    cells, counts, per_col = [], collections.Counter(), collections.Counter()
    for i, (cr, dr) in enumerate(zip(clean, dirty)):
        for j in cols:
            a, b = norm(cr[j]), norm(dr[j])
            err = a != b
            etype = classify(a, b, clean_values[j]) if err else ""
            if err:
                counts[etype] += 1
                per_col[header[j]] += 1
            cells.append({"row": i, "column": header[j], "dirty_value": dr[j], "clean_value": cr[j],
                          "is_error": err, "error_type": etype})
    return cells, counts, per_col


# ---- splitting ----------------------------------------------------------------------------------
def split_groups(groups: dict, n_rows: int, seed: int = SEED):
    """70 / 15 / 15 by group (best fit); rows of a group stay together."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = {"train": 0.70 * n_rows, "val": 0.15 * n_rows, "test": 0.15 * n_rows}
    out = {s: [] for s in target}
    for k in keys:
        best = max(target, key=lambda s: (target[s] - len(out[s])) / target[s])
        out[best].extend(groups[k])
    return {s: sorted(v) for s, v in out.items()}


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_splits(src, header, clean, dirty, ids, cells, splits_dir: Path):
    n_cols = len(header)
    if src["entity"]:
        j = header.index(src["entity"])
        groups = collections.defaultdict(list)
        for i, r in enumerate(clean):
            groups[r[j]].append(i)  # the entity is read from the CLEAN table: dirty keys have typos
    else:
        groups = {i: [i] for i in range(len(clean))}
    parts = split_groups(groups, len(clean))
    sets = {s: set(v) for s, v in parts.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    assert sum(len(v) for v in parts.values()) == len(clean)
    if src["entity"]:
        ent = {s: {clean[i][j] for i in v} for s, v in parts.items()}
        assert not (ent["train"] & ent["val"] or ent["train"] & ent["test"] or ent["val"] & ent["test"])
    summary = []
    for s, rows in parts.items():
        base = splits_dir / src["name"]
        write_csv(base / f"{s}.csv", ["row_id"] + header, [[ids[i]] + dirty[i] for i in rows])
        write_csv(base / f"{s}_clean.csv", ["row_id"] + header, [[ids[i]] + clean[i] for i in rows])
        recs = []
        for new, i in enumerate(rows):
            for c in cells[i * n_cols:(i + 1) * n_cols]:
                recs.append([new, c["column"], c["dirty_value"], c["clean_value"], c["is_error"], c["error_type"]])
        write_csv(base / f"{s}_cells.csv", ["row", "column", "dirty_value", "clean_value", "is_error", "error_type"], recs)
        err = [r for r in recs if r[4]]
        types = collections.Counter(r[5] for r in err)
        summary.append({"table": src["name"], "split": s, "entity_column": src["entity"] or "row",
                        "entities": len({clean[i][j] for i in rows}) if src["entity"] else len(rows), "rows": len(rows),
                        "cells": len(recs), "error_cells": len(err), "missing_value": types["missing_value"],
                        "pattern_violation": types["pattern_violation"], "rule_violation": types["rule_violation"],
                        "format_artifact": types["format_artifact"]})
    return summary


# ---- dictionary -----------------------------------------------------------------------------------
def dictionary_rows(src, header, clean, dirty, skip=()):
    rows = []
    for j, col in enumerate(header):
        if j in skip:
            continue
        vals = [norm(r[j]) for r in clean]
        present = [v for v in vals if v.lower() not in MISSING]
        nums = sum(1 for v in present if NUM.match(v.replace(",", "")))
        kind = ("boolean" if present and set(v.lower() for v in present) <= {"true", "false"} else
                "numeric" if present and nums / len(present) >= 0.9 else "text/categorical")
        miss = sum(1 for r in dirty if norm(r[j]).lower() in MISSING) / max(1, len(dirty)) * 100
        rows.append({"table_name": src["name"], "column_name": col, "data_type": kind,
                     "semantic_meaning": MEANINGS.get(src["name"], {}).get(col, f"Attribute {col}"),
                     "example_value": present[0] if present else "", "missing_rate_percent": f"{miss:.2f}%"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="docs/tables")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    raw, out = root / args.raw, root / args.out
    sources, dictionary, split_rows = [], [], []

    for src in SOURCES:
        header, clean, dirty, ids = load_pair(src, raw)
        skip = {0} if src.get("id_column") else set()
        cells, counts, per_col = build_cells(header, clean, dirty, skip)
        data_cols = len(header) - len(skip)
        n_err = sum(counts.values())
        true_err = n_err - counts["format_artifact"]
        write_csv(root / "data/processed/external" / f"{src['name']}_cells.csv",
                  ["row", "column", "dirty_value", "clean_value", "is_error", "error_type"],
                  [[c["row"], c["column"], c["dirty_value"], c["clean_value"], c["is_error"], c["error_type"]] for c in cells])
        if src["name"] == "germany":
            write_csv(root / "data/processed/germany/germany_5000_clean.csv", header, clean)
            write_csv(root / "data/processed/germany/germany_5000_dirty.csv", header, dirty)
        if src["role"] == "training":
            data_hdr = [h for j, h in enumerate(header) if j not in skip]
            split_rows += write_splits(src, header, clean, dirty, ids, cells, root / "data/splits/external")
        dictionary += dictionary_rows(src, header, clean, dirty, skip)
        sources.append({
            "dataset": src["title"], "role": src["role"], "source": src["source"], "url": src["url"], "version": src["version"],
            "licence_basis": src["licence"], "status": APPROVED, "format": "CSV (clean and dirty pair)",
            "rows_used": len(clean), "data_columns": data_cols, "cells": len(clean) * data_cols,
            "error_cells_raha_rule": n_err, "format_artifacts": counts["format_artifact"], "true_error_cells": true_err,
            "missing_value": counts["missing_value"], "pattern_violation": counts["pattern_violation"],
            "rule_violation": counts["rule_violation"], "error_rate_pct": round(100 * n_err / (len(clean) * data_cols), 2),
            "clean_sha256": src["sha"][0], "dirty_sha256": src["sha"][1],
            "note": ("20,000 rows drawn with random.Random(42).sample; full table 200,000 rows" if src["name"] == "tax" else
                     "5,000 rows drawn with random.Random(42).sample over sorted common ids (full slice 374,533 rows)"
                     if src["name"] == "germany" else "evaluation only: never used for training, prompts or generator settings"
                     if src["role"] == "unseen test" else "dirty file's leading unnamed index column dropped after checking it is 0..n-1"),
        })
        print(f"{src['name']:12s} rows={len(clean):6d} cols={data_cols:2d} errors={n_err:6d} "
              f"(artifacts {counts['format_artifact']}, true {true_err})", flush=True)

    write_csv(out / "m2_external_sources.csv", list(sources[0]), [list(r.values()) for r in sources])
    write_csv(out / "m2_external_dictionary.csv", list(dictionary[0]), [list(r.values()) for r in dictionary])
    write_csv(out / "m2_external_splits.csv", list(split_rows[0]), [list(r.values()) for r in split_rows])
    (out / "m2_germany_sample_metadata.json").write_text(json.dumps({
        "source": "https://zenodo.org/records/20841898 (AddressTable.zip, de_slice/named)",
        "sampling_method": "random.Random(42).sample over the sorted ids present in both files (same as the M1 audit)",
        "seed": SEED, "sample_rows": DE_SAMPLE, "raw_clean_sha256": SOURCES[4]["sha"][0], "raw_dirty_sha256": SOURCES[4]["sha"][1],
        "sample_files": {"germany_5000_clean.csv": sha256(root / "data/processed/germany/germany_5000_clean.csv"),
                         "germany_5000_dirty.csv": sha256(root / "data/processed/germany/germany_5000_dirty.csv")}},
        indent=2) + "\n", encoding="utf-8")
    print("wrote m2_external_sources.csv, m2_external_dictionary.csv, m2_external_splits.csv, m2_germany_sample_metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
