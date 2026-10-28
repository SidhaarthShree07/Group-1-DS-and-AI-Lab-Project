#!/usr/bin/env python3
"""Outlier and dependency analysis of the eight training tables (train split of each).

Author: Aaradhya (file prepared with Sidhaarth's review fixes)

Outliers: two unsupervised detectors on the numeric MEASURE columns (robust z-score with the median absolute deviation, and
Isolation Forest) scored against the cell truth of the train split. A column is a measure when at least 95% of its clean
values are numbers, it has at least 20 distinct values, fewer than half of its values are distinct, and its name is not an
identifier (id, index, zip, phone, code, number, npi, ccn). Cells that are only a number written in another format (3 and 3.0,
type `format_artifact`) are not errors for this analysis and are left out. Recall is reported over ALL true errors of the
column, because an error that is not even a number (for example `0.09%`) is missed by construction, and also over the
errors that are numbers.

Dependencies: src/profiling/dependencies.py (g3 scan with key, support and gain filters) on the dirty train table, plus a
heatmap of the single-column strengths.

Needs the split files written by src/data/split.py and src/data/load_external.py.
Usage: python scripts/eda_analysis.py [--root .]
Outputs: docs/tables/m2_outliers.csv, docs/tables/m2_dependencies.csv, docs/figures/m2/dependency_heatmap.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.profiling.context import norm  # noqa: E402
from src.profiling.dependencies import scan_dependencies, strength_matrix  # noqa: E402

TABLES = [("core", t) for t in ("hospital", "flights", "beers", "rayyan", "movies")] + \
         [("external", t) for t in ("tax", "inpatient", "facilities")]
META = {"row_id", "tuple_id", "index", "id_col"}
ID_NAME = re.compile(r"(^id$|_id$|^index$|zip|phone|code|npi|ccn|number|numbr)", re.I)
MAD_THRESHOLD, IF_CONTAMINATION, SEED = 3.5, 0.05, 42


def load_train(root: Path, group: str, table: str):
    base = root / "data" / "splits" / group / table
    dirty = pd.read_csv(base / "train.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clean = pd.read_csv(base / "train_clean.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    cells = pd.read_csv(base / "train_cells.csv", dtype=str, keep_default_na=False)
    keep = [i for i, c in enumerate(dirty.columns) if c.strip().lower() not in META]
    dirty, clean = dirty.iloc[:, keep].reset_index(drop=True), clean.iloc[:, keep].reset_index(drop=True)
    clean.columns = dirty.columns  # aligned by position
    return dirty, clean, cells


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(lambda v: norm(v).replace(",", "")), errors="coerce")


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * np.abs(x - med) / mad


def measure_columns(dirty: pd.DataFrame, clean: pd.DataFrame) -> list:
    n = len(clean)
    cols = []
    for c in clean.columns:
        if ID_NAME.search(c):
            continue
        present = clean[c][clean[c].map(norm) != ""]
        if len(present) == 0:
            continue
        nums = to_num(present)
        if nums.notna().mean() >= 0.95 and nums.nunique() >= 20 and clean[c].nunique() < 0.5 * n:
            cols.append(c)
    return cols


def analyse_outliers(root: Path) -> pd.DataFrame:
    rows = []
    for group, table in TABLES:
        dirty, clean, cells = load_train(root, group, table)
        n_cols = len(dirty.columns)
        truth = cells["is_error"].eq("True").to_numpy().reshape(len(dirty), n_cols)
        art = cells["error_type"].eq("format_artifact").to_numpy().reshape(len(dirty), n_cols)
        measures = measure_columns(dirty, clean)
        for c in measures:
            j = list(dirty.columns).index(c)
            use = ~art[:, j]  # cells that are only a format change are left out
            err = truth[:, j] & use
            x = to_num(dirty[c]).to_numpy()
            ok = ~np.isnan(x) & use
            res = {"table": table, "column": c, "cells_used": int(use.sum()), "numeric_cells": int(ok.sum()),
                   "true_errors": int(err.sum()), "true_errors_numeric": int((err & ok).sum())}
            for name in ("mad", "if"):
                flag = np.zeros(len(x), bool)
                if name == "mad":
                    flag[ok] = robust_z(x[ok]) > MAD_THRESHOLD
                elif ok.sum() >= 20:
                    flag[ok] = IsolationForest(contamination=IF_CONTAMINATION, random_state=SEED).fit_predict(x[ok].reshape(-1, 1)) == -1
                tp = int((flag & err).sum())
                fp = int((flag & ~err & use).sum())
                res.update({f"{name}_flagged": int(flag.sum()), f"{name}_tp": tp, f"{name}_fp": fp,
                            f"{name}_precision": round(tp / flag.sum(), 4) if flag.sum() else 0.0,
                            f"{name}_recall_all_errors": round(tp / err.sum(), 4) if err.sum() else 0.0,
                            f"{name}_recall_numeric_errors": round(tp / (err & ok).sum(), 4) if (err & ok).sum() else 0.0})
            rows.append(res)
        print(f"{table:11s} measure columns: {measures}", flush=True)
    return pd.DataFrame(rows)


def analyse_dependencies(root: Path):
    frames, mats = [], {}
    for group, table in TABLES:
        dirty, _, _ = load_train(root, group, table)
        df = dirty.map(norm)
        found = scan_dependencies(df)
        found.insert(0, "table", table)
        frames.append(found)
        mats[table] = strength_matrix(df)
        print(f"{table:11s} dependencies kept: {len(found)}", flush=True)
    return pd.concat(frames, ignore_index=True), mats


def draw_heatmaps(mats: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(34, 15))
    for ax, (table, m) in zip(axes.ravel(), mats.items()):
        sns.heatmap(m.astype(float), vmin=0, vmax=1, cmap="Blues", ax=ax, cbar_kws={"label": "strength (1 - g3)"},
                    linewidths=0.2, linecolor="#e5e7eb", mask=m.isna())
        ax.set_title(f"{table.capitalize()} (rows: determinant, columns: determined; blank = key-like or same column)", fontsize=9)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_t, out_f = root / "docs" / "tables", root / "docs" / "figures" / "m2"
    out_t.mkdir(parents=True, exist_ok=True)
    outliers = analyse_outliers(root)
    outliers.to_csv(out_t / "m2_outliers.csv", index=False)
    deps, mats = analyse_dependencies(root)
    deps.to_csv(out_t / "m2_dependencies.csv", index=False)
    draw_heatmaps(mats, out_f / "dependency_heatmap.png")
    print(f"outlier columns: {len(outliers)}; dependencies: {len(deps)}; wrote tables and figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
