#!/usr/bin/env python3
"""M5 baselines (X8, X9): the real HoloClean, dBoost with Raha's whole grid, Raha with every strategy and the real Baran.

Author: Aaradhya (Group 1, DSAI Lab, IIT Madras BS Degree)

Each system runs in its own environment through scripts/baseline_workers/. This script prepares the inputs, runs the
workers and scores every method with score_flags (scripts/run_baselines_full.py) on the same cells.
  X8  headline: the 20 rows of M4 (export_baselines, seed 42) are left out for every method
  X9  label budgets 5 < 10 < 20 < 50 (nested); the same 50 rows are left out at every budget
Only Raha and Baran see truth, and only for labelled rows: their clean file equals the dirty file everywhere else.

Usage:
  python scripts/run_baselines_m5.py --splits data/splits/core --raha-python <env>/bin/python --hc-python <env>/bin/python
         --hc-src <HoloClean clone> --hc-socket <dir> [--tables ...] [--split test] [--methods holoclean dboost raha]
Needs a running PostgreSQL with a database `holo` and a role `holocleanuser` for HoloClean.
Outputs: outputs/runs/<run-id>/ (not in git), docs/tables/m5_baselines_full.csv, m5_raha_label_budgets.csv, m5_dboost_grid.csv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_baselines_full import score_flags  # noqa: E402
from src.data.export_baselines import DBOOST_GRID, hc_constraint  # noqa: E402
from src.llm.build_examples import META_COLUMNS, load_split  # noqa: E402
from src.profiling.context import norm  # noqa: E402
from src.profiling.dependencies import scan_dependencies  # noqa: E402

CORE = ["hospital", "flights", "beers", "rayyan", "movies"]
M4_EXTRA = [["histogram", "0.2", "0.2", "s=0.2"], ["histogram", "0.8", "0.8", "s=0.8"]]  # the two M4 settings outside Raha's grid
BUDGETS, SEEDS, SEED = [5, 10, 20, 50], [0, 1, 2, 3, 4], 42
WORKERS = ROOT / "scripts" / "baseline_workers"
METRICS = ["n_scored_cells", "total_true_errors", "detected_cells", "tp", "fp", "precision", "recall", "f1",
           "executed_repairs", "successful_repairs", "incorrect_repairs", "repair_accuracy", "false_repair_rate"]


def label_order(table: str, n: int) -> list:
    """50 rows in labelling order. The first 20 are the M4 rows (export_baselines) in a seeded order; prefixes are nested."""
    m4 = sorted(random.Random(f"{SEED}:{table}").sample(range(n), min(20, n)))
    order = m4[:]
    random.Random(f"{SEED}:{table}:order").shuffle(order)
    rest = [i for i in range(n) if i not in set(m4)]
    return order + random.Random(f"{SEED}:{table}:extra").sample(rest, 30)


def load_table(splits: str, table: str, split: str):
    """load_split, or the same for a table that has no train split (the unseen tables)."""
    try:
        return load_split(ROOT, [splits], table, split)
    except FileNotFoundError:
        base = ROOT / splits / table
        dirty = pd.read_csv(base / f"{split}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        dirty = dirty[[c for c in dirty.columns if c.strip().lower() not in META_COLUMNS]].reset_index(drop=True)
        cells = pd.read_csv(base / f"{split}_cells.csv", dtype=str, keep_default_na=False)
        if len(cells) != dirty.size or (cells["dirty_value"].to_numpy() != dirty.to_numpy().ravel()).any():
            raise ValueError(f"{table}: the cell file does not match the dirty table")
        return dirty, cells


def cell_records(dirty: pd.DataFrame, cells: pd.DataFrame) -> list:
    cols, n = list(dirty.columns), dirty.shape[1]
    return [{"row": i // n, "column": cols[i % n], "dirty_value": d, "clean_value": c, "error_type": t}
            for i, (d, c, t) in enumerate(zip(cells["dirty_value"], cells["clean_value"], cells["error_type"]))]


def call(cmd: list, timeout: int, log: Path, env: dict | None = None) -> str:
    t0 = time.time()
    with open(log, "w") as f:
        try:
            code = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                  env={**os.environ, "PYTHONHASHSEED": "0", **(env or {})}).returncode
        except subprocess.TimeoutExpired:
            return f"failed: timeout after {timeout} s"
    return "ok" if code == 0 else f"failed: exit code {code} after {time.time() - t0:.0f} s"


def run_holoclean(table, dirty, work, args):
    # HoloClean's constraint parser cannot read names such as "birthyear (integer)": it sees c0, c1, ...
    names = {f"c{i}": c for i, c in enumerate(dirty.columns)}
    safe = dirty.set_axis(list(names), axis=1)
    safe.to_csv(work / "hc_dirty.csv", index=False)
    lines = [hc_constraint(r.lhs, r.rhs) for r in scan_dependencies(safe.map(norm)).itertuples()]
    (work / "constraints.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    cfg, out = work / "hc.json", work / "hc_out.json"
    cfg.write_text(json.dumps({"name": table, "dirty": str(work / "hc_dirty.csv"), "constraints": str(work / "constraints.txt"),
                               "seed": 45, "db": args.hc_db}))  # 45 is HoloClean's default seed
    out.unlink(missing_ok=True)
    state = call([args.hc_python, "-W", "ignore", WORKERS / "holoclean_worker.py", args.hc_src, args.hc_socket, cfg, out],
                 args.hc_timeout, work / "hc.log")
    if not out.exists():
        return {"status": state, "seconds": None, "constraints": len(lines)}
    r = {"constraints": len(lines), **json.loads(out.read_text())}
    r["noisy"] = [[i, names[c]] for i, c in r.get("noisy", [])]
    r["repairs"] = [[i, names[c], v] for i, c, v in r.get("repairs", [])]
    return r


def run_dboost(dirty_path, name, work, args, grid):
    cfg, out = work / f"dboost_{name}.json", work / f"dboost_{name}_out.json"
    cfg.write_text(json.dumps({"dirty": str(dirty_path), "grid": grid}))
    state = call([args.raha_python, WORKERS / "dboost.py", cfg, out], 3600, work / f"dboost_{name}.log")
    return json.loads(out.read_text()) if state == "ok" else [{"setting": g, "status": state, "cells": [], "seconds": None} for g in grid]


def run_raha(table, dirty, clean, work, args):
    order, masked = label_order(table, len(dirty)), {}
    for k in args.budgets:
        m = dirty.copy()
        m.iloc[order[:k]] = clean.iloc[order[:k]].to_numpy()  # truth for the labelled rows only
        masked[str(k)] = str(work / f"masked_clean_{k}.csv")
        m.to_csv(masked[str(k)], index=False)
    clean.to_csv(work / "oracle.csv", index=False)
    jobs = [{"budget": k, "seed": s, "mode": "fixed"} for s in args.fixed_seeds for k in args.budgets]
    jobs += [{"budget": 20, "seed": s, "mode": "own"} for s in SEEDS]
    cfg, out = work / "raha.json", work / "raha_out.json"
    cfg.write_text(json.dumps({"name": table, "dirty": str(work / "dirty.csv"), "masked_clean": masked, "jobs": jobs,
                               "oracle": str(work / "oracle.csv"), "label_order": {str(k): order[:k] for k in args.budgets}, "baran_seconds": args.baran_seconds}))
    out.unlink(missing_ok=True)
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    state = call([args.raha_python, WORKERS / "raha_baran.py", cfg, out], args.raha_timeout, work / "raha.log", {"TMPDIR": str(tmp)})
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {(r["budget"], r["seed"], r["mode"]) for r in done}
    return done + [{**j, "status": state if state != "ok" else "failed: no result"} for j in jobs if (j["budget"], j["seed"], j["mode"]) not in seen]


def scored(base: dict, res: dict | None) -> dict:
    return {**base, **({m: res[m] for m in METRICS} if res else {m: "" for m in METRICS})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="data/splits/core")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tables", nargs="*", default=CORE)
    ap.add_argument("--methods", nargs="*", default=["holoclean", "dboost", "raha"])
    ap.add_argument("--run-id", default="m5_baselines")
    ap.add_argument("--raha-python", required=True)
    ap.add_argument("--hc-python")
    ap.add_argument("--hc-src")
    ap.add_argument("--hc-socket")
    ap.add_argument("--hc-timeout", type=int, default=2400)
    ap.add_argument("--hc-db", default="holo")
    ap.add_argument("--budgets", nargs="*", type=int, default=BUDGETS)
    ap.add_argument("--fixed-seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--baran-seconds", type=int, default=0, help="time limit of one Baran run (0 = none)")
    ap.add_argument("--raha-timeout", type=int, default=7200)
    ap.add_argument("--dboost-setting", nargs="*", help="one fixed setting, for tables without a train split")
    ap.add_argument("--dboost-grid", choices=["raha34", "union36"], default="raha34", help="union36 adds M4's two other settings")
    ap.add_argument("--tables-out", default="docs/tables")
    args = ap.parse_args()
    out_dir, run = Path(args.tables_out), ROOT / "outputs" / "runs" / args.run_id
    rows, curve, grid_rows = [], [], []
    for table in args.tables:
        work = run / table
        work.mkdir(parents=True, exist_ok=True)
        dirty, cells = load_table(args.splits, table, args.split)
        dirty.to_csv(work / "dirty.csv", index=False)
        recs, n = cell_records(dirty, cells), len(dirty)
        clean = pd.DataFrame(cells["clean_value"].to_numpy().reshape(dirty.shape), columns=dirty.columns)
        order = label_order(table, n)
        ex20, ex50 = set(order[:20]), set(order)
        common = {"table": table, "rows": n, "rows_scored": n - 20}

        if "holoclean" in args.methods:
            r = run_holoclean(table, dirty, work, args)
            ok = r["status"] == "completed"
            note = f"{r.get('constraints', 0)} constraints mined on the dirty {args.split} table"
            if ok:
                rep = {(a, b): v for a, b, v in r["repairs"]}
                rows.append(scored({"baseline": "B1_HoloClean", **common, "status": r["status"], "seconds": r["seconds"], "notes": note},
                                   score_flags(list(rep), recs, ex20, rep)))
                rows.append(scored({"baseline": "B1_HoloClean_noisy_cells", **common, "status": r["status"], "seconds": r["seconds"],
                                    "notes": note + "; detection only: every cell HoloClean marked noisy"},
                                   score_flags([tuple(x) for x in r["noisy"]], recs, ex20)))
            else:
                rows.append(scored({"baseline": "B1_HoloClean", **common, "status": r["status"], "seconds": r["seconds"], "notes": note}, None))

        if "dboost" in args.methods:
            if args.dboost_setting:
                best, f1s = args.dboost_setting, None
            else:
                td, tc = load_split(ROOT, [args.splits], table, "train")
                td.to_csv(work / "train_dirty.csv", index=False)
                trecs, cols = cell_records(td, tc), list(td.columns)
                f1s, grid = [], DBOOST_GRID + (M4_EXTRA if args.dboost_grid == "union36" else [])
                for g in run_dboost(work / "train_dirty.csv", "train", work, args, grid):
                    s = score_flags([(i, cols[j]) for i, j in g["cells"]], trecs)
                    f1s.append(s["f1"] if g["status"] == "completed" else -1)
                    grid_rows.append({"table": table, "setting": " ".join(g["setting"]), "status": g["status"],
                                      "train_precision": s["precision"], "train_recall": s["recall"], "train_f1": s["f1"]})
                best = grid[int(np.argmax(f1s))]
            g = run_dboost(work / "dirty.csv", args.split, work, args, [best])[0]
            cols = list(dirty.columns)
            res = score_flags([(i, cols[j]) for i, j in g["cells"]], recs, ex20) if g["status"] == "completed" else None
            note = "setting " + " ".join(best) + (f", train F1 {max(f1s):.4f}, best of {len(f1s)}" if f1s else ", fixed in advance")
            rows.append(scored({"baseline": "B2_dBoost" + ("_union36" if args.dboost_grid == "union36" else ""), **common, "status": g["status"], "seconds": g["seconds"], "notes": note}, res))

        if "raha" in args.methods:
            for r in run_raha(table, dirty, clean, work, args):
                ok = r["status"] == "completed"
                flags = [tuple(x) for x in r.get("flags", [])]
                rep = {(a, b): v for a, b, v in r.get("repairs", [])}
                secs = round(r.get("strategy_seconds", 0) + r.get("detect_seconds", 0) + r.get("baran_seconds", 0), 1) if ok else None
                if r["mode"] == "fixed" and r["budget"] == 20 and r["seed"] == 0:
                    note = (r["baran_status"] + "; " if "baran_status" in r else "") + "strategies run/with output: " + ", ".join(f"{a} {x}/{y}" for a, (x, y) in sorted(r.get("strategies", {}).items()))
                    rows.append(scored({"baseline": "B3_Raha_Baran", **common, "status": r["status"], "seconds": secs, "notes": note},
                                       score_flags(flags, recs, ex20, rep) if ok else None))
                for mask, ex in (("50 rows" if r["mode"] == "fixed" else "", ex50), ("20 rows", ex20)):
                    if not mask or (mask == "20 rows" and r["budget"] != 20) or (mask == "50 rows" and 50 not in args.budgets):
                        continue
                    ex = ex | set(r.get("labelled_rows", [])) if r["mode"] == "own" else ex
                    curve.append(scored({"table": table, "budget": r["budget"], "labelling": r["mode"], "worker_seed": r["seed"],
                                         "left_out": mask + (" + own labelled rows" if r["mode"] == "own" else ""),
                                         "rows_scored": n - len(ex), "status": r["status"], "seconds": secs},
                                        score_flags(flags, recs, ex, rep) if ok else None))
        for name, data in (("m5_baselines_full", rows), ("m5_raha_label_budgets", curve), ("m5_dboost_grid", grid_rows)):
            if data:
                pd.DataFrame(data).to_csv(out_dir / f"{name}.csv", index=False)
        print(table, "done", [(r["baseline"], r["status"][:60], r["f1"]) for r in rows if r["table"] == table], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
