#!/usr/bin/env python3
"""End-to-end ContextGuard pipeline on a small subset (Milestone 3) or any split.

Author: Sidhaarth Shree

  stage 0  rules proposed by the model, checked on the dirty table (D7, optional)   src/detectors/llm_rules.py
  stage 1  cheap detectors D1 to D6 (+ D7) -> candidate cells                       src/detectors/
  stage 2  one prompt per candidate cell from the DIRTY table                         src/profiling/context.py
  stage 3  Gemma 4 E4B-it (base, or base + LoRA adapter) -> JSON answer + P(error)    src/llm/infer.py
  stage 4  parse, calibrate (identity until fitted on validation data in M4), gate  src/evaluation/
           -> flag / repair / review decisions and the change log
  scoring  every cell of the tables, at its natural error rate                       src/evaluation/metrics.py

Usage:
  python scripts/run_pipeline.py --model <path> --out outputs/runs/<run-id> [--d7] [--blocks profile dependencies similar]
         [--adapter <dir>] [--tables hospital flights beers] [--source data/subsets/m3] [--split val]
Run outputs are not committed; summary tables are copied to docs/tables/m3_*.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detectors import build_candidates  # noqa: E402
from src.detectors import llm_rules  # noqa: E402
from src.evaluation.metrics import score  # noqa: E402
from src.llm.build_examples import META_COLUMNS  # noqa: E402
from src.llm.infer import Generator, to_prediction, write_jsonl  # noqa: E402
from src.profiling.context import ALL_BLOCKS, TableContext  # noqa: E402

FLAG_THRESHOLD = 0.5    # placeholder until the gate is fitted on validation data (M4)
REPAIR_THRESHOLD = 0.8  # placeholder until the gate is fitted on validation data (M4)


def load_table(source: Path, table: str, split: str):
    """Dirty table (metadata columns dropped) and the cell truth with a `table` field (needed for scoring)."""
    dirty = pd.read_csv(source / table / f"{split}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    dirty = dirty[[c for c in dirty.columns if c.strip().lower() not in META_COLUMNS]].reset_index(drop=True)
    cells = pd.read_csv(source / table / f"{split}_cells.csv", dtype=str, keep_default_na=False)
    if len(cells) != len(dirty) * len(dirty.columns):
        raise ValueError(f"{table}: {len(cells)} cells for {len(dirty)} rows x {len(dirty.columns)} columns")
    # cell files are in row-major order and may carry the CLEAN header names (Hospital's differ from the dirty ones,
    # M2 Section 6.1); name every cell by the dirty column at its position, as the detectors and prompts do
    cells["column"] = [dirty.columns[i % len(dirty.columns)] for i in range(len(cells))]
    cells["row"] = [str(i // len(dirty.columns)) for i in range(len(cells))]
    cells["table"] = table
    return dirty, cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="data/subsets/m3")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tables", nargs="*", default=["hospital", "flights", "beers"])
    ap.add_argument("--blocks", nargs="*", default=list(ALL_BLOCKS), choices=ALL_BLOCKS)
    ap.add_argument("--d7", action="store_true", help="run stage 0 (rules proposed by the model)")
    ap.add_argument("--all-cells", action="store_true", help="send every cell to the model (no candidate filter, E3)")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--memory-fraction", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing, counts = {}, []
    t = time.time()
    gen = Generator(args.model, args.adapter, args.batch_size, memory_fraction=args.memory_fraction)
    timing["load model"] = time.time() - t

    all_cells, all_cands, preds, rules_out, changes = [], [], [], {}, []
    for table in args.tables:
        dirty, cells = load_table(root / args.source, table, args.split)
        all_cells.append(cells)
        c = {"table": table, "rows": len(dirty), "cells": len(cells), "errors": int(cells["is_error"].eq("True").sum())}

        d7 = None
        if args.d7:  # stage 0
            t = time.time()
            text = gen.complete([llm_rules.build_prompt(table, dirty)])[0]
            checked = llm_rules.check_rules(llm_rules.parse_rules(text), dirty)
            rules_out[table] = {"raw_answer": text, "rules": checked}
            d7 = llm_rules.make_detector(checked)
            c.update(rules_proposed=len(checked), rules_kept=sum(r["kept"] for r in checked))
            timing[f"stage 0 {table}"] = time.time() - t

        t = time.time()  # stage 1
        cands = build_candidates(dirty, table=table, split=args.split, d7_detector=d7)
        if args.all_cells:
            cands["is_candidate"] = True
        all_cands.append(cands)
        chosen = cands[cands["is_candidate"]][["row", "column"]].itertuples(index=False, name=None)
        chosen = list(chosen)
        c["candidates"] = len(chosen)
        truth = {(int(r), col): e == "True" for r, col, e in zip(cells["row"], cells["column"], cells["is_error"])}
        c["errors_in_candidates"] = sum(truth.get((int(r), col), False) for r, col in chosen)
        timing[f"stage 1 {table}"] = time.time() - t

        t = time.time()  # stage 2
        ctx = TableContext(table, dirty)
        prompts = [ctx.prompt(int(r), col, blocks=args.blocks) for r, col in chosen]
        timing[f"stage 2 {table}"] = time.time() - t

        t = time.time()  # stage 3
        answers = gen.answer(prompts) if prompts else []
        timing[f"stage 3 {table}"] = time.time() - t
        for (r, col), a in zip(chosen, answers):
            meta = {"id": f"{table}:{args.split}:{r}:{col}", "table": table, "split": args.split, "row": int(r), "column": col}
            p = to_prediction(meta, a)
            # stage 4: gate (calibration is the identity until it is fitted on validation data in M4)
            if not p["valid_json"]:
                p["decision"] = "review (invalid answer)"
            elif p["is_error"] and p["p_error"] >= REPAIR_THRESHOLD and p["repair"] is not None:
                p["decision"] = "repair"
                changes.append({"table": table, "row": r, "column": col, "old": dirty.at[int(r), col],
                                "new": p["repair"], "p_error": round(p["p_error"], 4), "evidence": p["evidence"]})
            elif p["is_error"] and p["p_error"] >= FLAG_THRESHOLD:
                p["decision"] = "flag for review"
            else:
                p["decision"] = "keep"
            preds.append(p)
        c["model_calls"] = len(prompts)
        c["invalid_answers"] = sum(1 for p in preds if p["table"] == table and not p["valid_json"])
        counts.append(c)
        print(json.dumps(c), flush=True)

    cells = pd.concat(all_cells, ignore_index=True)
    cands = pd.concat(all_cands, ignore_index=True)
    write_jsonl(out / "predictions.jsonl", preds)
    cands.to_csv(out / "candidates.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "change_log.csv", index=False)
    if rules_out:
        (out / "rules.json").write_text(json.dumps(rules_out, indent=1, ensure_ascii=False))
    t = time.time()
    metrics = score(out / "predictions.jsonl", cells.to_dict(orient="records"), cands.to_dict(orient="records"),
                    output_csv_path=out / "metrics_by_table_type.csv", n_bootstrap=200)
    timing["scoring"] = time.time() - t
    pd.DataFrame(counts).to_csv(out / "stage_counts.csv", index=False)
    pd.DataFrame([{"stage": k, "seconds": round(v, 2)} for k, v in timing.items()]).to_csv(out / "stage_times.csv", index=False)
    keep = {k: metrics[k] for k in ("n_cells", "detection", "macro_f1_tables", "per_table", "repair", "false_repair",
                                    "false_alarm", "pr_auc", "ece", "candidates", "format_artifacts_skipped")}
    keep["error_types_macro_f1"] = metrics["error_types"].get("macro_f1")
    (out / "metrics.json").write_text(json.dumps(keep, indent=1, default=float))
    import torch
    import transformers
    run = {"args": vars(args), "flag_threshold": FLAG_THRESHOLD, "repair_threshold": REPAIR_THRESHOLD,
           "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "pandas": pd.__version__,
           "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print("DONE", json.dumps({"detection": keep["detection"], "candidates": keep["candidates"]}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
