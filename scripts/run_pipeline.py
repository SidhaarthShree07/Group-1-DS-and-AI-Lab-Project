#!/usr/bin/env python3
"""End-to-end ContextGuard pipeline on the M3 subset or any split.

Author: Sidhaarth Shree

  stage 0  rules proposed by the model, checked on the dirty table (optional; E6)   src/detectors/llm_rules.py
  stage 1  cheap detectors D1 to D6, optionally D7 and the D5 extension -> candidates src/detectors/
  stage 2  one prompt per candidate cell from the DIRTY table                         src/profiling/context.py
  stage 3  Gemma 4 E4B-it (base, or base + LoRA adapter) -> JSON answer + P(error)    src/llm/infer.py
  stage 4  parse, gate on the raw P(error) (+ repair confidence)                    src/evaluation/
           -> flag / repair / review decisions and the change log
  stage 4b (--validate, added in M4) every repair the gate accepts is checked against independent rules before
           it is applied; a rejected or unverifiable repair goes to review   src/validation/repair_validator.py
  scoring  every cell of the tables, at its natural error rate                       src/evaluation/metrics.py

Usage:
  python scripts/run_pipeline.py --model <path> --out outputs/runs/<run-id> [--adapter <dir>]
         [--tables hospital flights beers] [--source data/subsets/m3] [--split val] [--blocks profile dependencies similar]
  final M4 configuration: configs/m4_final_pipeline.yaml
Run outputs are not committed; summary tables are copied to docs/tables/.
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
from src.detectors import build_candidates, d2_fire_gate, m5_detectors  # noqa: E402
from src.detectors import d5_entity_consensus, d7_format_mix, llm_rules  # noqa: E402
from src.detectors.candidates import DEFAULT_DETECTORS  # noqa: E402
from src.evaluation.metrics import score  # noqa: E402
from src.llm.build_examples import META_COLUMNS  # noqa: E402
from src.llm.infer import Generator, to_prediction, write_jsonl  # noqa: E402
from src.profiling.context import ALL_BLOCKS, TableContext  # noqa: E402
from src.detectors.d5_entity_consensus import contested_columns  # noqa: E402
from src.validation.repair_validator import repair_state, validate_predictions  # noqa: E402

FLAG_THRESHOLD = 0.5    # on P(error); fixed before M3, kept in M4 (calibration is reported as a study only)
REPAIR_THRESHOLD = 0.8


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


def gate_decision(pred: dict, flag_threshold: float, repair_threshold: float, repair_conf: float = 0.0) -> str:
    """Stage 4 gate on the raw P(error). A repair also needs a repair value the model is sure of (repair_conf)."""
    if not pred["valid_json"]:
        return "review (invalid answer)"
    sure_of_value = pred.get("p_repair_min") is None or pred["p_repair_min"] >= repair_conf
    if pred["is_error"] and pred["p_error"] >= repair_threshold and pred["repair"] is not None and sure_of_value:
        return "repair"
    if pred["is_error"] and pred["p_error"] >= flag_threshold:
        return "flag for review"
    return "keep"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--quantize", choices=["8bit", "4bit"], default=None, help="load the model in 8-bit or 4-bit (X1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="data/subsets/m3")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tables", nargs="*", default=["hospital", "flights", "beers"])
    ap.add_argument("--blocks", nargs="*", default=list(ALL_BLOCKS), choices=ALL_BLOCKS)
    ap.add_argument("--stage0-rules", action="store_true", help="stage 0: rules proposed by the model (E6)")
    ap.add_argument("--stage0-kinds", nargs="*", default=list(llm_rules.FORMAT_AND_RANGE_KINDS),
                    choices=sorted(llm_rules.KINDS), help="rule kinds stage 0 asks for")
    ap.add_argument("--d7-format-mix", action="store_true", help="detector D7: numeric columns written in several formats")
    ap.add_argument("--d5-entity-consensus", action="store_true",
                    help="D5 extension: cells that disagree with the other rows of the same entity")
    ap.add_argument("--m5-detectors", action="store_true", help="the M5 detector set (X7); replaces the two flags above")
    ap.add_argument("--m5-composite-key", action="store_true", help="with --m5-detectors: add O1b (post-hoc, not the frozen set)")
    ap.add_argument("--evidence-threshold", type=float, default=0.9,
                    help="lowest dependency strength shown for the entity column (training used 0.9)")
    ap.add_argument("--skip-missing-keys", action="store_true", help="never group rows by an empty value")
    ap.add_argument("--weighted-majority", action="store_true", help="group majority weighted by row reliability")
    ap.add_argument("--repair-conf", type=float, default=0.0,
                    help="also require this minimum token probability of the repair value (0 = off)")
    ap.add_argument("--all-cells", action="store_true", help="send every cell to the model (no candidate filter, E3)")
    ap.add_argument("--contested-threshold", type=float, default=0.0,
                    help="send gate repairs to review in columns whose entity rows agree less than this (0 = off)")
    ap.add_argument("--close-values", action="store_true",
                    help="M5: show the model a frequent value within two edits of the cell (likely misspelling)")
    ap.add_argument("--blank-context", action="store_true",
                    help="M5: for an empty cell, add the closed value set of the column and the filled share of the row to the profile")
    ap.add_argument("--d2-fire-gate", type=float, default=0.0,
                    help="M5: do not send a cell flagged only by D2 when D2 flags more than this share of its column (0 = off)")
    ap.add_argument("--validate", action="store_true", help="stage 4b: validate every repair the gate accepts")
    ap.add_argument("--apply-unverified", action="store_true",
                    help="with --validate: apply gate repairs in columns no rule covers, marked 'not verified'")
    ap.add_argument("--flag-threshold", type=float, default=FLAG_THRESHOLD)
    ap.add_argument("--repair-threshold", type=float, default=REPAIR_THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--memory-fraction", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing, counts = {}, []
    t = time.time()
    gen = Generator(args.model, args.adapter, args.batch_size, memory_fraction=args.memory_fraction, quantize=args.quantize)
    timing["load model"] = time.time() - t

    all_cells, all_cands, preds, rules_out, changes, audit = [], [], [], {}, [], []
    for table in args.tables:
        dirty, cells = load_table(root / args.source, table, args.split)
        all_cells.append(cells)
        c = {"table": table, "rows": len(dirty), "cells": len(cells), "errors": int(cells["is_error"].eq("True").sum())}

        detectors = dict(DEFAULT_DETECTORS)
        if args.stage0_rules:  # stage 0
            t = time.time()
            text = gen.complete([llm_rules.build_prompt(table, dirty, args.stage0_kinds)])[0]
            checked = llm_rules.check_rules(llm_rules.parse_rules(text, set(args.stage0_kinds)), dirty)
            rules_out[table] = {"raw_answer": text, "rules": checked}
            detectors["stage0"] = llm_rules.make_detector(checked)
            c.update(rules_proposed=len(checked), rules_kept=sum(r["kept"] for r in checked))
            timing[f"stage 0 {table}"] = time.time() - t

        t = time.time()  # stage 1
        if args.d5_entity_consensus:
            detectors["d5c"] = lambda df, name=table: d5_entity_consensus.detect(df, name)
        if args.d7_format_mix:
            detectors["d7"] = d7_format_mix.detect
        if args.m5_detectors:
            detectors.update(m5_detectors(table, composite_key=args.m5_composite_key))
        cands = build_candidates(dirty, table=table, split=args.split, detectors=detectors)
        if args.d2_fire_gate > 0:
            cands = d2_fire_gate.apply(cands, args.d2_fire_gate)
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
        ctx = TableContext(table, dirty, evidence_threshold=args.evidence_threshold,
                           skip_missing_keys=args.skip_missing_keys, weighted_majority=args.weighted_majority,
                           close_values=args.close_values, blank_context=args.blank_context)
        prompts = [ctx.prompt(int(r), col, blocks=args.blocks) for r, col in chosen]
        timing[f"stage 2 {table}"] = time.time() - t

        t = time.time()  # stage 3
        print(f"{table}: {len(prompts)} candidates sent to the model", flush=True)
        answers = gen.answer(prompts, label=table) if prompts else []
        timing[f"stage 3 {table}"] = time.time() - t
        table_preds = []
        for (r, col), a in zip(chosen, answers):
            meta = {"id": f"{table}:{args.split}:{r}:{col}", "table": table, "split": args.split, "row": int(r), "column": col}
            p = to_prediction(meta, a)
            p["decision"] = gate_decision(p, args.flag_threshold, args.repair_threshold, args.repair_conf)  # stage 4
            p["old_value"] = dirty.at[int(r), col]
            table_preds.append(p)
        if args.contested_threshold > 0:  # no automatic repair where the sources disagree (abstain, review instead)
            contested = contested_columns(dirty, args.contested_threshold, table)
            for p in table_preds:
                if p["decision"] == "repair" and p["column"] in contested:
                    p["decision"] = "review (contested column: sources disagree)"
        if args.validate:  # stage 4b
            t = time.time()
            for res in validate_predictions(table, dirty, table_preds, apply_unverified=args.apply_unverified):
                audit.append({"table": table, **res.as_dict()})
            timing[f"stage 4b {table}"] = time.time() - t
        for p in table_preds:  # the change log keeps the old value of every applied repair, so it can be reverted
            p.setdefault("state", repair_state(p["decision"], (p.get("validation") or {}).get("outcome")))
            if p["decision"] == "repair" and str(p["repair"]).strip() != str(p["old_value"]).strip():
                changes.append({"table": table, "row": p["row"], "column": p["column"], "old": p["old_value"],
                                "new": p["repair"], "p_error": round(p["p_error"], 4), "evidence": p["evidence"],
                                "state": p["state"], "validation": (p.get("validation") or {}).get("reason", "not validated")})
        preds.extend(table_preds)
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
    if audit:
        with open(out / "validation_log.jsonl", "w", encoding="utf-8") as f:
            for x in audit:
                f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
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
    run = {"args": vars(args), "flag_threshold": args.flag_threshold, "repair_threshold": args.repair_threshold,
           "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "pandas": pd.__version__,
           "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print("DONE", json.dumps({"detection": keep["detection"], "candidates": keep["candidates"]}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
