#!/usr/bin/env python3
"""Ablation of a finished run (M5, experiment X14): remove one component at a time and rescore, with no model calls.

Author: Sidhaarth Shree

Removing a detector drops the candidates that only it found; removing the repair-confidence gate re-runs the gate and
the validation layer; the contested-column rule (X3) is added on top of the final decisions. Scored rows only (Raha's
20 labelled rows per table left out), as in the test comparison against the baselines.

Usage: python scripts/ablate_run.py <run dir> <repo root> <folder with <table>/test.csv> > ablation.csv
"""
import copy, csv, json, random, re, sys
import pandas as pd

run_dir, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from src.validation.repair_validator import validate_predictions  # noqa: E402
from src.detectors.d5_entity_consensus import contested_columns  # noqa: E402

norm = lambda s: re.sub(r"\s+", " ", str(s or "").strip())
cells = pd.read_csv(f"{run_dir}/cells.csv", dtype=str, keep_default_na=False)
cand = pd.read_csv(f"{run_dir}/candidates.csv", dtype=str, keep_default_na=False)
preds_all = [json.loads(l) for l in open(f"{run_dir}/predictions.jsonl")]
DET = [c for c in cand.columns if re.fullmatch(r"d\d+c?", c)]


def gate(p, repair_conf):
    if not p["valid_json"]:
        return "review (invalid answer)"
    sure = p.get("p_repair_min") is None or p["p_repair_min"] >= repair_conf
    if p["is_error"] and p["p_error"] >= 0.8 and p["repair"] is not None and sure:
        return "repair"
    return "flag for review" if p["is_error"] and p["p_error"] >= 0.5 else "keep"


rows = {}
for table, c in cells.groupby("table"):
    dirty = pd.read_csv(f"{sys.argv[3]}/{table}/test.csv", dtype=str, keep_default_na=False)
    n = len(dirty)
    lab = set(random.Random(f"42:{table}").sample(range(n), min(20, n)))
    truth = {(int(r), col): (d, cl, e == "True") for r, col, d, cl, e in
             zip(c["row"], c["column"], c["dirty_value"], c["clean_value"], c["is_error"]) if int(r) not in lab}
    ct = cand[cand["table"] == table]
    fired = {(int(r["row"]), r["column"]): {d for d in DET if float(r[d] or 0) > 0} for _, r in ct.iterrows()}
    preds = [p for p in preds_all if p["table"] == table]
    contested = contested_columns(dirty, 0.7, table)

    def score(name, keep_cell, decisions):
        tp = fp = fn = right = wrong = review = 0
        for k, (d, cl, y) in truth.items():
            p = decisions.get(k)
            if p is None or not keep_cell(k):
                fn += y
                continue
            flag = p["valid_json"] and p["is_error"]
            tp += flag and y; fp += flag and not y; fn += (not flag) and y
            dec = p["decision"]
            if dec == "repair" and norm(p["repair"]) != norm(d):
                if norm(p["repair"]) == norm(cl): right += 1
                else: wrong += 1
            elif dec.startswith("review") or dec == "flag for review":
                review += 1
        r = rows.setdefault(name, dict(tp=0, fp=0, fn=0, right=0, wrong=0, review=0))
        for key, v in dict(tp=tp, fp=fp, fn=fn, right=right, wrong=wrong, review=review).items():
            r[key] += v

    final = {(int(p["row"]), p["column"]): p for p in preds}
    everything = lambda k: True
    only = lambda det: (lambda k: fired.get(k, set()) != {det} if fired.get(k) else True)
    score("Final pipeline", everything, final)
    score("without D7 (format mix)", only("d7"), final)
    score("without D5 consensus", only("d5c"), final)
    score("without D7 and D5 consensus", lambda k: not (fired.get(k) and fired[k] <= {"d7", "d5c"}), final)
    no_val = {k: dict(p, decision=p.get("gate_decision", p["decision"])) for k, p in final.items()}
    score("without validation layer", everything, no_val)
    loose = copy.deepcopy(preds)
    for p in loose:
        p.pop("validation", None); p["decision"] = gate(p, 0.0); p.pop("gate_decision", None)
    no_rc_novalid = {(int(p["row"]), p["column"]): dict(p) for p in loose}
    score("without repair-confidence gate and validation", everything, no_rc_novalid)
    validate_predictions(table, dirty, loose)
    score("without repair-confidence gate", everything, {(int(p["row"]), p["column"]): p for p in loose})
    plus = {k: (dict(p, decision="review (contested column)") if p["decision"] == "repair" and k[1] in contested else p)
            for k, p in final.items()}
    score("final + contested-column rule (X3)", everything, plus)

w = csv.writer(sys.stdout)
w.writerow(["variant", "precision", "recall", "f1", "applied_right", "applied_wrong", "false_repair_rate", "to_review"])
for name, r in rows.items():
    p = r["tp"] / max(1, r["tp"] + r["fp"]); rc = r["tp"] / max(1, r["tp"] + r["fn"])
    w.writerow([name, round(p, 4), round(rc, 4), round(2 * p * rc / max(1e-9, p + rc), 4), r["right"], r["wrong"],
                round(r["wrong"] / max(1, r["right"] + r["wrong"]), 4), r["review"]])
