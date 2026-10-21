"""Score every model's outputs and pick the winner.

Score = 0.4 * detection macro-F1 (over the 5 tables) + 0.3 * repair accuracy on true errors + 0.3 * AUROC of P(is_error),
valid only if >= 90% of answers are parseable JSON. All individual metrics are reported too.
"""
import json, os, re, html, sys
import numpy as np
from sklearn.metrics import roc_auc_score

E = os.environ.get("E", "workdir")  # working folder with data/, results/ and weights/
NAMES = [n for n in ("qwen3_4b", "qwen3_5_9b", "gemma4_e4b") if os.path.exists(f"{E}/results/{n}_outputs.jsonl")]
TYPES = {"missing", "pattern", "rule", "outlier", "none"}
TYPE_MAP = {"missing_value": "missing", "pattern_violation": "pattern", "rule_violation": "rule", "outliers": "outlier", "outlier": "outlier"}
recs = {}
for l in open(f"{E}/data/bakeoff_prompts.jsonl"):
    r = json.loads(l); recs[r["id"]] = r


def norm(v):
    v = html.unescape(str(v)); v = re.sub("[\t\n ]+", " ", v); return v.strip("\t\n ")


def parse(text):
    raw = '{"is_error":' + text
    try:                                   # raw_decode reads one complete JSON object and ignores anything after it,
        d, _ = json.JSONDecoder().raw_decode(raw)   # and braces inside strings (e.g. "{NULL}") are handled correctly
        if isinstance(d, dict) and isinstance(d.get("is_error"), bool):
            return d
    except Exception:
        pass
    m = re.match(r'\{"is_error":\s*(true|false)', raw)
    return {"is_error": m.group(1) == "true", "_partial": True} if m else None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ece(p, y, bins=10):
    p, y = np.array(p), np.array(y); e = 0.0
    for b in range(bins):
        m = (p > b / bins) & (p <= (b + 1) / bins) if b else (p <= 1 / bins)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


results = {}
for name in NAMES:
    meta = json.load(open(f"{E}/results/{name}_meta.json")) if os.path.exists(f"{E}/results/{name}_meta.json") else {}
    rows = []
    for l in open(f"{E}/results/{name}_outputs.jsonl"):
        o = json.loads(l); r = recs[o["id"]]; d = parse(o["text"])
        rows.append((r, o, d))
    n = len(rows)
    valid_full = sum(1 for _, _, d in rows if d is not None and not d.get("_partial"))
    pred = [bool(d["is_error"]) if d else False for _, _, d in rows]
    y = [r["is_error"] for r, _, _ in rows]
    pt = [o["p_true"] for _, o, _ in rows]
    per_table, f1s = {}, []
    for t in sorted({r["table"] for r, _, _ in rows}):
        idx = [i for i, (r, _, _) in enumerate(rows) if r["table"] == t]
        tp = sum(1 for i in idx if pred[i] and y[i]); fp = sum(1 for i in idx if pred[i] and not y[i]); fn = sum(1 for i in idx if not pred[i] and y[i])
        p_, r_, f_ = prf(tp, fp, fn)
        per_table[t] = {"precision": p_, "recall": r_, "f1": f_, "n": len(idx)}
        f1s.append(f_)
    tp = sum(1 for i in range(n) if pred[i] and y[i]); fp = sum(1 for i in range(n) if pred[i] and not y[i]); fn = sum(1 for i in range(n) if not pred[i] and y[i])
    P, R, F = prf(tp, fp, fn)
    auroc = roc_auc_score(y, pt)
    n_err = sum(y)
    rep_ok = sum(1 for (r, _, d), yy in zip(rows, y) if yy and d and d.get("is_error") and norm(d.get("repair") or "") == norm(r["clean"]))
    rep_all = [(r, d) for r, _, d in rows if d and d.get("is_error") and d.get("repair") not in (None, "")]
    changed = [(r, d) for r, d in rep_all if norm(d["repair"]) != norm(r["dirty"])]
    false_rep = sum(1 for r, d in changed if norm(d["repair"]) != norm(r["clean"]))
    tp_rows = [(r, d) for (r, _, d), yy in zip(rows, y) if yy and d and d.get("is_error")]
    typed = [(r, d) for r, d in tp_rows if r["gold_type"]]
    type_ok = sum(1 for r, d in typed if TYPE_MAP.get(r["gold_type"], r["gold_type"]) == str(d.get("error_type", "")).lower())
    res = {
        "n": n, "json_valid_pct": 100 * valid_full / n, "precision": P, "recall": R, "f1_micro": F, "f1_macro_tables": float(np.mean(f1s)),
        "auroc": auroc, "ece_p_true": ece(pt, y), "repair_acc_on_errors": rep_ok / n_err,
        "false_repair_rate": (false_rep / len(changed)) if changed else 0.0, "n_repairs_applied": len(changed),
        "type_acc_on_detected": (type_ok / len(typed)) if typed else None, "n_typed": len(typed),
        "sec_per_cell": meta.get("sec_per_cell"), "peak_vram_gib": meta.get("peak_vram_gib"), "params_b": meta.get("params_b"),
        "per_table": per_table,
    }
    res["score"] = 0.4 * res["f1_macro_tables"] + 0.3 * res["repair_acc_on_errors"] + 0.3 * res["auroc"]
    res["eligible"] = res["json_valid_pct"] >= 90
    results[name] = res

winner = max((n for n in results if results[n]["eligible"]), key=lambda n: results[n]["score"], default=None)
json.dump({"winner": winner, "results": results}, open(f"{E}/results/bakeoff_results.json", "w"), indent=1)
hdr = ["model", "JSON ok %", "P", "R", "macro-F1", "AUROC", "repair acc", "false-repair", "type acc", "s/cell", "VRAM GiB", "SCORE"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for n, r in results.items():
    ta = f"{r['type_acc_on_detected']:.2f}" if r["type_acc_on_detected"] is not None else "-"
    print(f"| {n} | {r['json_valid_pct']:.1f} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1_macro_tables']:.3f} | {r['auroc']:.3f} | {r['repair_acc_on_errors']:.3f} | {r['false_repair_rate']:.2f} | {ta} | {r['sec_per_cell']:.2f} | {r['peak_vram_gib']:.1f} | **{r['score']:.3f}**{'' if r['eligible'] else ' (JSON<90%)'} |")
print("\nper-table F1:");
for n, r in results.items():
    print(" ", n, {t: round(v["f1"], 2) for t, v in r["per_table"].items()})
print("\nWINNER:", winner)
