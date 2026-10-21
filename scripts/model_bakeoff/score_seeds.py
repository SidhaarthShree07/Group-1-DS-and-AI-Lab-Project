"""Multi-seed bake-off scoring. For every sampling seed and model: the same metrics as score.py; then mean +- std (n = number of seeds)
and the winner by mean score. Score = 0.4*macro-F1 + 0.3*repair accuracy + 0.3*AUROC (valid if >= 90% JSON)."""
import html, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

E = os.environ.get("E", "workdir")  # working folder with data/, results/ and weights/
SEEDS = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
NAMES = ["qwen3_4b", "qwen3_5_9b", "gemma4_e4b"]


def norm(v):
    v = html.unescape(str(v)); v = re.sub("[\t\n ]+", " ", v); return v.strip("\t\n ")


def parse(text):
    raw = '{"is_error":' + text
    try:
        d, _ = json.JSONDecoder().raw_decode(raw)
        if isinstance(d, dict) and isinstance(d.get("is_error"), bool):
            return d, True
    except Exception:
        pass
    m = re.match(r'\{"is_error":\s*(true|false)', raw)
    return ({"is_error": m.group(1) == "true"}, False) if m else (None, False)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def one(seed, name):
    pf = f"{E}/data/bakeoff_prompts_s{seed}.jsonl"
    recs = {}
    for l in open(pf):
        r = json.loads(l); recs[r["id"]] = r
    rows, valid = [], 0
    for l in open(f"{E}/results/seed{seed}/{name}_outputs.jsonl"):
        o = json.loads(l); d, ok = parse(o["text"]); valid += ok; rows.append((recs[o["id"]], o, d))
    n = len(rows)
    y = np.array([r["is_error"] for r, _, _ in rows], bool)
    pred = np.array([bool(d["is_error"]) if d else False for _, _, d in rows])
    pt = np.array([o["p_true"] for _, o, _ in rows])
    tabs = np.array([r["table"] for r, _, _ in rows])
    f1s, per_table = [], {}
    for t in sorted(set(tabs)):
        m = tabs == t
        tp = (pred[m] & y[m]).sum(); fp = (pred[m] & ~y[m]).sum(); fn = (~pred[m] & y[m]).sum()
        per_table[t] = prf(tp, fp, fn)[2]; f1s.append(per_table[t])
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    P, R, F = prf(tp, fp, fn)
    rep_ok = sum(1 for (r, _, d), yy in zip(rows, y) if yy and d and d.get("is_error") and norm(d.get("repair") or "") == norm(r["clean"]))
    changed = [(r, d) for r, _, d in rows if d and d.get("is_error") and d.get("repair") not in (None, "") and norm(d["repair"]) != norm(r["dirty"])]
    false_rep = sum(1 for r, d in changed if norm(d["repair"]) != norm(r["clean"]))
    res = {"json_valid_pct": 100 * valid / n, "precision": P, "recall": R, "macro_f1": float(np.mean(f1s)),
           "auroc": roc_auc_score(y, pt), "repair_acc": rep_ok / y.sum(), "false_repair": false_rep / len(changed) if changed else 0.0,
           "per_table": per_table}
    res["score"] = 0.4 * res["macro_f1"] + 0.3 * res["repair_acc"] + 0.3 * res["auroc"]
    return res


R = {n: {s: one(s, n) for s in SEEDS} for n in NAMES}
KEYS = ["json_valid_pct", "precision", "recall", "macro_f1", "auroc", "repair_acc", "false_repair", "score"]
lines = [f"Seeds: {SEEDS} (each = a fresh draw of 1,500 cells; greedy decoding, so seeds only change which cells are tested)\n",
         "| model | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
agg = {}
for n in NAMES:
    agg[n] = {k: (float(np.mean([R[n][s][k] for s in SEEDS])), float(np.std([R[n][s][k] for s in SEEDS], ddof=1))) for k in KEYS}
    lines.append(f"| {n} | " + " | ".join(f"{agg[n][k][0]:.3f} +- {agg[n][k][1]:.3f}" for k in KEYS) + " |")
lines.append("\nScore per seed:")
lines.append("| seed | " + " | ".join(NAMES) + " | winner |"); lines.append("|---|" + "---|" * (len(NAMES) + 1))
wins = {n: 0 for n in NAMES}
for s in SEEDS:
    sc = {n: R[n][s]["score"] for n in NAMES}; w = max(sc, key=sc.get); wins[w] += 1
    lines.append(f"| {s} | " + " | ".join(f"{sc[n]:.3f}" for n in NAMES) + f" | {w} |")
lines.append("\nPer-table F1 (mean +- std over seeds):")
for n in NAMES:
    tabs = sorted(R[n][SEEDS[0]]["per_table"])
    lines.append(f"  {n}: " + ", ".join(f"{t} {np.mean([R[n][s]['per_table'][t] for s in SEEDS]):.2f}+-{np.std([R[n][s]['per_table'][t] for s in SEEDS], ddof=1):.2f}" for t in tabs))
g = "gemma4_e4b"
for o in ("qwen3_5_9b", "qwen3_4b"):
    d = np.array([R[g][s]["score"] - R[o][s]["score"] for s in SEEDS])
    lines.append(f"\nGemma - {o}: score diff per seed {np.round(d, 3).tolist()}, mean {d.mean():+.3f} +- {d.std(ddof=1):.3f}")
winner = max(NAMES, key=lambda n: agg[n]["score"][0] if agg[n]["json_valid_pct"][0] >= 90 else -1)
lines.append(f"\nSeed wins: {wins}\nWINNER (highest mean score): {winner}")
out = "\n".join(lines)
open(f"{E}/results/multiseed_results.md", "w").write(out + "\n")
json.dump({"seeds": SEEDS, "winner": winner, "mean_std": agg, "per_seed": {n: {str(s): R[n][s] for s in SEEDS} for n in NAMES}}, open(f"{E}/results/multiseed_results.json", "w"), indent=1)
print(out)
