"""Bootstrap uncertainty for the bake-off: resample the 1,500 test cells (stratified by table and error/correct),
recompute every metric and the composite score for each model on the SAME resample, report mean +- std and paired comparisons."""
import html, json, os, re
import numpy as np
from sklearn.metrics import roc_auc_score

E = os.environ.get("E", "workdir")  # working folder with data/, results/ and weights/
NAMES = ["qwen3_4b", "qwen3_5_9b", "gemma4_e4b"]
B, SEED = 2000, 0
recs = {}
for l in open(f"{E}/data/bakeoff_prompts.jsonl"):
    r = json.loads(l); recs[r["id"]] = r


def norm(v):
    v = html.unescape(str(v)); v = re.sub("[\t\n ]+", " ", v); return v.strip("\t\n ")


def parse(text):
    raw = '{"is_error":' + text
    try:
        d, _ = json.JSONDecoder().raw_decode(raw)
        if isinstance(d, dict) and isinstance(d.get("is_error"), bool):
            return d
    except Exception:
        pass
    m = re.match(r'\{"is_error":\s*(true|false)', raw)
    return {"is_error": m.group(1) == "true"} if m else None


ids = list(recs)
table = np.array([recs[i]["table"] for i in ids]); y = np.array([recs[i]["is_error"] for i in ids], dtype=bool)
D = {}
for n in NAMES:
    out = {}
    for l in open(f"{E}/results/{n}_outputs.jsonl"):
        o = json.loads(l); out[o["id"]] = o
    pred, pt, rep = np.zeros(len(ids), bool), np.zeros(len(ids)), np.zeros(len(ids), bool)
    for k, i in enumerate(ids):
        o = out[i]; d = parse(o["text"]); pt[k] = o["p_true"]
        if d:
            pred[k] = bool(d["is_error"])
            rep[k] = bool(y[k] and d["is_error"] and norm(d.get("repair") or "") == norm(recs[i]["clean"]))
    D[n] = (pred, pt, rep)


def metrics(idx, n):
    pred, pt, rep = D[n]
    f1s = []
    for t in np.unique(table):
        m = idx[table[idx] == t]
        tp = (pred[m] & y[m]).sum(); fp = (pred[m] & ~y[m]).sum(); fn = (~pred[m] & y[m]).sum()
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    f1 = float(np.mean(f1s)); auc = roc_auc_score(y[idx], pt[idx]); ra = rep[idx][y[idx]].mean()
    changed = pred[idx]
    return f1, ra, auc, 0.4 * f1 + 0.3 * ra + 0.3 * auc


rng = np.random.default_rng(SEED)
strata = [np.flatnonzero((table == t) & (y == v)) for t in np.unique(table) for v in (True, False)]
res = {n: [] for n in NAMES}
for _ in range(B):
    idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
    for n in NAMES:
        res[n].append(metrics(idx, n))
res = {n: np.array(v) for n, v in res.items()}
lines = ["| model | macro-F1 | repair acc | AUROC | SCORE (mean +- std, 95% interval) |", "|---|---|---|---|---|"]
for n in NAMES:
    a = res[n]; lo, hi = np.percentile(a[:, 3], [2.5, 97.5])
    lines.append(f"| {n} | {a[:,0].mean():.3f} +- {a[:,0].std():.3f} | {a[:,1].mean():.3f} +- {a[:,1].std():.3f} | {a[:,2].mean():.3f} +- {a[:,2].std():.3f} | {a[:,3].mean():.3f} +- {a[:,3].std():.3f}  [{lo:.3f}, {hi:.3f}] |")
g = res["gemma4_e4b"]
for other in ("qwen3_5_9b", "qwen3_4b"):
    diff = g[:, 3] - res[other][:, 3]
    lines.append(f"\nGemma - {other}: score diff {diff.mean():+.3f} (95% interval {np.percentile(diff,2.5):+.3f} to {np.percentile(diff,97.5):+.3f}); Gemma ahead in {100*(diff>0).mean():.1f}% of {B} resamples")
    for j, nm in enumerate(["macro-F1", "repair acc", "AUROC"]):
        d = g[:, j] - res[other][:, j]
        lines.append(f"   {nm}: {d.mean():+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}], Gemma ahead {100*(d>0).mean():.1f}%")
open(f"{E}/results/bootstrap_results.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
