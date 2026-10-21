"""Zero-shot run of ONE model over the bake-off prompts. Greedy decoding, thinking off, VRAM capped.

Usage: python run_model.py <name> [--limit N]
Writes results/<name>_outputs.jsonl with raw text, P(is_error=true) read from the first answer token, timing and VRAM.
"""
import argparse, gc, json, os, sys, time
import torch

E = os.environ.get("E", "workdir")  # working folder with data/, results/ and weights/
MODELS = {  # name -> local dir under weights/
    "qwen3_4b": "Qwen3-4B-Instruct-2507",
    "qwen3_5_9b": "Qwen3.5-9B",
    "gemma4_e4b": "gemma-4-E4B-it",
}
VRAM_FRACTION = float(os.environ.get("VRAM_FRACTION", 0.85))   # cap on the share of GPU memory per process
PREFILL = '{"is_error":'

ap = argparse.ArgumentParser()
ap.add_argument("name", choices=MODELS)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=12)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
args = ap.parse_args()

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))
torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
from transformers import AutoTokenizer
import transformers

path = f"{E}/weights/{MODELS[args.name]}"
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model, loader = None, None
for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        continue
    try:
        model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        loader = cls_name
        break
    except Exception as e:  # try the next loader
        print(f"[{cls_name}] failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
if model is None:
    sys.exit("could not load model")
model.eval()
print("loaded", args.name, "via", loader, "| params(B) =", round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
      "| VRAM after load (GiB) =", round(torch.cuda.memory_allocated() / 2**30, 1), flush=True)




def build(user):
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + PREFILL


def first_ids(strings):
    ids = set()
    for s in strings:
        t = tok.encode(s, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


TRUE_IDS, FALSE_IDS = first_ids([" true", "true", " True"]), first_ids([" false", "false", " False"])
print("true ids", TRUE_IDS, "false ids", FALSE_IDS, flush=True)

for seed in args.seeds:
    recs = [json.loads(l) for l in open(f"{E}/data/bakeoff_prompts_s{seed}.jsonl")]
    texts = [build(r["prompt"]) for r in recs]
    order = sorted(range(len(recs)), key=lambda k: len(texts[k]))
    out_path = f"{E}/results/seed{seed}/{args.name}_outputs.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with open(out_path, "w") as f:
        for b in range(0, len(order), args.bs):
            ids = order[b: b + args.bs]
            enc = tok([texts[k] for k in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            tb = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=200, do_sample=False, output_logits=True, return_dict_in_generate=True,
                                     pad_token_id=tok.pad_token_id)
            lg = gen.logits[0].float()                       # logits for the first answer token (true/false)
            lp = torch.log_softmax(lg, dim=-1)
            pt = torch.logsumexp(lp[:, TRUE_IDS], dim=-1)
            pf = torch.logsumexp(lp[:, FALSE_IDS], dim=-1)
            p_true = torch.sigmoid(pt - pf).cpu().tolist()
            mass = (pt.exp() + pf.exp()).cpu().tolist()
            new = gen.sequences[:, enc["input_ids"].shape[1]:]
            dec = tok.batch_decode(new, skip_special_tokens=True)
            el = time.time() - tb
            for k, d, p, m in zip(ids, dec, p_true, mass):
                f.write(json.dumps({"id": recs[k]["id"], "text": d, "p_true": p, "tf_mass": m, "prompt_tokens": int(enc["attention_mask"][ids.index(k)].sum())}) + "\n")
            f.flush()
            if (b // args.bs) % 10 == 0:
                print(f"{b + len(ids)}/{len(order)} done, {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    total = time.time() - t0
    meta = {"model": args.name, "loader": loader, "n": len(recs), "seconds": total, "sec_per_cell": total / len(recs),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "params_b": sum(p.numel() for p in model.parameters()) / 1e9, "transformers": transformers.__version__, "torch": torch.__version__}
    json.dump(meta, open(f"{E}/results/seed{seed}/{args.name}_meta.json", "w"), indent=1)
    print("DONE", json.dumps(meta), flush=True)
