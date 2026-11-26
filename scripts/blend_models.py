#!/usr/bin/env python3
"""Average the weights of several merged checkpoints into one model (M5 final model).

Author: Sidhaarth Shree

The M4 model and the two M5 retrains are fine-tunes of the same base model, so their tensors line up one to one.
Each tensor of the output is the mean of the same tensor in every input (equal weights), computed in float32 and
saved in the input dtype. A tensor that is identical in all inputs (everything the LoRA adapters did not touch) is
copied as it is. The result is one ordinary checkpoint: same size and same speed as any single input, so inference
needs one model call per cell. Configuration and tokenizer files are copied from the first input.

Usage: python scripts/blend_models.py <out_dir> <model_dir> <model_dir> [<model_dir> ...]
"""
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    out, sources = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    out.mkdir(parents=True, exist_ok=True)
    for f in sources[0].iterdir():  # config, tokenizer, chat template: taken from the first model
        if f.name != "model.safetensors":
            shutil.copy(f, out / f.name)

    handles = [safe_open(str(s / "model.safetensors"), "pt") for s in sources]
    names = list(handles[0].keys())
    if any(set(h.keys()) != set(names) for h in handles):
        raise SystemExit("the checkpoints do not have the same tensors, so they cannot be averaged")

    blended, averaged, start = {}, 0, time.time()
    for n, name in enumerate(names):
        tensors = [h.get_tensor(name) for h in handles]
        if all(torch.equal(tensors[0], t) for t in tensors[1:]):
            blended[name] = tensors[0]
        else:
            blended[name] = (sum(t.float() for t in tensors) / len(tensors)).to(tensors[0].dtype)
            averaged += 1
        if n % 200 == 0:
            print(f"progress {n}/{len(names)} tensors, {averaged} averaged, {(time.time() - start) / 60:.1f} min", flush=True)
    save_file(blended, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"done: {len(names)} tensors, {averaged} averaged, written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
