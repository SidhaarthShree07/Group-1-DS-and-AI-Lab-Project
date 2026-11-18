#!/usr/bin/env bash
# M4 hyper-parameter screening runs (short runs, same training sample, same validation sample).
# Every run: 6,400 training examples (seed 42 unless stated), validation loss on 256 examples every 50 steps.
# Usage: MODEL=<path to gemma-4-E4B-it> PY=<python> OUT=<runs folder> bash scripts/run_m4_screening.sh [run ...]
set -u
MODEL=${MODEL:?set MODEL}; PY=${PY:-python}; OUT=${OUT:-outputs/runs}
COMMON="--model $MODEL --n-train 6400 --n-val 256 --eval-steps 50 --save-steps 1000 --memory-fraction ${MEM:-0.45}"

declare -A RUNS=(
  [m4_h0_ref]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200"
  [m4_h1_lr1e-4]="--lr 1e-4 --rank 16 --alpha 32 --max-steps 200"
  [m4_h2_lr4e-4]="--lr 4e-4 --rank 16 --alpha 32 --max-steps 200"
  [m4_h3_r8]="--lr 2e-4 --rank 8 --alpha 16 --max-steps 200"
  [m4_h4_r32]="--lr 2e-4 --rank 32 --alpha 64 --max-steps 200"
  [m4_h5_batch16]="--lr 2e-4 --rank 16 --alpha 32 --accum 4 --max-steps 400"
  [m4_h6_constant]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200 --schedule constant"
  [m4_h7_wd0.01]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200 --weight-decay 0.01"
  [m4_h8_dropout0.1]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200 --dropout 0.1"
  [m4_s1_seed7]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200 --seed 7"
  [m4_s2_seed123]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200 --seed 123"
  [m4_e5_no_synthetic]="--lr 2e-4 --rank 16 --alpha 32 --max-steps 200 --exclude-synthetic"
)
ORDER=(m4_h0_ref m4_h1_lr1e-4 m4_h2_lr4e-4 m4_h3_r8 m4_h4_r32 m4_e5_no_synthetic m4_s1_seed7 m4_s2_seed123
       m4_h5_batch16 m4_h6_constant m4_h7_wd0.01 m4_h8_dropout0.1)
[ $# -gt 0 ] && ORDER=("$@")

for run in "${ORDER[@]}"; do
  if [ -f "$OUT/$run/train_summary.json" ]; then echo "skip $run (done)"; continue; fi
  echo "=== $run $(date -u +%H:%M:%S)"
  $PY src/llm/train_lora.py $COMMON ${RUNS[$run]} --out "$OUT/$run" > "$OUT/$run.log" 2>&1 \
    && echo "done $run" || echo "FAILED $run (see $OUT/$run.log)"
  rm -rf "$OUT/$run/adapter_last"  # screening runs keep only the best adapter
done
echo "=== all screening runs finished $(date -u +%H:%M:%S)"
