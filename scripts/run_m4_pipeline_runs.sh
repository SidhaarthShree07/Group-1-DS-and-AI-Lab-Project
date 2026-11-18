#!/usr/bin/env bash
# M4 experiment runs on the validation subset, after the full training run (F1) has finished.
# Every run uses the repair validation layer (--validate) and is then compared with the existing pipeline
# (scripts/validate_repairs.py writes validation_comparison.csv into the run folder).
#   E1  fine-tuned against base model           m4_e1_ft, m4_e1_base
#   E2  context: row only, on the fine-tuned model  m4_e2_row_ft
#   E3  every cell against candidates only       m4_e3_all_ft
#   E6  candidates with stage 0 rules (format and range only)  m4_e6_d7_ft
# Usage: MODEL=<base model> ADAPTER=<run>/adapter PY=<python> OUT=<runs folder> bash scripts/run_m4_pipeline_runs.sh
set -u
MODEL=${MODEL:?set MODEL}; ADAPTER=${ADAPTER:?set ADAPTER}; PY=${PY:-python}; OUT=${OUT:-outputs/runs}
WAIT_FOR=${WAIT_FOR:-}   # optional file to wait for (the training summary), checked every 5 minutes
WAIT_PID=${WAIT_PID:-}   # optional training process: if it ends without writing WAIT_FOR, stop instead of waiting forever
COMMON="--model $MODEL --memory-fraction ${MEM:-0.4} --validate"

if [ -n "$WAIT_FOR" ]; then
  while [ ! -f "$WAIT_FOR" ]; do
    if [ -n "$WAIT_PID" ] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
      sleep 60  # the summary is written just before the process ends
      [ -f "$WAIT_FOR" ] && break
      echo "TRAINING FAILED: process $WAIT_PID ended without $WAIT_FOR $(date -u +%H:%M:%S)"; exit 1
    fi
    sleep 300
  done
  echo "found $WAIT_FOR $(date -u +%H:%M:%S)"
fi

run() {  # run id, then pipeline arguments
  local id=$1; shift
  if [ -f "$OUT/$id/validation_comparison.csv" ]; then echo "skip $id (done)"; return; fi
  echo "=== $id $(date -u +%H:%M:%S)"
  if $PY scripts/run_pipeline.py $COMMON --out "$OUT/$id" "$@" > "$OUT/$id.log" 2>&1; then
    $PY scripts/validate_repairs.py "$OUT/$id" >> "$OUT/$id.log" 2>&1 && echo "done $id" || echo "FAILED comparison $id"
  else
    echo "FAILED $id (see $OUT/$id.log)"
  fi
}

run m4_e1_ft     --adapter "$ADAPTER"
run m4_e1_base
run m4_e2_row_ft --adapter "$ADAPTER" --blocks
run m4_e6_d7_ft  --adapter "$ADAPTER" --stage0-rules
run m4_e3_all_ft --adapter "$ADAPTER" --all-cells
echo "=== all pipeline runs finished $(date -u +%H:%M:%S)"
