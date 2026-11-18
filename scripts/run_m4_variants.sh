#!/usr/bin/env bash
# M4 inference changes on the validation subset, added one at a time (no retraining); the last run is the final
# configuration. Results: docs/tables/m4_inference_variants.csv.
# Usage: MODEL=<gemma-4-E4B-it> ADAPTER=<m4_f1_full>/adapter PY=<python> OUT=<runs folder> bash scripts/run_m4_variants.sh
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:?set MODEL}; ADAPTER=${ADAPTER:?set ADAPTER}; PY=${PY:-python}; OUT=${OUT:-outputs/runs}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONTEXT="--evidence-threshold 0.5 --skip-missing-keys --weighted-majority"

run() {  # run id, then pipeline options
  local id=$1; shift
  $PY scripts/run_pipeline.py --model "$MODEL" --adapter "$ADAPTER" --validate --memory-fraction "${MEM:-0.22}" \
      --out "$OUT/$id" "$@" > "$OUT/$id.log" 2>&1 \
  && $PY scripts/validate_repairs.py "$OUT/$id" >> "$OUT/$id.log" 2>&1 \
  && echo "done $id" || echo "FAILED $id (see $OUT/$id.log)"
}

run m4_v_rc --repair-conf 0.9 &
run m4_v_d7 --d7-format-mix &
run m4_v_d7_ctx --d7-format-mix $CONTEXT &
run m4_v_d7_ctx_rc --d7-format-mix $CONTEXT --repair-conf 0.9 &
wait
echo "all variants finished"
