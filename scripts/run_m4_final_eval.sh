#!/usr/bin/env bash
# M4 final evaluation: the frozen configuration (configs/m4_final_pipeline.yaml), run on the test splits and on
# the full validation splits, then scored on the same cells as the baselines (Raha's 20 labelled rows left out).
# The configuration was chosen on validation data (the M3 subset and the validation splits).
# Usage: MODEL=<gemma-4-E4B-it> ADAPTER=<m4_f1_full>/adapter PY=<python> OUT=<runs folder> bash scripts/run_m4_final_eval.sh
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:?set MODEL}; ADAPTER=${ADAPTER:?set ADAPTER}; PY=${PY:-python}; OUT=${OUT:-outputs/runs}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FINAL="--d7-format-mix --d5-entity-consensus --evidence-threshold 0.5 --skip-missing-keys --weighted-majority
       --repair-conf 0.9 --validate"

run() {  # run id, split, tables...
  local id=$1 split=$2; shift 2
  $PY scripts/run_pipeline.py --model "$MODEL" --adapter "$ADAPTER" $FINAL --memory-fraction "${MEM:-0.22}" \
      --source data/splits/core --split "$split" --out "$OUT/$id" --tables "$@" > "$OUT/$id.log" 2>&1 \
  && $PY scripts/score_like_baselines.py "$OUT/$id" --source data/splits/core --split "$split" \
      --labelled data/labelled --out "$OUT/$id/score_like_baselines.csv" >> "$OUT/$id.log" 2>&1 \
  && echo "done $id" || echo "FAILED $id (see $OUT/$id.log)"
}

# the four runs fit on the GPU together (each capped by --memory-fraction)
run m4_final2_test_hfb test hospital flights beers &
run m4_final_test_rayyan test rayyan &
run m4_final_test_movies test movies &
run m4_final2_val_all val hospital flights beers &
wait
$PY scripts/build_test_vs_baselines.py --runs "$OUT/m4_final2_test_hfb" "$OUT/m4_final_test_rayyan" \
    "$OUT/m4_final_test_movies" --out docs/tables/m4_test_vs_baselines.csv
echo "final evaluation finished"
