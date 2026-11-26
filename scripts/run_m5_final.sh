#!/bin/bash
# M5 final pipeline: build the final model (weight average) and run it on the validation splits and the unseen tables.
#
# Author: Sidhaarth Shree
#
# Usage: bash scripts/run_m5_final.sh <weights_dir> <runs_dir> [<unseen_dir>] [<credit_dir>]
#   <weights_dir> holds the merged checkpoints: gemma-4-E4B-it-contextguard-merged (M4), ...-m5-2ep (retrain A),
#   ...-m5b (retrain B, with the blank-context lines). The average is written next to them as ...-blend.
#   <runs_dir> receives one folder per run. <unseen_dir> holds one folder per unseen table (default data/unseen), and
#   <credit_dir> the 30,000-row Credit subset (default data/unseen_credit30k), each with test.csv and test_cells.csv.
# --m5-detectors is the detector set of the X7 task (Aaradhya); the script needs his detector files in src/detectors/.
# The D2 fire-rate filter (--d2-fire-gate 0.5) is part of the final pipeline.
set -e
W=${1:?weights dir}; R=${2:?runs dir}; U=${3:-data/unseen}; C=${4:-data/unseen_credit30k}
M=$W/gemma-4-E4B-it-contextguard
python scripts/blend_models.py ${M}-blend ${M}-merged ${M}-m5-2ep ${M}-m5b

# Same settings for every run: M5 detectors, blank-context lines, gate 0.5 / 0.8, repair confidence 0.9,
# validator on, repairs in columns no rule covers applied and marked "not verified", contested-column cut 0.7.
OPTS="--blocks profile dependencies similar --m5-detectors --blank-context --d2-fire-gate 0.5 --evidence-threshold 0.5
  --skip-missing-keys --weighted-majority --repair-conf 0.9 --validate --apply-unverified --contested-threshold 0.7
  --flag-threshold 0.5 --repair-threshold 0.8 --batch-size 12"

# Validation splits of the five core tables: the data every model and pipeline choice is made on.
python scripts/run_pipeline.py --model ${M}-blend --tables hospital flights beers rayyan movies \
  --source data/splits/core --split val --out $R/final_val $OPTS

# Unseen tables, one run each.
for t in soccer german billionaire marketing restaurant; do
  python scripts/run_pipeline.py --model ${M}-blend --tables $t --source $U --split test --out $R/final_unseen_$t $OPTS
done
# Credit (30,000-row subset for the downstream test).
python scripts/run_pipeline.py --model ${M}-blend --tables credit --source $C --split test \
  --out $R/final_credit30k $OPTS
