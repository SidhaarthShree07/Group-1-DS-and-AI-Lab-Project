#!/usr/bin/env bash
# Rebuild every Milestone 2 data file from the public sources, one step after the other.
# Stops at the first failing step (including a failing leakage check). Run from the repository root:
#   bash scripts/build_all.sh [--skip-download]
# Steps: download -> core tables -> core splits -> external tables and splits -> synthetic tables -> description and EDA
#        -> baseline files -> fine-tuning examples -> leakage checks -> tests.
set -euo pipefail
PY="${PYTHON:-python}"

if [[ "${1:-}" != "--skip-download" ]]; then
  $PY scripts/download_data.py
fi
$PY src/data/load_core.py
$PY src/data/split.py
$PY src/data/load_external.py
$PY src/data/synthetic.py
$PY scripts/generate_m2_outputs.py
$PY scripts/eda_analysis.py
$PY src/data/export_baselines.py
$PY src/llm/build_examples.py
$PY src/data/check_leakage.py
$PY -m pytest -q tests
echo "Milestone 2 data build finished: data/processed/examples/{train,val,test}.jsonl"
