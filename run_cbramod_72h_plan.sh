#!/usr/bin/env bash
set -euo pipefail

# Copy this script into your project root after installing CBraMod.
# Edit CUDA_VISIBLE_DEVICES according to your server.

# Gate 1: smoke test, should finish quickly.
CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
  --dataset APAVA --mode head_only --end_fold 1 --epochs 2 \
  --output_dir ./results_cbramod_smoke

# Gate 2: APAVA full fine-tuning baseline.
CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
  --dataset APAVA --mode full --epochs 30 \
  --output_dir ./results_cbramod

# Gate 3: if APAVA is normal, run LOSO datasets. Parallelize manually if possible.
CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
  --dataset SleepEDF --mode full --epochs 30 \
  --output_dir ./results_cbramod

CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
  --dataset REFED --mode full --epochs 30 \
  --output_dir ./results_cbramod

python aggregate_pkl_results.py --glob './results_cbramod/*.pkl' --out ./results_cbramod/cbramod_summary.csv
