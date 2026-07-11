#!/usr/bin/env bash
set -euo pipefail

# Run this in your project root.
# It clones the official CBraMod repo and installs its Python dependencies.

if [ ! -d "CBraMod" ]; then
  git clone https://github.com/wjq-learning/CBraMod.git CBraMod
fi

python -m pip install -r CBraMod/requirements.txt

cat <<'EOF'

Next:
1. Make sure the pretrained checkpoint exists:
   CBraMod/pretrained_weights/pretrained_weights.pth

2. Smoke test:
   CUDA_VISIBLE_DEVICES=0 python cbramod_baseline_multidataset.py \
     --dataset APAVA --mode head_only --end_fold 1 --epochs 2

EOF
