#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  bash launch_revised_experiments_10gpu.sh main APAVA"
  echo "  bash launch_revised_experiments_10gpu.sh main SleepEDF"
  echo "  bash launch_revised_experiments_10gpu.sh main REFED"
  echo "  bash launch_revised_experiments_10gpu.sh ablation full"
  echo "  bash launch_revised_experiments_10gpu.sh ablation wo_scsn"
  echo "  bash launch_revised_experiments_10gpu.sh ablation wo_hse"
  exit 1
fi

mode="$1"
target="${2:-}"
result_dir="./results_corrected"
log_dir="./logs/revised_experiments"
mkdir -p "$result_dir" "$log_dir"

launch_main() {
  local dataset="$1"
  local -a starts
  local -a ends

  case "$dataset" in
    APAVA)
      starts=(0 1 2 3 4)
      ends=(1 2 3 4 5)
      ;;
    SleepEDF)
      starts=(0 2 4 6 8 10 12 14 16 18)
      ends=(2 4 6 8 10 12 14 16 18 20)
      ;;
    REFED)
      starts=(0 4 8 11 14 17 20 23 26 29)
      ends=(4 8 11 14 17 20 23 26 29 32)
      ;;
    *)
      echo "Unknown dataset: $dataset"
      exit 2
      ;;
  esac

  for index in "${!starts[@]}"; do
    local gpu="$index"
    local start="${starts[$index]}"
    local end="${ends[$index]}"
    local log_path="$log_dir/main_${dataset}_fold${start}-${end}.log"
    echo "Launching ${dataset} folds ${start}:${end} on GPU ${gpu}"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
      python -u two_stage_training_multidataset.py \
        --dataset "$dataset" \
        --start_fold "$start" \
        --end_fold "$end" \
        --output_dir "$result_dir" \
      >"$log_path" 2>&1 &
  done
}

launch_ablation() {
  local variant="$1"
  case "$variant" in
    full|wo_scsn|wo_hse) ;;
    *)
      echo "Unknown source-only ablation variant: $variant"
      exit 2
      ;;
  esac

  local -a starts=(0 2 4 6 8 10 12 14 16 18)
  local -a ends=(2 4 6 8 10 12 14 16 18 20)
  for index in "${!starts[@]}"; do
    local gpu="$index"
    local start="${starts[$index]}"
    local end="${ends[$index]}"
    local log_path="$log_dir/ablation_${variant}_fold${start}-${end}.log"
    echo "Launching ${variant} folds ${start}:${end} on GPU ${gpu}"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
      python -u ablation_sleepedf.py \
        --variant "$variant" \
        --source_only \
        --start_fold "$start" \
        --end_fold "$end" \
        --output_dir "$result_dir" \
      >"$log_path" 2>&1 &
  done
}

case "$mode" in
  main)
    launch_main "$target"
    ;;
  ablation)
    launch_ablation "$target"
    ;;
  *)
    echo "Unknown mode: $mode"
    exit 2
    ;;
esac

echo
echo "Jobs launched. Monitor with:"
echo "  watch -n 20 nvidia-smi"
echo "  tail -f $log_dir/*.log"
