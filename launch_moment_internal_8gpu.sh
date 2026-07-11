#!/usr/bin/env bash
set -euo pipefail

# launch_moment_internal_8gpu.sh
#
# Usage:
#   bash launch_moment_internal_8gpu.sh APAVA linear
#   bash launch_moment_internal_8gpu.sh SleepEDF lora
#   bash launch_moment_internal_8gpu.sh REFED full
#
# Environment overrides:
#   GPUS="0 1 2 3" BATCH_SIZE=16 NUM_WORKERS=0 bash launch_moment_internal_8gpu.sh REFED full

DATASET="${1:-}"
BASELINE_MODE="${2:-}"

if [[ -z "${DATASET}" || -z "${BASELINE_MODE}" ]]; then
  echo "Usage: bash launch_moment_internal_8gpu.sh APAVA|SleepEDF|REFED linear|full|lora"
  exit 2
fi

EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-2025}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUT_ROOT="${OUT_ROOT:-./results_moment_internal_parallel}"
MODEL_PATH="${MODEL_PATH:-./MOMENT-1-small-hf}"
LOG_ROOT="${LOG_ROOT:-./logs/moment_internal_parallel}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

fold_count() {
  case "$1" in
    APAVA) echo 5 ;;
    SleepEDF) echo 20 ;;
    REFED) echo 32 ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

total_folds="$(fold_count "${DATASET}")"
read -r -a gpu_arr <<< "${GPUS}"
n_gpu="${#gpu_arr[@]}"

echo "======================================================================"
echo "Launching MOMENT-${BASELINE_MODE} | dataset=${DATASET} | folds=${total_folds} | GPUs=${GPUS}"
echo "epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "======================================================================"

pid_file="${LOG_ROOT}/${DATASET}_${BASELINE_MODE}_pids.txt"
: > "${pid_file}"

for idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[$idx]}"
  start=$(( idx * total_folds / n_gpu ))
  end=$(( (idx + 1) * total_folds / n_gpu ))

  if [[ "${start}" -ge "${end}" ]]; then
    echo "Skip GPU ${gpu}: empty fold range ${start}:${end}"
    continue
  fi

  out_dir="${OUT_ROOT}/${DATASET}/${BASELINE_MODE}/gpu${gpu}_fold${start}_${end}"
  log_file="${LOG_ROOT}/${DATASET}_${BASELINE_MODE}_gpu${gpu}_fold${start}_${end}.log"
  mkdir -p "${out_dir}"

  echo "[${DATASET} ${BASELINE_MODE}] GPU ${gpu}: folds ${start}:${end} -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" python moment_internal_baselines_multidataset.py \
    --dataset "${DATASET}" \
    --mode "${BASELINE_MODE}" \
    --model_path "${MODEL_PATH}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --start_fold "${start}" \
    --end_fold "${end}" \
    --output_dir "${out_dir}" \
    > "${log_file}" 2>&1 &

  pid=$!
  echo "${pid} ${DATASET} ${BASELINE_MODE} gpu${gpu} folds_${start}_${end} ${log_file}" | tee -a "${pid_file}"
  sleep 2
done

echo "Started jobs. PID file: ${pid_file}"
echo "Waiting..."

failed=0
while read -r pid rest; do
  [[ -z "${pid}" ]] && continue
  if wait "${pid}"; then
    echo "OK: ${pid} ${rest}"
  else
    echo "FAILED: ${pid} ${rest}"
    failed=1
  fi
done < "${pid_file}"

echo "All jobs finished. Merging results..."

python merge_moment_internal_parallel.py \
  --dataset "${DATASET}" \
  --mode "${BASELINE_MODE}" \
  --seed "${SEED}" \
  --root "${OUT_ROOT}/${DATASET}/${BASELINE_MODE}" \
  --expected_folds "${total_folds}" \
  --out_prefix "${OUT_ROOT}/${DATASET}/${BASELINE_MODE}/moment_${BASELINE_MODE}_${DATASET}_merged"

if [[ "${failed}" -ne 0 ]]; then
  echo "WARNING: one or more jobs failed. Check logs under ${LOG_ROOT}."
  exit 1
fi

echo "Done."
