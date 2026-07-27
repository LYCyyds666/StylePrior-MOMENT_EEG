# Corrected experiment rerun

This update is based on repository commit `ef085c3`.

## What changed

- Every validation subject embedding is initialized from the current mean of
  the source-training subject embeddings immediately before every validation
  pass.
- Every test subject embedding uses the same source-training mean rule.
- The main script supports zero-based, half-open fold shards.
- Source-model test evaluation accumulates per-layer routing probabilities,
  top-1 and top-k utilization, router entropy, and subject/class routing
  counts.
- Corrected outputs are written under `results_corrected/` so that archived
  results are not overwritten before validation.
- The renamed ablation option is `wo_scsn`; `wo_sisl` remains a compatibility
  alias.

## Preflight

Run from the repository root:

```bash
python -m py_compile \
  experiment_protocol.py \
  routing_analysis.py \
  two_stage_training_multidataset.py \
  ablation_sleepedf.py \
  merge_styleprior_shards.py \
  merge_ablation_sleepedf_shards.py \
  export_routing_analysis.py \
  validate_revised_experiment_code.py

bash -n launch_revised_experiments_10gpu.sh
python validate_revised_experiment_code.py
chmod +x launch_revised_experiments_10gpu.sh
mkdir -p results_corrected logs/revised_experiments routing_analysis
```

## Main experiments

Run one dataset at a time because Sleep-EDF and REFED use all ten GPUs:

```bash
bash launch_revised_experiments_10gpu.sh main APAVA
```

After all five APAVA jobs finish:

```bash
python merge_styleprior_shards.py \
  --dataset APAVA \
  --input_dir results_corrected \
  --output_dir results_corrected
```

Then run:

```bash
bash launch_revised_experiments_10gpu.sh main SleepEDF
```

After all ten Sleep-EDF jobs finish:

```bash
python merge_styleprior_shards.py \
  --dataset SleepEDF \
  --input_dir results_corrected \
  --output_dir results_corrected
```

Then run:

```bash
bash launch_revised_experiments_10gpu.sh main REFED
```

After all ten REFED jobs finish:

```bash
python merge_styleprior_shards.py \
  --dataset REFED \
  --input_dir results_corrected \
  --output_dir results_corrected
```

## Sleep-EDF source-only ablations

Run one variant at a time:

```bash
bash launch_revised_experiments_10gpu.sh ablation full
```

After completion:

```bash
python merge_ablation_sleepedf_shards.py \
  --variant full \
  --input_dir results_corrected \
  --output_dir results_corrected
```

Repeat for SCSN:

```bash
bash launch_revised_experiments_10gpu.sh ablation wo_scsn

python merge_ablation_sleepedf_shards.py \
  --variant wo_scsn \
  --input_dir results_corrected \
  --output_dir results_corrected
```

Repeat for HSE:

```bash
bash launch_revised_experiments_10gpu.sh ablation wo_hse

python merge_ablation_sleepedf_shards.py \
  --variant wo_hse \
  --input_dir results_corrected \
  --output_dir results_corrected
```

## Monitor

```bash
watch -n 20 nvidia-smi
```

Inspect logs without relying on wildcard ordering:

```bash
find logs/revised_experiments -maxdepth 1 -type f -name "*.log" -print
```

For a specific job:

```bash
tail -f logs/revised_experiments/main_SleepEDF_fold0-2.log
```

Check failures:

```bash
rg -n "Traceback|RuntimeError|CUDA out of memory|No successful" \
  logs/revised_experiments
```

## Export routing analysis

CSV export does not require an additional package:

```bash
for dataset in APAVA SleepEDF REFED; do
  python export_routing_analysis.py \
    --dataset "$dataset" \
    --results_dir results_corrected \
    --output_dir routing_analysis
done
```

To create publication-ready heatmap PNGs:

```bash
python -m pip install matplotlib

for dataset in APAVA SleepEDF REFED; do
  python export_routing_analysis.py \
    --dataset "$dataset" \
    --results_dir results_corrected \
    --output_dir routing_analysis \
    --plot
done
```

The exports contain:

- per-layer mean pre-top-k router probability;
- per-layer top-1 expert fraction;
- per-layer top-k assignment fraction;
- raw and normalized router entropy;
- layer-wise subject-expert routing fractions;
- layer-wise class-expert routing fractions;
- subject-expert and class-expert PNG heatmaps when `--plot` is enabled.

## Final validation before replacing archived results

```bash
python - <<'PY'
import pickle

expected = {"APAVA": 5, "SleepEDF": 20, "REFED": 32}
for dataset, number_of_folds in expected.items():
    path = f"results_corrected/ss_moment_{dataset}_seed2025.pkl"
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    assert result["completed_folds"] == number_of_folds, path
    assert len(result["fold_results"]) == number_of_folds, path
    assert result.get("routing_analysis", {}).get("layers"), path
    assert all(
        fold.get("test_metrics") is not None
        and fold.get("tta_metrics") is not None
        for fold in result["fold_results"]
    ), path

for variant in ("full", "wo_scsn", "wo_hse"):
    path = (
        "results_corrected/"
        f"ablation_sleepedf_{variant}_source_only_seed2025.pkl"
    )
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    assert result["completed_folds"] == 20, path
    assert len(result["fold_results"]) == 20, path
    assert result["source_only"] is True, path

print("PASS: all corrected main and ablation results are complete")
PY
```

Do not overwrite `results/` or regenerate the paper tables until this check
passes and the corrected metrics have been reviewed.
