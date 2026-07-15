# StylePrior-MOMENT

Code and experiment records for **StylePrior-MOMENT: Style-Aware Adaptation of Time-Series Foundation Models for Cross-Subject EEG Classification**.

StylePrior-MOMENT studies how a general time-series foundation model can be adapted to EEG recorded from unseen subjects. The source-stage method freezes MOMENT-1-small and trains a Style-Adaptive Mixture-of-Experts (SA-MoE) module together with the classifier. The repository also includes Spatio-Temporal Style Adaptation (STSA), an optional batch-wise test-time procedure that updates a small target style adapter on an unlabeled subject stream.

The experimental evidence supports SA-MoE as the main adaptation mechanism. STSA is not treated as a universally improving component: it helps on APAVA, changes the class-balanced REFED metrics only slightly, and reduces performance on Sleep-EDF.

## Method overview

The framework has two stages:

1. **Source-stage adaptation.** MOMENT-1-small remains frozen. SA-MoE learns subject-conditioned feature-statistic alignment, sparse top-2 routing over five experts, and a hybrid-shared expert (HSE).
2. **Optional streaming adaptation.** For each unseen subject, STSA initializes a new `StyleAdaptor` and Adam optimizer, processes target batches in order, and updates only 16,384 style parameters using discrepancy-weighted pseudo-label loss.

The source model and the STSA extension are reported separately throughout the experiments.

## Repository layout

The main files are organized as follows:

```text
.
├── MoE_moment/
│   └── momentfm/models/
│       ├── SS_MOMENT.py
│       └── layers/
│           ├── SA_MoE.py
│           └── SA_MoE_components.py
├── two_stage_training_multidataset.py
├── moment_internal_baselines_multidataset.py
├── ablation_sleepedf.py
├── count_trainable_params.py
├── eegnet_baseline.py
├── graphsleepnet_baseline.py
├── salientsleepnet_baseline.py
├── mmcnn_baseline.py
├── bimtta_baseline.py
├── cbramod_baseline_multidataset.py
├── preprocessing.py
├── preprocessing_sleepedf.py
├── preprocessing_refed.py
├── download.py
├── collect_metrics_table.py
├── aggregate_pkl_results.py
├── audit_paper_consistency.py
├── run_sleepedf_baseline_shard.py
├── merge_sleepedf_baseline_shards.py
├── regenerate_main_paper_tables.py
├── inject_styleprior_metrics.py
├── merge_bimtta_results.py
├── merge_moment_internal_parallel.py
├── merge_cbramod_parallel.py
├── utils.py
├── requirements.txt
├── datasets/
├── results/
└── MOMENT-1-small/
```

`two_stage_training_multidataset.py` is the main entry point. It trains the source model and, when enabled, evaluates STSA using the same subject-disjoint fold definitions.

## Environment

The archived environment uses PyTorch 2.8.0 with CUDA 12.8. Create an isolated environment and install the pinned dependencies from the repository root:

```bash
conda create -n styleprior-moment python=3.10 -y
conda activate styleprior-moment
pip install -r requirements.txt
```

The pinned packages are:

```text
huggingface_hub==1.7.2
numpy==2.4.3
scikit_learn==1.8.0
torch==2.8.0+cu128
torcheeg==1.1.3
transformers==5.3.0
```

If `torch==2.8.0+cu128` is unavailable from the default package index, install the matching CUDA 12.8 wheel from the official PyTorch index before installing the remaining requirements.

## MOMENT checkpoint

Download the official `AutonLab/MOMENT-1-small` checkpoint with the included helper:

```bash
python download.py
```

The training scripts expect the downloaded checkpoint at:

```text
./MOMENT-1-small/
```

The tracked `MOMENT-1-small/config.json` records the model configuration. The downloaded weight files are not expected to be committed to Git. Change `model_path` in the training scripts if the checkpoint is stored elsewhere.

## Datasets

The experiments use three datasets:

| Dataset | Task | Subjects | Evaluation |
|---|---|---:|---|
| APAVA | Alzheimer's disease vs. healthy control | 23 | 5-fold cross-subject |
| Sleep-EDF | Wake/N1/N2/N3/REM sleep staging | 20 | 20-fold LOSO |
| REFED | EEG-only binary valence classification | 32 | 32-fold LOSO |

Preprocess Sleep-EDF from the original PSG EDF files:

```bash
python preprocessing_sleepedf.py \
  --data_dir ./raw_sleepedf \
  --output_dir ./datasets/sleepedf
```

This creates `datasets/sleepedf/sleepedf_all.pkl`, which is the path expected by the training scripts.

For REFED, retain the provider's `data/` and `annotations/` directories and run:

```bash
python preprocessing_refed.py \
  --data_dir ./datasets/REFED \
  --output_dir ./datasets/refed
```

This creates `datasets/refed/refed_all.pkl`. The resulting processed-data layout is:

```text
datasets/
├── sleepedf/
└── refed/
```

APAVA loading and five-fold subject splits are implemented in `preprocessing.py`. Processed APAVA folds are cached under `./cache` by default.

The datasets are not redistributed in this repository. Download them from their original providers and follow their access and usage conditions. The preprocessing code converts the inputs used by StylePrior-MOMENT to 16 channels, 256 Hz, and 256 samples per example. The cached Sleep-EDF array has shape `(n_epochs, 16, 256)`. The reported Sleep-EDF configurations of GraphSleepNet, SalientSleepNet, and MMCNN therefore use `seq_len=256`; no hidden padding or 1024-sample reconstruction is applied.

## Running StylePrior-MOMENT

Run the main experiment from the repository root:

```bash
python two_stage_training_multidataset.py --dataset APAVA
python two_stage_training_multidataset.py --dataset SleepEDF
python two_stage_training_multidataset.py --dataset REFED
```

To select a GPU for a single process:

```bash
CUDA_VISIBLE_DEVICES=0 python two_stage_training_multidataset.py --dataset SleepEDF
```

The aggregated files are written to:

```text
results/ss_moment_APAVA_seed2025.pkl
results/ss_moment_SleepEDF_seed2025.pkl
results/ss_moment_REFED_seed2025.pkl
```

Each file contains the source-model metrics, STSA metrics, and fold-level records:

```python
{
    "dataset": ...,
    "seed": 2025,
    "k_folds": ...,
    "completed_folds": ...,
    "baseline_metrics": ...,
    "tta_metrics": ...,
    "fold_results": ...,
}
```

Here, `baseline_metrics` refers to source-trained StylePrior-MOMENT without target updates. `tta_metrics` refers to the optional STSA evaluation.

## Test-time adaptation protocol

The reported STSA results follow a subject-wise streaming protocol:

- target batches use `shuffle=False` and remain in their original within-subject order;
- a new `StyleAdaptor` is initialized with `gamma = 1` and `beta = 0` for every held-out subject;
- Adam state is also reinitialized at the beginning of every subject;
- adapter state is retained only within the current subject and is never transferred across subjects or folds;
- the backbone, router, expert pool, HSE, and classifier remain frozen;
- only the 16,384 `StyleAdaptor` parameters receive gradients;
- one update is performed per target batch, followed by a second forward pass for the reported prediction;
- test labels are used only after prediction to calculate metrics.

For APAVA, a fold can contain more than one test subject. Each subject is adapted independently, after which their predictions are concatenated to calculate the fold-level metrics.

## Baselines

The repository contains the following comparison implementations:

- `moment_internal_baselines_multidataset.py`: MOMENT linear probing, LoRA, and full fine-tuning;
- `eegnet_baseline.py`;
- `graphsleepnet_baseline.py`;
- `salientsleepnet_baseline.py`;
- `mmcnn_baseline.py`;
- `bimtta_baseline.py`;
- `cbramod_baseline_multidataset.py`.

Several baseline scripts expose fold-range and GPU arguments for parallel execution. Run the corresponding help command before launching a large job:

```bash
python moment_internal_baselines_multidataset.py --help
python bimtta_baseline.py --help
python cbramod_baseline_multidataset.py --help
```

The same-backbone MOMENT baselines use `linear`, `lora`, or `full` mode. For example:

```bash
python moment_internal_baselines_multidataset.py \
  --dataset SleepEDF \
  --mode lora \
  --model_path ./MOMENT-1-small \
  --gpu_id 0 \
  --start_fold 0 \
  --end_fold 20 \
  --output_dir ./results_moment_internal
```

For CBraMod, place the external implementation under `./CBraMod` and provide its pretrained checkpoint when it is not stored at the default location:

```bash
python cbramod_baseline_multidataset.py \
  --dataset APAVA \
  --mode full \
  --cbramod_root ./CBraMod \
  --checkpoint ./CBraMod/pretrained_weights/pretrained_weights.pth \
  --gpu_id 0
```

BiM-TTA fold shards can be merged after parallel evaluation:

```bash
python merge_bimtta_results.py --dataset REFED --seed 2025
```

All reported comparisons use the same subject partitions. For Sleep-EDF, the corrected GraphSleepNet, SalientSleepNet, and MMCNN configurations use the same 256-sample cached excerpts as the other evaluated methods. Architecture-specific processing remains inside each model.


### Corrected Sleep-EDF baseline runs

The earlier 1024-sample configuration was inconsistent with the cached 256-sample Sleep-EDF inputs. MMCNN and GraphSleepNet were rerun over all 20 LOSO folds after setting `seq_len=256`. SalientSleepNet did not require a numerical rerun because its `seq_len` field is not used in the forward computation, but its configuration was also corrected to 256.

| Method | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| GraphSleepNet | 72.27 ± 8.27 | 55.97 ± 5.40 | 47.92 ± 5.67 |
| MMCNN | 78.29 ± 6.93 | 63.76 ± 5.79 | 55.28 ± 5.48 |

The canonical merged outputs are `results/graphsleepnet_SleepEDF_seed2025.pkl` and `results/mmcnn_SleepEDF_seed2025.pkl`.

Example shard and merge commands:

    CUDA_VISIBLE_DEVICES=0 python run_sleepedf_baseline_shard.py \
      --model mmcnn --start_fold 0 --end_fold 4

    python merge_sleepedf_baseline_shards.py \
      --input_dir ./results_parallel_seq256 \
      --result_dir ./results
## Ablation study

The publication-level ablation uses all 20 Sleep-EDF folds and separates source-stage components from STSA. The source-only comparison contains:

- complete SA-MoE;
- source model without Subject-Invariant Style Learning (SISL);
- source model without HSE.

STSA must remain disabled for all three source-only rows. The complete source model with STSA is reported as a separate deployment comparison. This distinction is important: an ablation that removes SISL or HSE while leaving STSA active does not isolate the source-stage contribution.

Use `ablation_sleepedf.py --help` to check the options provided by the checked-out version. The archived fold-level results should be treated as the canonical values used in the manuscript.

| Sleep-EDF variant | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Source w/o SISL | 82.79 ± 7.31 | 56.83 ± 4.69 | 53.97 ± 5.61 |
| Source w/o HSE | 83.43 ± 6.06 | 56.28 ± 3.44 | 53.75 ± 4.94 |
| Complete source model | 83.59 ± 6.46 | 57.13 ± 4.23 | 54.57 ± 5.83 |
| Complete source model + STSA | 82.77 ± 4.90 | 50.27 ± 5.99 | 49.81 ± 5.16 |

Neither source-component difference remains significant after Holm correction (all adjusted p ≥ 0.1289). The means support the complete SA-MoE design, but they do not establish a statistically separable contribution for SISL or HSE alone.

## Main results

Values below are mean ± standard deviation over subject-disjoint folds.

| Dataset | Method | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| APAVA | MOMENT-Linear | 57.96 ± 20.48 | 65.72 ± 14.64 | 47.94 ± 17.05 |
| APAVA | MOMENT-LoRA | 58.31 ± 18.82 | 65.51 ± 15.66 | 48.85 ± 14.33 |
| APAVA | MOMENT-Full | 55.14 ± 16.82 | 60.38 ± 14.31 | 46.80 ± 15.64 |
| APAVA | StylePrior-MOMENT (source) | 60.75 ± 19.03 | 67.77 ± 16.51 | 50.97 ± 12.63 |
| APAVA | StylePrior-MOMENT + STSA | 67.63 ± 16.58 | 71.09 ± 15.40 | 59.68 ± 16.49 |
| Sleep-EDF | MOMENT-Linear | 78.47 ± 7.53 | 48.19 ± 2.99 | 44.21 ± 4.92 |
| Sleep-EDF | MOMENT-LoRA | 81.31 ± 6.91 | 54.06 ± 4.52 | 50.41 ± 5.63 |
| Sleep-EDF | MOMENT-Full | 82.30 ± 7.60 | 54.26 ± 4.17 | 51.90 ± 5.49 |
| Sleep-EDF | StylePrior-MOMENT (source) | 83.59 ± 6.46 | 57.13 ± 4.23 | 54.57 ± 5.83 |
| Sleep-EDF | StylePrior-MOMENT + STSA | 82.77 ± 4.90 | 50.27 ± 5.99 | 49.81 ± 5.16 |
| REFED | MOMENT-Linear | 75.99 ± 5.61 | 50.03 ± 0.18 | 43.21 ± 1.95 |
| REFED | MOMENT-LoRA | 75.57 ± 5.68 | 50.00 ± 0.37 | 43.76 ± 1.91 |
| REFED | MOMENT-Full | 74.49 ± 5.85 | 49.99 ± 0.63 | 44.79 ± 2.62 |
| REFED | StylePrior-MOMENT (source) | 71.19 ± 6.12 | 50.71 ± 1.39 | 48.77 ± 2.71 |
| REFED | StylePrior-MOMENT + STSA | 68.66 ± 5.49 | 50.72 ± 1.67 | 49.67 ± 2.36 |

The controlled same-backbone comparison favors source-trained StylePrior-MOMENT on balanced accuracy and macro-F1. The broader ranking remains metric and task dependent: CBraMod is strongest in APAVA accuracy and macro-F1, while MMCNN leads the class-balanced Sleep-EDF metrics and REFED balanced accuracy.

Two-sided paired Wilcoxon signed-rank tests with Holm correction are reported for Sleep-EDF and REFED. APAVA is kept descriptive because it has only five fold-level observations. The source-only SISL and HSE changes on Sleep-EDF are numerically positive but do not remain significant after correction.

## Parameter accounting

| Dataset | Total parameters | Source-trainable parameters | Trainable ratio | STSA-updated parameters |
|---|---:|---:|---:|---:|
| APAVA | 46,067,655 | 10,730,247 | 23.292% | 16,384 |
| Sleep-EDF | 46,090,698 | 10,753,290 | 23.331% | 16,384 |
| REFED | 46,072,263 | 10,734,855 | 23.300% | 16,384 |

SA-MoE updates approximately 69.6% fewer source-training parameters than full MOMENT fine-tuning. It is not the smallest stored model and updates more parameters than LoRA. The term *parameter-efficient* therefore refers to the reduction relative to full MOMENT fine-tuning and to the restricted 16,384-parameter test-time update, not to minimum storage, runtime, or memory use.

Regenerate the parameter table with:

```bash
python count_trainable_params.py \
  --all_datasets \
  --model_path ./MOMENT-1-small \
  --out ./paper_tables/param_efficiency.csv
```

Regenerate the manuscript-facing metric tables after replacing result files:

    python regenerate_main_paper_tables.py
To inspect a directory of serialized experiment outputs without rerunning training:

```bash
python aggregate_pkl_results.py \
  --glob './results/**/*.pkl' \
  --out results_summary.csv
```

## Reproducibility settings

The manuscript results use:

| Setting | Value |
|---|---:|
| Random seed | 2025 |
| Source batch size | 32 |
| Maximum epochs | 30 |
| Optimizer | AdamW |
| Source learning rate | 5e-5 |
| Weight decay | 1e-5 |
| Early-stopping patience | 5 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor / patience | 0.5 / 3 |
| Validation criterion | Balanced accuracy |
| Number of experts / top-k | 5 / 2 |
| Subject / expert embedding dimension | 64 / 32 |
| Auxiliary routing-loss weight | 0.001 |
| STSA optimizer | Adam |
| STSA learning rate | 5e-4 |
| STSA batch size | 64 |
| STSA steps per batch | 1 |

The reported standard deviations measure variation across held-out-subject folds. All optimization runs use one random seed.

## Known limitations

- Training stochasticity has not been evaluated over multiple seeds.
- Runtime, peak memory, throughput, and energy use have not been benchmarked.
- The source-only component ablation is limited to Sleep-EDF.
- STSA is sensitive to the target stream and should not be assumed to improve every dataset.

## Citation

The manuscript is being prepared for journal submission. A complete BibTeX entry will be added after a public preprint or final publication becomes available.

## Contact

Yicong Lei
Data Science and Technology Program, School of Science and School of Engineering
The Hong Kong University of Science and Technology
Email: `yleiaq@connect.ust.hk`
