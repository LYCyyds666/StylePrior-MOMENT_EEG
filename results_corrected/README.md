# Corrected StylePrior-MOMENT results

These files were generated after correcting the unseen-subject embedding
protocol. Every validation or test subject embedding is initialized from the
current mean of the source-training subject embeddings immediately before the
corresponding evaluation pass.

## Canonical outputs

- `ss_moment_APAVA_seed2025.pkl`: five cross-subject folds.
- `ss_moment_SleepEDF_seed2025.pkl`: 20 LOSO folds.
- `ss_moment_REFED_seed2025.pkl`: 32 LOSO folds.
- `ablation_sleepedf_wo_scsn_source_only_seed2025.pkl`: 20 source-only LOSO
  folds without SCSN.
- `ablation_sleepedf_wo_hse_source_only_seed2025.pkl`: 20 source-only LOSO
  folds without HSE.
- `styleprior_corrected_statistics.csv`: paired Wilcoxon tests and Holm-adjusted
  p-values used in the revised manuscript.

Each main result file contains source metrics, STSA metrics, fold-level
records, and aggregated routing analysis. The raw datasets and model
checkpoints are not redistributed.

## Core results

Values are mean ± standard deviation over subject-disjoint folds.

| Dataset | Variant | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| APAVA | Source | 66.35 ± 19.08 | 71.52 ± 17.41 | 56.50 ± 16.14 |
| APAVA | Source + STSA | 67.61 ± 16.89 | 71.10 ± 15.88 | 59.67 ± 16.86 |
| Sleep-EDF | Source | 83.70 ± 5.74 | 56.94 ± 3.93 | 54.31 ± 5.12 |
| Sleep-EDF | Source + STSA | 82.99 ± 4.29 | 50.23 ± 5.92 | 49.80 ± 4.82 |
| REFED | Source | 71.48 ± 6.19 | 50.62 ± 1.26 | 48.43 ± 2.56 |
| REFED | Source + STSA | 68.58 ± 5.83 | 50.69 ± 1.68 | 49.60 ± 2.35 |

## Sleep-EDF source-only ablation

| Variant | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Source w/o SCSN | 82.70 ± 7.29 | 57.04 ± 4.96 | 54.40 ± 5.92 |
| Source w/o HSE | 83.41 ± 6.11 | 56.16 ± 3.47 | 53.58 ± 4.86 |
| Complete source model | 83.70 ± 5.74 | 56.94 ± 3.93 | 54.31 ± 5.12 |

Use `REVISED_EXPERIMENTS_GUIDE.md` for rerun, merge, validation, and routing
export commands.
