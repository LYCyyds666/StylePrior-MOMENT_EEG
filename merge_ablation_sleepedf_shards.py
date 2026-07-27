"""Merge corrected 20-fold Sleep-EDF source-only ablation shards."""

import argparse
import os
import pickle
import re

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    required=True,
    choices=["full", "wo_scsn", "wo_hse"],
)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--input_dir", default="./results_corrected")
parser.add_argument("--output_dir", default="./results_corrected")
args = parser.parse_args()

pattern = re.compile(
    rf"^ablation_sleepedf_{re.escape(args.variant)}_source_only_"
    rf"fold(\d+)-(\d+)_seed{args.seed}\.pkl$"
)

paths = []
for filename in sorted(os.listdir(args.input_dir)):
    match = pattern.match(filename)
    if match:
        paths.append(os.path.join(args.input_dir, filename))

if not paths:
    raise FileNotFoundError(
        f"No source-only {args.variant} shards found in {args.input_dir}"
    )

fold_results_by_id = {}
for path in paths:
    with open(path, "rb") as handle:
        shard = pickle.load(handle)
    if shard.get("variant") != args.variant:
        raise ValueError(f"Variant mismatch in {path}")
    if not shard.get("source_only"):
        raise ValueError(f"Non-source-only shard found: {path}")
    for result in shard.get("fold_results", []):
        fold_id = int(result["fold"])
        if fold_id in fold_results_by_id:
            raise ValueError(f"Duplicate fold {fold_id} in {path}")
        fold_results_by_id[fold_id] = result

expected_fold_ids = list(range(1, 21))
missing = sorted(set(expected_fold_ids) - set(fold_results_by_id))
unexpected = sorted(set(fold_results_by_id) - set(expected_fold_ids))
if missing or unexpected:
    raise RuntimeError(
        f"Invalid fold coverage. Missing={missing}, unexpected={unexpected}"
    )

fold_results = [fold_results_by_id[fold_id] for fold_id in expected_fold_ids]
failed = [
    result["fold"]
    for result in fold_results
    if result.get("status") == "failed"
    or result.get("baseline_metrics") is None
]
if failed:
    raise RuntimeError(f"Failed folds cannot be merged: {failed}")

metric_values = {}
for result in fold_results:
    for key, value in result["baseline_metrics"].items():
        if isinstance(value, (int, float)):
            metric_values.setdefault(key, []).append(float(value))

metric_mean = {
    key: float(np.mean(values))
    for key, values in metric_values.items()
}
metric_std = {
    key: float(np.std(values))
    for key, values in metric_values.items()
}

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(
    args.output_dir,
    f"ablation_sleepedf_{args.variant}_source_only_seed{args.seed}.pkl",
)
with open(output_path, "wb") as handle:
    pickle.dump(
        {
            "variant": args.variant,
            "dataset": "SleepEDF",
            "seed": args.seed,
            "k_folds": 20,
            "completed_folds": 20,
            "source_only": True,
            "metrics_mean": metric_mean,
            "metrics_std": metric_std,
            "fold_results": fold_results,
            "source_files": paths,
        },
        handle,
    )

print(f"Merged 20 folds into {output_path}")
for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
    print(
        f"{metric}: {metric_mean[metric]:.4f} ± {metric_std[metric]:.4f}"
    )
