"""Merge corrected StylePrior-MOMENT fold shards."""

import argparse
import os
import pickle
import re

import numpy as np

from routing_analysis import merge_routing_analyses


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    choices=["APAVA", "SleepEDF", "REFED"],
)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--input_dir", default="./results_corrected")
parser.add_argument("--output_dir", default="./results_corrected")
args = parser.parse_args()

expected_folds = {
    "APAVA": 5,
    "SleepEDF": 20,
    "REFED": 32,
}[args.dataset]

pattern = re.compile(
    rf"^ss_moment_{re.escape(args.dataset)}_"
    rf"fold(\d+)-(\d+)_seed{args.seed}\.pkl$"
)

shard_paths = []
for filename in sorted(os.listdir(args.input_dir)):
    match = pattern.match(filename)
    if match:
        shard_paths.append(
            (
                int(match.group(1)),
                int(match.group(2)),
                os.path.join(args.input_dir, filename),
            )
        )

if not shard_paths:
    raise FileNotFoundError(
        f"No StylePrior shards found in {args.input_dir} for {args.dataset}"
    )

fold_results_by_id = {}
for declared_start, declared_end, path in shard_paths:
    with open(path, "rb") as handle:
        shard = pickle.load(handle)

    if shard.get("dataset") != args.dataset:
        raise ValueError(f"Dataset mismatch in {path}")
    if int(shard.get("seed", -1)) != args.seed:
        raise ValueError(f"Seed mismatch in {path}")
    if int(shard.get("start_fold", declared_start)) != declared_start:
        raise ValueError(f"Start-fold metadata mismatch in {path}")
    if int(shard.get("end_fold", declared_end)) != declared_end:
        raise ValueError(f"End-fold metadata mismatch in {path}")

    for result in shard.get("fold_results", []):
        fold_id = int(result["fold"])
        if fold_id in fold_results_by_id:
            raise ValueError(f"Duplicate fold {fold_id} found in {path}")
        fold_results_by_id[fold_id] = result

expected_fold_ids = list(range(1, expected_folds + 1))
present_fold_ids = sorted(fold_results_by_id)
missing_fold_ids = sorted(set(expected_fold_ids) - set(present_fold_ids))
unexpected_fold_ids = sorted(set(present_fold_ids) - set(expected_fold_ids))
if missing_fold_ids or unexpected_fold_ids:
    raise RuntimeError(
        "Fold coverage is incomplete or invalid. "
        f"Missing={missing_fold_ids}, unexpected={unexpected_fold_ids}"
    )

fold_results = [fold_results_by_id[fold_id] for fold_id in expected_fold_ids]
failed = [
    result["fold"]
    for result in fold_results
    if result.get("status") == "failed"
    or result.get("test_metrics") is None
]
if failed:
    raise RuntimeError(f"Failed folds cannot be merged: {failed}")


def aggregate(metric_key):
    metric_values = {}
    for result in fold_results:
        metrics = result.get(metric_key)
        if not metrics:
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))

    means = {
        key: float(np.mean(values))
        for key, values in metric_values.items()
    }
    standard_deviations = {
        key: float(np.std(values))
        for key, values in metric_values.items()
    }
    return means, standard_deviations


baseline_mean, baseline_std = aggregate("test_metrics")
tta_mean, tta_std = aggregate("tta_metrics")
routing_analysis = merge_routing_analyses(
    [
        result.get("routing_analysis")
        for result in fold_results
        if result.get("routing_analysis")
    ]
)

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(
    args.output_dir,
    f"ss_moment_{args.dataset}_seed{args.seed}.pkl",
)
with open(output_path, "wb") as handle:
    pickle.dump(
        {
            "dataset": args.dataset,
            "seed": args.seed,
            "k_folds": expected_folds,
            "completed_folds": expected_folds,
            "start_fold": 0,
            "end_fold": expected_folds,
            "baseline_metrics": baseline_mean,
            "baseline_metrics_std": baseline_std,
            "tta_metrics": tta_mean or None,
            "tta_metrics_std": tta_std or None,
            "routing_analysis": routing_analysis,
            "fold_results": fold_results,
            "source_files": [path for _, _, path in shard_paths],
        },
        handle,
    )

print(f"Merged {expected_folds} folds into {output_path}")
for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
    if metric in baseline_mean:
        print(
            f"Source {metric}: "
            f"{baseline_mean[metric]:.4f} ± {baseline_std[metric]:.4f}"
        )
    if tta_mean and metric in tta_mean:
        print(
            f"STSA   {metric}: "
            f"{tta_mean[metric]:.4f} ± {tta_std[metric]:.4f}"
        )
print(
    "Routing analysis: "
    + ("available" if routing_analysis else "not collected")
)
