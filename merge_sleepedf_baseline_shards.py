#!/usr/bin/env python3
"""Merge five completed Sleep-EDF shards into the canonical result files."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./results_parallel_seq256")
    parser.add_argument("--result_dir", default="./results")
    return parser.parse_args()


def merge_model(model_name, input_dir, result_dir):
    files = sorted((input_dir / model_name).glob(
        f"{model_name}_SleepEDF_fold*-*_seed2025.pkl"
    ))
    if len(files) != 5:
        raise RuntimeError(f"{model_name}: expected 5 shard files, found {len(files)}")

    by_fold = {}
    for path in files:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        for row in payload.get("fold_results", []):
            fold = int(row["fold"])
            if fold in by_fold:
                raise RuntimeError(f"{model_name}: duplicate fold {fold}")
            by_fold[fold] = row

    expected = set(range(1, 21))
    present = set(by_fold)
    if present != expected:
        raise RuntimeError(
            f"{model_name}: missing={sorted(expected - present)}, "
            f"unexpected={sorted(present - expected)}"
        )

    rows = [by_fold[fold] for fold in sorted(by_fold)]
    successful = [
        row for row in rows
        if row.get("status") != "failed" and row.get("test_metrics")
    ]
    if len(successful) != 20:
        raise RuntimeError(f"{model_name}: only {len(successful)}/20 successful folds")

    numeric = {}
    for row in successful:
        for key, value in row["test_metrics"].items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric.setdefault(key, []).append(float(value))
    means = {key: float(np.mean(values)) for key, values in numeric.items()}
    stds = {key: float(np.std(values)) for key, values in numeric.items()}

    merged = {
        "dataset": "SleepEDF",
        "model": model_name,
        "seed": 2025,
        "k_folds": 20,
        "completed_folds": 20,
        "baseline_metrics": means,
        "metrics_std": stds,
        "fold_results": rows,
        "source_files": [str(path) for path in files],
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / f"{model_name}_SleepEDF_seed2025.pkl"
    with output.open("wb") as handle:
        pickle.dump(merged, handle)

    print(f"\n{model_name}: {output}")
    for key in ["accuracy", "balanced_accuracy", "f1_macro"]:
        print(f"  {key}: {means[key]:.4f} ± {stds[key]:.4f}")
    return output


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    result_dir = Path(args.result_dir)
    merge_model("mmcnn", input_dir, result_dir)
    merge_model("graphsleepnet", input_dir, result_dir)


if __name__ == "__main__":
    main()
