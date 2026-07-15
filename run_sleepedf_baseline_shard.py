#!/usr/bin/env python3
"""Run a contiguous Sleep-EDF fold shard for one baseline on one visible GPU."""

import argparse
import gc
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mmcnn", "graphsleepnet"], required=True)
    parser.add_argument("--start_fold", type=int, required=True, help="Zero-based, inclusive")
    parser.add_argument("--end_fold", type=int, required=True, help="Zero-based, exclusive")
    parser.add_argument("--data", default="./datasets/sleepedf/sleepedf_all.pkl")
    parser.add_argument("--output_dir", default="./results_parallel_seq256")
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def load_module(model_name):
    module_name = {
        "mmcnn": "mmcnn_baseline",
        "graphsleepnet": "graphsleepnet_baseline",
    }[model_name]
    original_argv = sys.argv[:]
    try:
        sys.argv = [f"{module_name}.py", "--dataset", "SleepEDF"]
        module = importlib.import_module(module_name)
    finally:
        sys.argv = original_argv
    return module


def save_shard(path, model_name, start_fold, end_fold, fold_results):
    successful = [
        row for row in fold_results
        if row.get("status") != "failed" and row.get("test_metrics")
    ]
    numeric = {}
    for row in successful:
        for key, value in row["test_metrics"].items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric.setdefault(key, []).append(float(value))
    means = {key: float(np.mean(values)) for key, values in numeric.items()}
    stds = {key: float(np.std(values)) for key, values in numeric.items()}
    payload = {
        "dataset": "SleepEDF",
        "model": model_name,
        "seed": 2025,
        "k_folds": 20,
        "start_fold": start_fold,
        "end_fold": end_fold,
        "completed_folds": len(successful),
        "baseline_metrics": means,
        "metrics_std": stds,
        "fold_results": fold_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if not (0 <= args.start_fold < args.end_fold <= 20):
        raise ValueError("Require 0 <= start_fold < end_fold <= 20")

    baseline = load_module(args.model)
    if baseline.seq_len != 256:
        raise RuntimeError(
            f"{args.model} SleepEDF seq_len is {baseline.seq_len}; patch it to 256 first"
        )

    from preprocessing_sleepedf import SleepEDFDataset, _samples

    with open(args.data, "rb") as handle:
        all_data = pickle.load(handle)
    subject_ids = sorted(all_data)
    if len(subject_ids) != 20:
        raise RuntimeError(f"Expected 20 subjects, found {len(subject_ids)}")

    first_shape = tuple(all_data[subject_ids[0]]["X"].shape)
    if first_shape[-2:] != (16, 256):
        raise RuntimeError(f"Expected cached samples (16, 256), found {first_shape}")

    output_path = (
        Path(args.output_dir)
        / args.model
        / f"{args.model}_SleepEDF_fold{args.start_fold}-{args.end_fold}_seed2025.pkl"
    )
    fold_results = []

    print(f"Model: {args.model}")
    print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"PyTorch device: {baseline.device}")
    print(f"Folds: [{args.start_fold}, {args.end_fold})")
    print(f"Output: {output_path}")

    for fold_index in range(args.start_fold, args.end_fold):
        test_subject = subject_ids[fold_index]
        train_subjects = [sid for sid in subject_ids if sid != test_subject]
        print("\n" + "=" * 72)
        print(f"Fold {fold_index + 1}/20; held-out subject {test_subject}")
        print("=" * 72)

        train_dataset = SleepEDFDataset(_samples(all_data, train_subjects))
        test_dataset = SleepEDFDataset(_samples(all_data, [test_subject]))
        train_subset, validation_subset = baseline.split_by_subject(train_dataset)

        train_loader = DataLoader(
            train_subset,
            batch_size=baseline.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        validation_loader = DataLoader(
            validation_subset,
            batch_size=baseline.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=baseline.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        baseline.set_all_seeds(baseline.seed)
        best_state = baseline.train_one_fold(train_loader, validation_loader)
        if best_state is None:
            fold_results.append(
                {"fold": fold_index + 1, "status": "failed", "test_metrics": None}
            )
        else:
            model = baseline.build_model()
            model.load_state_dict(best_state["state_dict"])
            test_metrics = baseline.evaluate(model, test_loader)
            baseline.print_validation_results(
                test_metrics, fold_index + 1, f"Fold {fold_index + 1} Baseline: "
            )
            fold_results.append(
                {
                    "fold": fold_index + 1,
                    "status": "completed",
                    "test_metrics": test_metrics,
                    "tta_metrics": test_metrics,
                    "train_metrics": best_state.get("val_metrics", {}),
                    "seed": baseline.seed,
                }
            )
            del model

        save_shard(
            output_path,
            args.model,
            args.start_fold,
            args.end_fold,
            fold_results,
        )
        print(f"Checkpointed shard: {output_path}")

        del train_loader, validation_loader, test_loader
        del train_subset, validation_subset, train_dataset, test_dataset
        if best_state is not None:
            del best_state
        baseline.clear_gpu_memory()
        gc.collect()

    print(f"\nCompleted shard: {output_path}")


if __name__ == "__main__":
    main()
