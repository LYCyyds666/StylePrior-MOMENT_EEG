#!/usr/bin/env python3
"""
merge_cbramod_parallel.py

Merge CBraMod chunk results produced by multiple GPUs.

Example:
  python merge_cbramod_parallel.py \
    --dataset REFED \
    --mode full \
    --root ./results_cbramod_parallel/REFED \
    --expected_folds 32 \
    --out_prefix ./results_cbramod_parallel/REFED/cbramod_REFED_full_merged
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "roc_auc",
    "average_precision",
]


def load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_result_files(root: str, dataset: str, mode: str, seed: int) -> List[str]:
    final_pat = os.path.join(root, "**", f"cbramod_{dataset}_{mode}_seed{seed}.pkl")
    partial_pat = os.path.join(root, "**", f"cbramod_{dataset}_{mode}_seed{seed}_partial.pkl")
    finals = sorted(glob.glob(final_pat, recursive=True))
    if finals:
        return finals
    return sorted(glob.glob(partial_pat, recursive=True))


def collect_folds(paths: List[str]) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    fold_map: Dict[int, Dict[str, Any]] = {}
    failed: List[Dict[str, Any]] = []

    # Sort by mtime, so if duplicate fold exists, latest successful result wins.
    paths = sorted(paths, key=lambda p: os.path.getmtime(p))

    for path in paths:
        obj = load_pkl(path)
        for r in obj.get("fold_results", []):
            fold = int(r.get("fold", -1))
            if r.get("status") == "ok" and r.get("test_metrics"):
                rr = dict(r)
                rr["_source_file"] = path
                fold_map[fold] = rr
            else:
                failed.append({"fold": fold, "source_file": path, "error": r.get("error", "")})
    return fold_map, failed


def aggregate(fold_map: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    vals: Dict[str, List[float]] = {}
    for fold, r in sorted(fold_map.items()):
        metrics = r.get("test_metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                vals.setdefault(k, []).append(float(v))
    means = {k: float(np.mean(v)) for k, v in vals.items()}
    stds = {k: float(np.std(v)) for k, v in vals.items()}
    return means, stds


def fmt(means: Dict[str, float], stds: Dict[str, float], metric: str) -> str:
    if metric not in means:
        return ""
    return f"{100 * means[metric]:.2f} ± {100 * stds.get(metric, 0.0):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["REFED", "SleepEDF", "APAVA"])
    ap.add_argument("--mode", default="full")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--root", required=True)
    ap.add_argument("--expected_folds", type=int, required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    paths = find_result_files(args.root, args.dataset, args.mode, args.seed)
    if not paths:
        raise FileNotFoundError(f"No pkl files found under {args.root}")

    fold_map, failed = collect_folds(paths)
    means, stds = aggregate(fold_map)

    present = sorted(fold_map)
    expected = list(range(1, args.expected_folds + 1))
    missing = [f for f in expected if f not in fold_map]

    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "expected_folds": args.expected_folds,
        "completed_folds": len(fold_map),
        "present_folds": present,
        "missing_folds": missing,
        "metrics_mean": means,
        "metrics_std": stds,
        "source_files": paths,
        "failed_records": failed,
        "fold_results": [fold_map[f] for f in present],
    }

    out_pkl = args.out_prefix + ".pkl"
    out_json = args.out_prefix + ".json"
    out_csv = args.out_prefix + ".csv"
    out_md = args.out_prefix + ".md"

    Path(os.path.dirname(out_pkl)).mkdir(parents=True, exist_ok=True)

    with open(out_pkl, "wb") as f:
        pickle.dump(summary, f)

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "mode", "completed_folds", "expected_folds"] + [m + "_mean" for m in METRICS] + [m + "_std" for m in METRICS])
        w.writerow([args.dataset, args.mode, len(fold_map), args.expected_folds] + [means.get(m, "") for m in METRICS] + [stds.get(m, "") for m in METRICS])

    with open(out_md, "w") as f:
        f.write(f"# CBraMod merged result: {args.dataset} ({args.mode})\n\n")
        f.write(f"Completed folds: **{len(fold_map)}/{args.expected_folds}**\n\n")
        if missing:
            f.write(f"Missing folds: `{missing}`\n\n")
        f.write("| Metric | Mean ± Std (%) |\n")
        f.write("|---|---:|\n")
        for m in METRICS:
            if m in means:
                f.write(f"| {m} | {fmt(means, stds, m)} |\n")
        f.write("\n## Fold-level results\n\n")
        f.write("| Fold | Acc | BAcc | Macro-F1 | Source file |\n")
        f.write("|---:|---:|---:|---:|---|\n")
        for fold in present:
            r = fold_map[fold]
            tm = r.get("test_metrics", {})
            f.write(
                f"| {fold} | "
                f"{100*tm.get('accuracy', float('nan')):.2f} | "
                f"{100*tm.get('balanced_accuracy', float('nan')):.2f} | "
                f"{100*tm.get('f1_macro', float('nan')):.2f} | "
                f"`{r.get('_source_file','')}` |\n"
            )

    print("=" * 70)
    print(f"CBraMod merged summary | dataset={args.dataset} | mode={args.mode}")
    print(f"Completed folds: {len(fold_map)}/{args.expected_folds}")
    if missing:
        print(f"Missing folds: {missing}")
    for m in METRICS:
        if m in means:
            print(f"  {m:20s}: {means[m]:.4f} ± {stds[m]:.4f}")
    print(f"Saved: {out_pkl}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
