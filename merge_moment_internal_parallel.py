#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

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


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_files(root: str, dataset: str, mode: str, seed: int):
    pat1 = os.path.join(root, "**", f"moment_{mode}_{dataset}_seed{seed}.pkl")
    pat2 = os.path.join(root, "**", f"moment_{mode}_{dataset}_seed{seed}_partial.pkl")
    files = sorted(glob.glob(pat1, recursive=True))
    if files:
        return files
    return sorted(glob.glob(pat2, recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", required=True, choices=["linear", "full", "lora"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--root", required=True)
    ap.add_argument("--expected_folds", type=int, required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    files = find_files(args.root, args.dataset, args.mode, args.seed)
    if not files:
        raise FileNotFoundError(f"No pkl files found under {args.root}")

    fold_map: Dict[int, Dict[str, Any]] = {}
    failed = []

    # Latest file wins on duplicate fold.
    for fp in sorted(files, key=lambda p: os.path.getmtime(p)):
        obj = load_pkl(fp)
        for r in obj.get("fold_results", []):
            fold = int(r.get("fold", -1))
            if r.get("status") == "ok" and r.get("test_metrics"):
                rr = dict(r)
                rr["_source_file"] = fp
                fold_map[fold] = rr
            else:
                failed.append({"fold": fold, "source_file": fp, "error": r.get("error", "")})

    vals = {}
    for fold, r in sorted(fold_map.items()):
        for k, v in r.get("test_metrics", {}).items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                vals.setdefault(k, []).append(float(v))

    means = {k: float(np.mean(v)) for k, v in vals.items()}
    stds = {k: float(np.std(v)) for k, v in vals.items()}

    present = sorted(fold_map)
    missing = [f for f in range(1, args.expected_folds + 1) if f not in fold_map]

    summary = {
        "dataset": args.dataset,
        "model": f"MOMENT-{args.mode}",
        "mode": args.mode,
        "seed": args.seed,
        "expected_folds": args.expected_folds,
        "completed_folds": len(fold_map),
        "present_folds": present,
        "missing_folds": missing,
        "metrics_mean": means,
        "metrics_std": stds,
        "fold_results": [fold_map[f] for f in present],
        "failed_records": failed,
        "source_files": files,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(str(out_prefix) + ".pkl", "wb") as f:
        pickle.dump(summary, f)

    with open(str(out_prefix) + ".json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(str(out_prefix) + ".md", "w") as f:
        f.write(f"# MOMENT-{args.mode} merged result: {args.dataset}\n\n")
        f.write(f"Completed folds: **{len(fold_map)}/{args.expected_folds}**\n\n")
        if missing:
            f.write(f"Missing folds: `{missing}`\n\n")
        f.write("| Metric | Mean ± Std (%) |\n")
        f.write("|---|---:|\n")
        for m in METRICS:
            if m in means:
                f.write(f"| {m} | {100*means[m]:.2f} ± {100*stds[m]:.2f} |\n")
        f.write("\n## Fold-level results\n\n")
        f.write("| Fold | Acc | BAcc | Macro-F1 | Source |\n")
        f.write("|---:|---:|---:|---:|---|\n")
        for fold in present:
            r = fold_map[fold]
            tm = r["test_metrics"]
            f.write(
                f"| {fold} | {100*tm.get('accuracy', float('nan')):.2f} | "
                f"{100*tm.get('balanced_accuracy', float('nan')):.2f} | "
                f"{100*tm.get('f1_macro', float('nan')):.2f} | `{r.get('_source_file','')}` |\n"
            )

    print("=" * 70)
    print(f"MOMENT-{args.mode} merged summary | dataset={args.dataset}")
    print(f"Completed folds: {len(fold_map)}/{args.expected_folds}")
    if missing:
        print(f"Missing folds: {missing}")
    for m in METRICS:
        if m in means:
            print(f"  {m:20s}: {means[m]:.4f} ± {stds[m]:.4f}")
    print(f"Saved: {out_prefix}.pkl")
    print(f"Saved: {out_prefix}.md")


if __name__ == "__main__":
    main()
