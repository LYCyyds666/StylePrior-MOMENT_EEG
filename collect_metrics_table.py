#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_metrics_table.py

Collect mean±std metrics from result .pkl files, including files such as:
  results/eegnet_APAVA_seed2025.pkl
  results/bimtta_APAVA_seed2025.pkl
  results/ss_moment_APAVA_seed2025.pkl
  results_cbramod/cbramod_APAVA_full_seed2025.pkl
  results_cbramod_parallel/REFED/cbramod_REFED_full_merged.pkl

It tries to handle different pkl structures:
  - {"fold_results": [{"test_metrics": {...}}, ...]}
  - {"fold_results": [{"tta_metrics": {...}}, ...]}
  - {"metrics_mean": {...}, "metrics_std": {...}}
  - top-level metric arrays/lists or scalar metrics

Outputs:
  - long_metrics.csv
  - paper_table_accuracy.md / paper_table_core_metrics.md / paper_table_all_metrics.md
  - paper_table_all_metrics.csv
  - metrics_summary.xlsx, if pandas + openpyxl are available
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DATASET_CANON = {
    "apava": "APAVA",
    "sleepedf": "Sleep-EDF",
    "sleep_edf": "Sleep-EDF",
    "sleep-edf": "Sleep-EDF",
    "refed": "REFED",
}

DATASET_ORDER = ["APAVA", "Sleep-EDF", "REFED"]

MODEL_DISPLAY = {
    "eegnet": "EEGNet",
    "graphsleepnet": "GraphSleepNet",
    "graph_sleepnet": "GraphSleepNet",
    "salientsleepnet": "SalientSleepNet",
    "salient_sleepnet": "SalientSleepNet",
    "mmcnn": "MMCNN",
    "bimtta": "BiM-TTA",
    "bim_tta": "BiM-TTA",
    "ss_moment": "StylePrior-MOMENT",
    "ss-moment": "StylePrior-MOMENT",
    "styleprior_moment": "StylePrior-MOMENT",
    "styleprior-moment": "StylePrior-MOMENT",
    "cbramod": "CBraMod",
    "cbramod_full": "CBraMod",
    "cbramod_head_only": "CBraMod-head",
    "moment_linear": "MOMENT-linear",
    "moment_full": "MOMENT-full",
    "moment_lora": "MOMENT-LoRA",
    "lora": "LoRA",
}

MODEL_ORDER = [
    "EEGNet",
    "GraphSleepNet",
    "SalientSleepNet",
    "MMCNN",
    "BiM-TTA",
    "CBraMod",
    "MOMENT-linear",
    "MOMENT-full",
    "MOMENT-LoRA",
    "LoRA",
    "StylePrior-MOMENT",
]

METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "F1 Score (Macro)",
    "precision_macro": "Precision (Macro)",
    "recall_macro": "Recall (Macro)",
    "roc_auc": "ROC AUC",
    "average_precision": "Avg Prec",
}

METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "roc_auc",
    "average_precision",
]

ALIASES = {
    "accuracy": [
        "accuracy", "acc", "test_acc", "tta_acc", "baseline_acc",
        "eval_accuracy", "val_accuracy"
    ],
    "balanced_accuracy": [
        "balanced_accuracy", "balanced_acc", "bacc", "balancedaccuracy",
        "test_bacc", "val_bacc", "balanced acc"
    ],
    "f1_macro": [
        "f1_macro", "macro_f1", "f1", "f1_score", "f1 score", "f1_macro_score",
        "macro f1", "f1 macro"
    ],
    "precision_macro": [
        "precision_macro", "macro_precision", "precision", "precision_score",
        "precision macro", "macro precision"
    ],
    "recall_macro": [
        "recall_macro", "macro_recall", "recall", "recall_score",
        "recall macro", "macro recall"
    ],
    "roc_auc": [
        "roc_auc", "auc", "rocauc", "roc auc"
    ],
    "average_precision": [
        "average_precision", "avg_prec", "avg precision", "ap",
        "averageprecision", "average_prec"
    ],
}


def norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


ALIAS_TO_CANON = {}
for canon, aliases in ALIASES.items():
    ALIAS_TO_CANON[norm_key(canon)] = canon
    for a in aliases:
        ALIAS_TO_CANON[norm_key(a)] = canon


def is_number(x: Any) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.isfinite(float(x))
    return False


def to_float_list(x: Any) -> List[float]:
    if is_number(x):
        return [float(x)]
    if isinstance(x, np.ndarray):
        x = x.flatten().tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if is_number(v):
                out.append(float(v))
        return out
    return []


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_dataset_and_model(path: str, obj: Any = None) -> Tuple[str, str]:
    name = Path(path).stem
    low = name.lower()

    dataset = None
    dataset_token = None
    for token, canon in DATASET_CANON.items():
        # Match both exact underscore-separated and substring, because names vary.
        if token in low.replace("-", "_"):
            dataset = canon
            dataset_token = token
            break

    if dataset is None and isinstance(obj, dict):
        for key in ["dataset", "dataset_name"]:
            if key in obj:
                val = str(obj[key]).lower().replace("-", "_")
                dataset = DATASET_CANON.get(val, str(obj[key]))
                dataset_token = val
                break

    if dataset is None:
        dataset = "UNKNOWN"

    # Model is usually before dataset token.
    model_part = low
    if dataset_token:
        idx = low.replace("-", "_").find(dataset_token)
        if idx >= 0:
            model_part = low[:idx].strip("_-")
    model_part = re.sub(r"_?seed\d+.*$", "", model_part)
    model_part = re.sub(r"_?merged.*$", "", model_part)
    model_part = re.sub(r"_?full$", "_full", model_part)
    model_part = model_part.strip("_-")

    # Special case: cbramod_APAVA_full_seed2025 => cbramod_full
    if low.startswith("cbramod") and "_head" in low:
        model_key = "cbramod_head_only"
    elif low.startswith("cbramod"):
        model_key = "cbramod"
    else:
        model_key = model_part or "UNKNOWN"

    display = MODEL_DISPLAY.get(model_key, MODEL_DISPLAY.get(model_key.replace("-", "_"), model_key))
    return dataset, display


def extract_metrics_from_dict(d: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    if not isinstance(d, dict):
        return out

    # Direct metric keys.
    for k, v in d.items():
        nk = norm_key(k)
        # Skip summary subdicts here; handled elsewhere.
        if nk in ALIAS_TO_CANON:
            canon = ALIAS_TO_CANON[nk]
            vals = to_float_list(v)
            if vals:
                out[canon].extend(vals)

    # Sometimes metrics are saved as mean/std keys.
    for k, v in d.items():
        nk = norm_key(k)
        if nk.endswith("_mean"):
            base = nk[:-5]
            canon = ALIAS_TO_CANON.get(base)
            vals = to_float_list(v)
            if canon and vals:
                out[canon].extend(vals)

    return out


def choose_fold_metric_dict(fold_result: Dict[str, Any], model: str, prefer: str) -> Optional[Dict[str, Any]]:
    if not isinstance(fold_result, dict):
        return None

    if prefer == "test":
        candidates = ["test_metrics", "metrics", "final_metrics", "best_metrics"]
    elif prefer == "tta":
        candidates = ["tta_metrics", "test_metrics", "metrics", "final_metrics", "best_metrics"]
    elif prefer == "baseline":
        candidates = ["baseline_metrics", "test_metrics", "metrics", "final_metrics"]
    else:
        # auto: if TTA exists, use it. This is normally what we want for SS-MOMENT / BiM-TTA.
        candidates = ["tta_metrics", "test_metrics", "metrics", "final_metrics", "best_metrics", "baseline_metrics"]

    for key in candidates:
        if key in fold_result and isinstance(fold_result[key], dict):
            return fold_result[key]

    # Some scripts save metric scalars directly in each fold result.
    direct = extract_metrics_from_dict(fold_result)
    if direct:
        return fold_result

    return None


def collect_metric_values(obj: Any, model: str, prefer: str = "auto") -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = defaultdict(list)

    if not isinstance(obj, dict):
        return values

    # 1) Our common format: fold_results list.
    if isinstance(obj.get("fold_results"), list):
        for fr in obj["fold_results"]:
            md = choose_fold_metric_dict(fr, model, prefer)
            if md is None:
                continue
            extracted = extract_metrics_from_dict(md)
            for k, vals in extracted.items():
                values[k].extend(vals)
        if values:
            return values

    # 2) Alternative result-list keys.
    for list_key in ["results", "all_results", "fold_metrics", "per_fold", "folds"]:
        if isinstance(obj.get(list_key), list):
            for fr in obj[list_key]:
                if isinstance(fr, dict):
                    md = choose_fold_metric_dict(fr, model, prefer) or fr
                    extracted = extract_metrics_from_dict(md)
                    for k, vals in extracted.items():
                        values[k].extend(vals)
            if values:
                return values

    # 3) Merged summary with metrics_mean and metrics_std.
    # Here we store the mean as a singleton, but keep std separately later.
    if isinstance(obj.get("metrics_mean"), dict):
        for k, v in obj["metrics_mean"].items():
            nk = norm_key(k)
            canon = ALIAS_TO_CANON.get(nk)
            if canon and is_number(v):
                values[canon].append(float(v))
        if values:
            return values

    # 4) Direct top-level metric arrays/scalars.
    extracted = extract_metrics_from_dict(obj)
    for k, vals in extracted.items():
        values[k].extend(vals)

    return values


def collect_metric_std_from_summary(obj: Any) -> Dict[str, float]:
    out = {}
    if isinstance(obj, dict) and isinstance(obj.get("metrics_std"), dict):
        for k, v in obj["metrics_std"].items():
            canon = ALIAS_TO_CANON.get(norm_key(k))
            if canon and is_number(v):
                out[canon] = float(v)
    # Also support keys like accuracy_std.
    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = norm_key(k)
            if nk.endswith("_std"):
                canon = ALIAS_TO_CANON.get(nk[:-4])
                if canon and is_number(v):
                    out[canon] = float(v)
    return out


def mean_std(vals: List[float], summary_std: Optional[float] = None) -> Tuple[Optional[float], Optional[float], int]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return None, None, 0
    mean = float(np.mean(vals))
    if len(vals) > 1:
        std = float(np.std(vals))
    else:
        std = summary_std
    return mean, std, len(vals)


def fmt_cell(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def sort_models(models: Iterable[str]) -> List[str]:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order.get(m, 999), m.lower()))


def sort_datasets(ds: Iterable[str]) -> List[str]:
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    return sorted(ds, key=lambda d: (order.get(d, 999), d.lower()))


def write_markdown_table(path: str, rows: List[Dict[str, str]], models: List[str], metrics: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        header = ["Datasets", "Metrics"] + models
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        prev_ds = None
        for row in rows:
            ds = row["dataset"]
            metric = row["metric_display"]
            ds_cell = ds if ds != prev_ds else ""
            prev_ds = ds
            vals = [row.get(m, "") for m in models]
            f.write("| " + " | ".join([ds_cell, metric] + vals) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["./results"], help="Directories to search recursively.")
    ap.add_argument("--out_dir", default="./paper_tables", help="Output directory.")
    ap.add_argument("--prefer", default="auto", choices=["auto", "test", "tta", "baseline"],
                    help="Which metric dict to use when a fold has multiple metric sources.")
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths = []
    for root in args.roots:
        pkl_paths.extend(glob.glob(os.path.join(root, "**", "*.pkl"), recursive=True))
    pkl_paths = sorted(set(pkl_paths))

    records = []
    skipped = []

    for path in pkl_paths:
        try:
            obj = load_pickle(path)
        except Exception as e:
            skipped.append((path, f"load failed: {e}"))
            continue

        dataset, model = infer_dataset_and_model(path, obj)
        vals = collect_metric_values(obj, model, prefer=args.prefer)
        std_summary = collect_metric_std_from_summary(obj)

        if not vals:
            skipped.append((path, "no recognized metrics"))
            continue

        if args.debug:
            print(f"[OK] {path} -> dataset={dataset}, model={model}, metrics={list(vals)}")

        for metric in METRIC_ORDER:
            if metric not in vals:
                continue
            mean, std, n = mean_std(vals[metric], std_summary.get(metric))
            if mean is None:
                continue
            records.append({
                "source_file": path,
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "metric_display": METRIC_DISPLAY.get(metric, metric),
                "mean": mean,
                "std": std,
                "n": n,
                "cell": fmt_cell(mean, std, args.digits),
            })

    if not records:
        raise RuntimeError("No metrics found. Try --debug and inspect pkl structure.")

    # If multiple files map to same dataset/model/metric, keep the latest by mtime.
    dedup = {}
    for r in records:
        key = (r["dataset"], r["model"], r["metric"])
        cur = dedup.get(key)
        if cur is None or os.path.getmtime(r["source_file"]) >= os.path.getmtime(cur["source_file"]):
            dedup[key] = r
    records = list(dedup.values())

    datasets = sort_datasets({r["dataset"] for r in records})
    models = sort_models({r["model"] for r in records})

    # Long csv.
    long_csv = out_dir / "long_metrics.csv"
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "metric", "metric_display", "mean", "std", "n", "cell", "source_file"])
        w.writeheader()
        for r in sorted(records, key=lambda x: (DATASET_ORDER.index(x["dataset"]) if x["dataset"] in DATASET_ORDER else 999, x["metric"], MODEL_ORDER.index(x["model"]) if x["model"] in MODEL_ORDER else 999)):
            w.writerow(r)

    def make_rows(metric_subset: List[str]) -> List[Dict[str, str]]:
        rows = []
        rec_map = {(r["dataset"], r["metric"], r["model"]): r for r in records}
        for ds in datasets:
            present_metrics = [m for m in metric_subset if any((ds, m, model) in rec_map for model in models)]
            for metric in present_metrics:
                row = {"dataset": ds, "metric": metric, "metric_display": METRIC_DISPLAY.get(metric, metric)}
                for model in models:
                    rr = rec_map.get((ds, metric, model))
                    row[model] = rr["cell"] if rr else ""
                rows.append(row)
        return rows

    rows_accuracy = make_rows(["accuracy"])
    rows_core = make_rows(["accuracy", "balanced_accuracy", "f1_macro"])
    rows_all = make_rows(METRIC_ORDER)

    write_markdown_table(str(out_dir / "paper_table_accuracy.md"), rows_accuracy, models, ["accuracy"])
    write_markdown_table(str(out_dir / "paper_table_core_metrics.md"), rows_core, models, ["accuracy", "balanced_accuracy", "f1_macro"])
    write_markdown_table(str(out_dir / "paper_table_all_metrics.md"), rows_all, models, METRIC_ORDER)

    # Wide CSV for all metrics.
    wide_csv = out_dir / "paper_table_all_metrics.csv"
    with open(wide_csv, "w", newline="", encoding="utf-8") as f:
        header = ["Datasets", "Metrics"] + models
        w = csv.writer(f)
        w.writerow(header)
        prev = None
        for row in rows_all:
            ds = row["dataset"]
            ds_cell = ds if ds != prev else ""
            prev = ds
            w.writerow([ds_cell, row["metric_display"]] + [row.get(m, "") for m in models])

    # Optional Excel.
    try:
        import pandas as pd
        xlsx = out_dir / "metrics_summary.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            pd.DataFrame(records).to_excel(writer, index=False, sheet_name="long_metrics")
            pd.DataFrame(rows_all).to_excel(writer, index=False, sheet_name="all_metrics")
            pd.DataFrame(rows_core).to_excel(writer, index=False, sheet_name="core_metrics")
            pd.DataFrame(rows_accuracy).to_excel(writer, index=False, sheet_name="accuracy")
        print(f"Saved Excel: {xlsx}")
    except Exception as e:
        print(f"Excel not written: {e}")

    # Skipped report.
    if skipped:
        with open(out_dir / "skipped_files.txt", "w", encoding="utf-8") as f:
            for path, reason in skipped:
                f.write(f"{path}\t{reason}\n")

    print("=" * 80)
    print(f"Found pkl files: {len(pkl_paths)}")
    print(f"Used metric records: {len(records)}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Saved: {long_csv}")
    print(f"Saved: {out_dir / 'paper_table_all_metrics.md'}")
    print(f"Saved: {wide_csv}")
    if skipped:
        print(f"Skipped files: {len(skipped)} -> {out_dir / 'skipped_files.txt'}")


if __name__ == "__main__":
    main()
