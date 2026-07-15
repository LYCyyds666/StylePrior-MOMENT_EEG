#!/usr/bin/env python3
"""Insert canonical source and STSA metrics into generated paper tables."""
from pathlib import Path
import csv
import math
import pickle
import re


DATASETS = {
    "APAVA": Path("results/ss_moment_APAVA_seed2025.pkl"),
    "Sleep-EDF": Path("results/ss_moment_SleepEDF_seed2025.pkl"),
    "REFED": Path("results/ss_moment_REFED_seed2025.pkl"),
}

METRICS = {
    "Accuracy": ("accuracy",),
    "Balanced Accuracy": ("balanced_accuracy",),
    "F1 Score (Macro)": ("f1_macro",),
    "Precision (Macro)": ("precision_macro",),
    "Recall (Macro)": ("recall_macro",),
    "ROC AUC": ("roc_auc", "roc_auc_ovr"),
    "Avg Prec": ("average_precision", "average_precision_macro"),
}


def norm(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def summarize(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std, len(values)


def get_metric(metrics, candidates):
    for key in candidates:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value), key
    raise KeyError(f"none of {candidates} found in {sorted(metrics)}")


def load_summaries():
    summaries = {}
    for dataset, path in DATASETS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        folds = payload.get("fold_results")
        if not isinstance(folds, list) or not folds:
            raise ValueError(f"invalid fold_results in {path}")
        for variant, field in (("source", "test_metrics"), ("stsa", "tta_metrics")):
            for display, candidates in METRICS.items():
                values = []
                selected_key = None
                for fold in folds:
                    metrics = fold.get(field, {})
                    try:
                        value, key = get_metric(metrics, candidates)
                    except KeyError:
                        values = []
                        break
                    values.append(value)
                    selected_key = key
                if values:
                    summaries[(dataset, variant, display)] = (
                        *summarize(values),
                        selected_key,
                        str(path),
                    )
    return summaries


def write_markdown(path, rows):
    escaped = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    header = escaped[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|---|---|" + "|".join("---:" for _ in header[2:]) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in escaped[1:])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    summaries = load_summaries()
    table_path = Path("paper_tables/paper_table_all_metrics.csv")
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"empty table: {table_path}")

    old_header = rows[0]
    keep_indices = [
        index for index, name in enumerate(old_header)
        if "styleprior" not in norm(name)
    ]
    new_header = [old_header[index] for index in keep_indices]
    new_header += ["StylePrior-MOMENT + STSA", "StylePrior-MOMENT (source)"]
    new_rows = [new_header]
    current_dataset = ""
    for raw in rows[1:]:
        row = list(raw) + [""] * (len(old_header) - len(raw))
        row = row[:len(old_header)]
        if row[0].strip():
            current_dataset = row[0].strip()
        display = row[1].strip() if len(row) > 1 else ""
        kept = [row[index] for index in keep_indices]
        for variant in ("stsa", "source"):
            item = summaries.get((current_dataset, variant, display))
            kept.append(f"{item[0]:.4f} ± {item[1]:.4f}" if item else "")
        new_rows.append(kept)

    with table_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(new_rows)
    write_markdown(Path("paper_tables/paper_table_all_metrics.md"), new_rows)

    long_path = Path("paper_tables/long_metrics.csv")
    with long_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        long_header = reader.fieldnames
        long_rows = list(reader)
    if not long_header:
        raise ValueError(f"missing header: {long_path}")
    long_rows = [
        row for row in long_rows if "styleprior" not in norm(row.get("model", ""))
    ]
    for dataset in DATASETS:
        for variant, model in (
            ("stsa", "StylePrior-MOMENT + STSA"),
            ("source", "StylePrior-MOMENT (source)"),
        ):
            for display, candidates in METRICS.items():
                item = summaries.get((dataset, variant, display))
                if not item:
                    continue
                mean, std, n, selected_key, source_file = item
                row = {name: "" for name in long_header}
                row.update({
                    "dataset": dataset,
                    "model": model,
                    "metric": selected_key,
                    "metric_display": display,
                    "mean": repr(mean),
                    "std": repr(std),
                    "n": str(n),
                    "cell": f"{mean:.4f} ± {std:.4f}",
                    "source_file": "./" + source_file.lstrip("./"),
                })
                long_rows.append(row)
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(long_rows)


if __name__ == "__main__":
    main()
