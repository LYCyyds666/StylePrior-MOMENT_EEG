"""Export merged routing summaries and heatmap-ready CSV files."""

import argparse
import csv
import os
import pickle

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    choices=["APAVA", "SleepEDF", "REFED"],
)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--results_dir", default="./results_corrected")
parser.add_argument("--output_dir", default="./routing_analysis")
parser.add_argument(
    "--plot",
    action="store_true",
    help="Also create PNG heatmaps; requires matplotlib",
)
args = parser.parse_args()

result_path = os.path.join(
    args.results_dir,
    f"ss_moment_{args.dataset}_seed{args.seed}.pkl",
)
with open(result_path, "rb") as handle:
    result = pickle.load(handle)

analysis = result.get("routing_analysis")
if not analysis or not analysis.get("layers"):
    raise RuntimeError(
        f"No routing analysis found in {result_path}. "
        "Rerun without --no_routing_analysis."
    )

layers = analysis["layers"]
first_layer = next(iter(layers.values()))
num_experts = int(first_layer["num_experts"])
expert_columns = [f"expert_{index}" for index in range(num_experts)]

os.makedirs(args.output_dir, exist_ok=True)

layer_path = os.path.join(
    args.output_dir,
    f"{args.dataset}_layer_routing.csv",
)
with open(layer_path, "w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "dataset",
        "layer",
        "token_count",
        "router_entropy",
        "normalized_router_entropy",
        "measure",
        *expert_columns,
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for layer_id, layer in sorted(layers.items(), key=lambda item: int(item[0])):
        for measure in (
            "mean_router_probability",
            "top1_fraction",
            "topk_assignment_fraction",
        ):
            row = {
                "dataset": args.dataset,
                "layer": int(layer_id),
                "token_count": layer["token_count"],
                "router_entropy": layer["router_entropy"],
                "normalized_router_entropy": layer[
                    "normalized_router_entropy"
                ],
                "measure": measure,
            }
            row.update(
                {
                    column: value
                    for column, value in zip(
                        expert_columns,
                        layer[measure],
                    )
                }
            )
            writer.writerow(row)


def export_group_heatmap(group_key, group_name):
    output_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{group_name}_expert_heatmap.csv",
    )
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "layer",
            group_name,
            *expert_columns,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_id, layer in sorted(
            layers.items(),
            key=lambda item: int(item[0]),
        ):
            for group_value, fractions in sorted(
                layer[group_key].items(),
                key=lambda item: int(item[0]),
            ):
                row = {
                    "dataset": args.dataset,
                    "layer": int(layer_id),
                    group_name: int(group_value),
                }
                row.update(
                    {
                        column: value
                        for column, value in zip(expert_columns, fractions)
                    }
                )
                writer.writerow(row)
    return output_path


subject_path = export_group_heatmap(
    "subject_topk_fraction",
    "subject",
)
class_path = export_group_heatmap(
    "class_topk_fraction",
    "class",
)


def aggregate_heatmap(group_key):
    counts = {}
    for layer_id, raw_layer in analysis["_counts"].items():
        del layer_id
        raw_groups = raw_layer[group_key]
        for group_value, values in raw_groups.items():
            values_array = np.asarray(values, dtype=np.float64)
            if group_value not in counts:
                counts[group_value] = np.zeros_like(values_array)
            counts[group_value] += values_array
    group_ids = sorted(counts, key=int)
    matrix = np.vstack([counts[group_id] for group_id in group_ids])
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    return group_ids, matrix


if args.plot:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Install matplotlib before using --plot: pip install matplotlib"
        ) from exc

    plot_specs = [
        ("subject_topk_count", "Subject", "subject"),
        ("class_topk_count", "Class", "class"),
    ]
    for group_key, axis_label, filename_label in plot_specs:
        group_ids, matrix = aggregate_heatmap(group_key)
        figure_width = max(6.0, 0.7 * num_experts + 2.5)
        figure_height = max(3.5, 0.35 * len(group_ids) + 2.0)
        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            constrained_layout=True,
        )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(num_experts), expert_columns)
        axis.set_yticks(range(len(group_ids)), group_ids)
        axis.set_xlabel("Expert")
        axis.set_ylabel(axis_label)
        axis.set_title(
            f"{args.dataset}: {axis_label.lower()}-expert routing"
        )
        figure.colorbar(image, ax=axis, label="Top-k assignment fraction")
        figure.savefig(
            os.path.join(
                args.output_dir,
                f"{args.dataset}_{filename_label}_expert_heatmap.png",
            ),
            dpi=300,
        )
        plt.close(figure)

print(f"Saved: {layer_path}")
print(f"Saved: {subject_path}")
print(f"Saved: {class_path}")
if args.plot:
    print(f"Saved PNG heatmaps under: {args.output_dir}")
