#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_VERSION = "2026-07-15-v3"


EXPECTED_BASELINES = {
    ("graphsleepnet", "accuracy"): (0.7226864307658495, 0.0827464283309643),
    ("graphsleepnet", "balanced_accuracy"): (
        0.5596965892731893,
        0.05399145430568894,
    ),
    ("graphsleepnet", "f1_macro"): (
        0.4792064412172133,
        0.056690681665339054,
    ),
    ("mmcnn", "accuracy"): (0.7829447976356168, 0.06934000918431076),
    ("mmcnn", "balanced_accuracy"): (
        0.6376062291987468,
        0.057899664798745594,
    ),
    ("mmcnn", "f1_macro"): (0.5527824586394985, 0.05481774409829366),
}

EXPECTED_SOURCE_CELLS = {
    "0.8359 ± 0.0646",
    "0.5713 ± 0.0423",
    "0.5457 ± 0.0583",
}

OLD_RESULT_CELLS = {
    "71.15 ± 9.82",
    "55.80 ± 5.65",
    "47.63 ± 6.49",
    "77.93 ± 7.45",
    "63.68 ± 6.78",
    "55.11 ± 6.15",
    "0.7115 ± 0.0982",
    "0.5580 ± 0.0565",
    "0.4763 ± 0.0649",
    "0.7793 ± 0.0745",
    "0.6368 ± 0.0678",
    "0.5511 ± 0.0615",
}


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    result = subprocess.run(args, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}: {' '.join(args)}"
        )
    return result


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def patch_sleepedf_config(path: Path) -> None:
    require_file(path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'("SleepEDF"\s*:\s*dict\(\s*seq_len\s*=\s*)1024',
        r"\g<1>256",
        text,
    )
    if not re.search(
        r'"SleepEDF"\s*:\s*dict\(\s*seq_len\s*=\s*256',
        text,
    ):
        raise RuntimeError(f"Could not verify SleepEDF seq_len=256 in {path}")
    path.write_text(text, encoding="utf-8")


def patch_readme(path: Path) -> None:
    require_file(path)
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    text = text.replace("├── preprocessing_bci2a.py\n", "")

    tree_anchor = "├── merge_bimtta_results.py\n"
    if "├── run_sleepedf_baseline_shard.py\n" not in text:
        if tree_anchor not in text:
            raise RuntimeError("README repository tree anchor was not found")
        text = text.replace(
            tree_anchor,
            "├── run_sleepedf_baseline_shard.py\n"
            "├── merge_sleepedf_baseline_shards.py\n"
            "├── regenerate_main_paper_tables.py\n"
            + tree_anchor,
            1,
        )
    elif "├── regenerate_main_paper_tables.py\n" not in text:
        text = text.replace(
            "├── merge_sleepedf_baseline_shards.py\n",
            "├── merge_sleepedf_baseline_shards.py\n"
            "├── regenerate_main_paper_tables.py\n",
            1,
        )
    if "├── inject_styleprior_metrics.py\n" not in text:
        text = text.replace(
            "├── regenerate_main_paper_tables.py\n",
            "├── regenerate_main_paper_tables.py\n"
            "├── inject_styleprior_metrics.py\n",
            1,
        )

    dataset_paragraph = (
        "The datasets are not redistributed in this repository. Download them "
        "from their original providers and follow their access and usage "
        "conditions. The preprocessing code converts the inputs used by "
        "StylePrior-MOMENT to 16 channels, 256 Hz, and 256 samples per example. "
        "The cached Sleep-EDF array has shape `(n_epochs, 16, 256)`. The reported "
        "Sleep-EDF configurations of GraphSleepNet, SalientSleepNet, and MMCNN "
        "therefore use `seq_len=256`; no hidden padding or 1024-sample "
        "reconstruction is applied."
    )
    text, count = re.subn(
        r"The datasets are not redistributed in this repository\..*?"
        r"(?=\n\n## Running StylePrior-MOMENT)",
        dataset_paragraph,
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("README dataset-input paragraph was not found")

    protocol = (
        "All reported comparisons use the same subject partitions. For "
        "Sleep-EDF, the corrected GraphSleepNet, SalientSleepNet, and MMCNN "
        "configurations use the same 256-sample cached excerpts as the other "
        "evaluated methods. Architecture-specific processing remains inside "
        "each model."
    )
    text, count = re.subn(
        r"All reported comparisons use the same subject partitions\.[^\n]*",
        protocol,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("README baseline-protocol paragraph was not found")

    corrected_section = """### Corrected Sleep-EDF baseline runs

The earlier 1024-sample configuration was inconsistent with the cached 256-sample Sleep-EDF inputs. MMCNN and GraphSleepNet were rerun over all 20 LOSO folds after setting `seq_len=256`. SalientSleepNet did not require a numerical rerun because its `seq_len` field is not used in the forward computation, but its configuration was also corrected to 256.

| Method | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| GraphSleepNet | 72.27 ± 8.27 | 55.97 ± 5.40 | 47.92 ± 5.67 |
| MMCNN | 78.29 ± 6.93 | 63.76 ± 5.79 | 55.28 ± 5.48 |

The canonical merged outputs are `results/graphsleepnet_SleepEDF_seed2025.pkl` and `results/mmcnn_SleepEDF_seed2025.pkl`.

Example shard and merge commands:

    CUDA_VISIBLE_DEVICES=0 python run_sleepedf_baseline_shard.py \\
      --model mmcnn --start_fold 0 --end_fold 4

    python merge_sleepedf_baseline_shards.py \\
      --input_dir ./results_parallel_seq256 \\
      --result_dir ./results
"""
    section_pattern = (
        r"### Corrected Sleep-EDF baseline runs\n.*?"
        r"(?=\n## Ablation study)"
    )
    if re.search(section_pattern, text, flags=re.S):
        text = re.sub(
            section_pattern,
            corrected_section.rstrip(),
            text,
            count=1,
            flags=re.S,
        )
    else:
        heading = "\n## Ablation study"
        if heading not in text:
            raise RuntimeError("README Ablation study heading was not found")
        text = text.replace(
            heading,
            "\n\n" + corrected_section.rstrip() + heading,
            1,
        )

    collect_pattern = (
        r"Regenerate the manuscript-facing metric tables after replacing "
        r"result files:\n.*?(?=\nTo inspect a directory of serialized "
        r"experiment outputs without rerunning training:)"
    )
    collect_block = (
        "Regenerate the manuscript-facing metric tables after replacing "
        "result files:\n\n"
        "    python regenerate_main_paper_tables.py\n"
    )
    if re.search(collect_pattern, text, flags=re.S):
        text = re.sub(
            collect_pattern,
            collect_block.rstrip(),
            text,
            count=1,
            flags=re.S,
        )
    else:
        anchor = (
            "To inspect a directory of serialized experiment outputs without "
            "rerunning training:"
        )
        if anchor not in text:
            raise RuntimeError("README result-aggregation paragraph was not found")
        text = text.replace(anchor, collect_block + "\n" + anchor, 1)

    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_regenerator(path: Path) -> None:
    content = '''#!/usr/bin/env python3
"""Regenerate main-experiment tables without treating ablations as main models."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    results = Path("results")
    ablations = sorted(
        p for p in results.rglob("*.pkl") if "ablation" in p.name.lower()
    )
    with tempfile.TemporaryDirectory(prefix="styleprior_ablation_") as tmp_name:
        tmp = Path(tmp_name)
        moved = []
        for source in ablations:
            relative = source.relative_to(results)
            target = tmp / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((target, source))
        try:
            subprocess.run(
                [
                    sys.executable,
                    "collect_metrics_table.py",
                    "--roots",
                    "./results",
                    "./results_cbramod",
                    "./results_moment_internal_parallel",
                    "--out_dir",
                    "./paper_tables",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "inject_styleprior_metrics.py"],
                check=True,
            )
        finally:
            for temporary, original in moved:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temporary), str(original))


if __name__ == "__main__":
    main()
'''
    path.write_text(content, encoding="utf-8")


def write_styleprior_injector(path: Path) -> None:
    content = r'''#!/usr/bin/env python3
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
'''
    path.write_text(content, encoding="utf-8")


def normalize_csv_line_endings(directory: Path) -> None:
    for path in directory.glob("*.csv"):
        content = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(content)


def write_derived_metric_tables(directory: Path) -> None:
    """Build the core and accuracy Markdown views from the freshly written CSV."""
    source = directory / "paper_table_all_metrics.csv"
    require_file(source)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or len(rows[0]) < 3:
        raise RuntimeError(f"Unexpected table layout in {source}")

    header = rows[0]
    wanted_datasets = {"APAVA", "Sleep-EDF", "REFED"}
    wanted_metrics = {"Accuracy", "Balanced Accuracy", "F1 Score (Macro)"}
    current_dataset = ""
    core_rows: list[list[str]] = []
    accuracy_rows: list[list[str]] = []

    for raw_row in rows[1:]:
        row = list(raw_row) + [""] * (len(header) - len(raw_row))
        row = row[: len(header)]
        if row[0].strip():
            current_dataset = row[0].strip()
        metric = row[1].strip() if len(row) > 1 else ""
        if current_dataset not in wanted_datasets or metric not in wanted_metrics:
            continue
        normalized_row = [current_dataset, metric] + row[2:]
        core_rows.append(normalized_row)
        if metric == "Accuracy":
            accuracy_rows.append(normalized_row)

    if len(core_rows) != 9:
        raise RuntimeError(
            f"Expected 9 APAVA/Sleep-EDF/REFED core rows, found {len(core_rows)}"
        )
    if len(accuracy_rows) != 3:
        raise RuntimeError(f"Expected 3 accuracy rows, found {len(accuracy_rows)}")

    def markdown_table(selected_rows: list[list[str]]) -> str:
        escaped_header = [cell.replace("|", "\\|") for cell in header]
        lines = ["| " + " | ".join(escaped_header) + " |"]
        lines.append(
            "|---|---|" + "|".join("---:" for _ in escaped_header[2:]) + "|"
        )
        for row in selected_rows:
            escaped = [cell.replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(escaped) + " |")
        return "\n".join(lines) + "\n"

    (directory / "paper_table_core_metrics.md").write_text(
        markdown_table(core_rows), encoding="utf-8"
    )
    (directory / "paper_table_accuracy.md").write_text(
        markdown_table(accuracy_rows), encoding="utf-8"
    )


def validate_tables(root: Path) -> None:
    long_metrics = root / "paper_tables" / "long_metrics.csv"
    core_table = root / "paper_tables" / "paper_table_core_metrics.md"
    require_file(long_metrics)
    require_file(core_table)

    with long_metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    failures = []
    for (model, metric), (expected_mean, expected_std) in EXPECTED_BASELINES.items():
        matches = [
            row
            for row in rows
            if normalize(row.get("dataset", "")) == "sleepedf"
            and normalize(row.get("model", "")) == model
            and row.get("metric", "") == metric
        ]
        if len(matches) != 1:
            failures.append(
                f"{model}/{metric}: expected one row, found {len(matches)}"
            )
            continue
        mean = float(matches[0]["mean"])
        std = float(matches[0]["std"])
        if abs(mean - expected_mean) > 1e-12 or abs(std - expected_std) > 1e-12:
            failures.append(
                f"{model}/{metric}: got ({mean}, {std}), expected "
                f"({expected_mean}, {expected_std})"
            )

    core_text = core_table.read_text(encoding="utf-8")
    for cell in EXPECTED_SOURCE_CELLS:
        if cell not in core_text:
            failures.append(f"main source-model cell is missing: {cell}")

    readable_files = [root / "README.md"] + list((root / "paper_tables").glob("*"))
    combined = ""
    for path in readable_files:
        if path.is_file():
            try:
                combined += path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass
    for cell in OLD_RESULT_CELLS:
        if cell in combined:
            failures.append(f"old Sleep-EDF result remains: {cell}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    for cell in (
        "72.27 ± 8.27",
        "55.97 ± 5.40",
        "47.92 ± 5.67",
        "78.29 ± 6.93",
        "63.76 ± 5.79",
        "55.28 ± 5.48",
    ):
        if cell not in readme:
            failures.append(f"README result is missing: {cell}")

    if failures:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(failures))

    print("Validated corrected Sleep-EDF results:")
    print("  GraphSleepNet: 0.7227 / 0.5597 / 0.4792")
    print("  MMCNN:          0.7829 / 0.6376 / 0.5528")
    print("  Full source:    0.8359 / 0.5713 / 0.5457")
    print("  Old result cells: none")


def git_publish(root: Path) -> None:
    allowed_exact = {
        "README.md",
        "mmcnn_baseline.py",
        "graphsleepnet_baseline.py",
        "salientsleepnet_baseline.py",
        "run_sleepedf_baseline_shard.py",
        "merge_sleepedf_baseline_shards.py",
        "regenerate_main_paper_tables.py",
        "inject_styleprior_metrics.py",
        "results/mmcnn_SleepEDF_seed2025.pkl",
        "results/graphsleepnet_SleepEDF_seed2025.pkl",
    }
    already_staged = run(
        ["git", "diff", "--cached", "--name-only"],
    ).stdout.splitlines()
    unrelated = [
        name
        for name in already_staged
        if name not in allowed_exact and not name.startswith("paper_tables/")
    ]
    if unrelated:
        raise RuntimeError(
            "Unrelated files are already staged; nothing was committed:\n- "
            + "\n- ".join(unrelated)
        )

    run(
        [
            "git",
            "add",
            "README.md",
            "mmcnn_baseline.py",
            "graphsleepnet_baseline.py",
            "salientsleepnet_baseline.py",
            "run_sleepedf_baseline_shard.py",
            "merge_sleepedf_baseline_shards.py",
            "regenerate_main_paper_tables.py",
            "inject_styleprior_metrics.py",
            "paper_tables",
        ]
    )
    run(
        [
            "git",
            "add",
            "-f",
            "results/mmcnn_SleepEDF_seed2025.pkl",
            "results/graphsleepnet_SleepEDF_seed2025.pkl",
        ]
    )
    run(["git", "diff", "--cached", "--check"])

    staged = run(["git", "diff", "--cached", "--name-only"]).stdout.strip()
    if not staged:
        print("No new commit was needed; pushing the current main branch.")
    else:
        print("Files to commit:\n" + staged)
        run(
            [
                "git",
                "commit",
                "-m",
                "Correct Sleep-EDF baseline configuration and results",
            ]
        )

    run(["git", "pull", "--rebase", "origin", "main"])
    run(["git", "push", "origin", "main"])

    local_head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    remote_line = run(
        ["git", "ls-remote", "origin", "refs/heads/main"]
    ).stdout.strip()
    remote_head = remote_line.split()[0] if remote_line else ""
    if local_head != remote_head:
        raise RuntimeError(
            f"Push verification failed: local={local_head}, remote={remote_head}"
        )
    print(f"GitHub verified: origin/main = {local_head}")
    run(["git", "status", "--short"], check=False)
    run(["git", "log", "--oneline", "-3"], check=False)


def main() -> None:
    print(f"StylePrior GitHub finalizer {SCRIPT_VERSION}")
    root = Path.cwd()
    if not (root / ".git").exists():
        raise RuntimeError("Run this program from the StylePrior-MOMENT_EEG root")

    required = [
        root / "collect_metrics_table.py",
        root / "run_sleepedf_baseline_shard.py",
        root / "merge_sleepedf_baseline_shards.py",
        root / "results" / "mmcnn_SleepEDF_seed2025.pkl",
        root / "results" / "graphsleepnet_SleepEDF_seed2025.pkl",
        root / "results" / "ss_moment_SleepEDF_seed2025.pkl",
    ]
    for path in required:
        require_file(path)

    for filename in (
        "mmcnn_baseline.py",
        "graphsleepnet_baseline.py",
        "salientsleepnet_baseline.py",
    ):
        patch_sleepedf_config(root / filename)

    write_regenerator(root / "regenerate_main_paper_tables.py")
    write_styleprior_injector(root / "inject_styleprior_metrics.py")
    patch_readme(root / "README.md")
    run([sys.executable, "regenerate_main_paper_tables.py"])
    normalize_csv_line_endings(root / "paper_tables")
    write_derived_metric_tables(root / "paper_tables")
    validate_tables(root)
    git_publish(root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSTOPPED SAFELY: nothing further was pushed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "The terminal remains open. Copy the complete message above if help is needed.",
            file=sys.stderr,
        )
