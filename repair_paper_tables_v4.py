#!/usr/bin/env python3
"""Repair StylePrior-MOMENT paper-table generation and validate canonical values."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


VERSION = "2026-07-15-v4"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def patch_injector(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    ap_guard = '''                if (
                    dataset == "Sleep-EDF"
                    and display == "Avg Prec"
                    and values
                    and all(abs(value) < 1e-12 for value in values)
                ):
                    values = []
'''
    ap_anchor = '''                if values:
                    summaries[(dataset, variant, display)] = (
'''
    if ap_guard not in text:
        if ap_anchor not in text:
            raise RuntimeError("Could not find the metric-summary insertion point.")
        text = text.replace(ap_anchor, ap_guard + ap_anchor, 1)

    derived_block = '''
    core_displays = {"Accuracy", "Balanced Accuracy", "F1 Score (Macro)"}
    core_rows = [new_rows[0]] + [
        row for row in new_rows[1:]
        if len(row) > 1 and row[1] in core_displays
    ]
    accuracy_rows = [new_rows[0]] + [
        row for row in new_rows[1:]
        if len(row) > 1 and row[1] == "Accuracy"
    ]
    write_markdown(Path("paper_tables/paper_table_core_metrics.md"), core_rows)
    write_markdown(Path("paper_tables/paper_table_accuracy.md"), accuracy_rows)
'''
    derived_anchor = '''    write_markdown(Path("paper_tables/paper_table_all_metrics.md"), new_rows)
'''
    marker = 'write_markdown(Path("paper_tables/paper_table_core_metrics.md"), core_rows)'
    if marker not in text:
        if derived_anchor not in text:
            raise RuntimeError("Could not find the derived-table insertion point.")
        text = text.replace(derived_anchor, derived_anchor + derived_block, 1)

    path.write_text(text, encoding="utf-8")


def validate() -> None:
    wide_path = Path("paper_tables/paper_table_all_metrics.csv")
    with wide_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RuntimeError(f"Empty table: {wide_path}")

    header = rows[0]
    stsa_col = header.index("StylePrior-MOMENT + STSA")
    source_col = header.index("StylePrior-MOMENT (source)")
    current_dataset = ""
    found_ap = False
    for raw in rows[1:]:
        row = list(raw) + [""] * (len(header) - len(raw))
        if row[0].strip():
            current_dataset = row[0].strip()
        if current_dataset == "Sleep-EDF" and row[1].strip() == "Avg Prec":
            found_ap = True
            if row[stsa_col].strip() or row[source_col].strip():
                raise RuntimeError("Sleep-EDF unavailable StylePrior AP is not blank.")
    if not found_ap:
        raise RuntimeError("Sleep-EDF Avg Prec row was not found.")

    with Path("paper_tables/long_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        bad_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == "Sleep-EDF"
            and "styleprior" in row.get("model", "").lower()
            and "average_precision" in row.get("metric", "").lower()
        ]
    if bad_rows:
        raise RuntimeError(f"Invalid zero AP rows remain: {bad_rows}")

    core = Path("paper_tables/paper_table_core_metrics.md").read_text(
        encoding="utf-8"
    )
    expected = [
        "0.7227 ± 0.0827",
        "0.5597 ± 0.0540",
        "0.4792 ± 0.0567",
        "0.7829 ± 0.0693",
        "0.6376 ± 0.0579",
        "0.5528 ± 0.0548",
        "0.8359 ± 0.0646",
        "0.5713 ± 0.0423",
        "0.5457 ± 0.0583",
    ]
    missing = [value for value in expected if value not in core]
    if missing:
        raise RuntimeError(f"Core results are missing: {missing}")

    accuracy = Path("paper_tables/paper_table_accuracy.md").read_text(
        encoding="utf-8"
    )
    if "0.8359 ± 0.0646" not in accuracy:
        raise RuntimeError("Source-model Sleep-EDF accuracy is missing.")

    print("PASS: tables are consistent; unavailable AP is blank.")


def main() -> None:
    print(f"StylePrior paper-table repair {VERSION}")
    required = [
        Path("inject_styleprior_metrics.py"),
        Path("regenerate_main_paper_tables.py"),
        Path("paper_tables"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Run this script from the repository root. Missing: " + ", ".join(missing)
        )

    patch_injector(Path("inject_styleprior_metrics.py"))
    run(sys.executable, "regenerate_main_paper_tables.py")
    validate()

    files = [
        "inject_styleprior_metrics.py",
        "paper_tables/long_metrics.csv",
        "paper_tables/paper_table_all_metrics.csv",
        "paper_tables/paper_table_all_metrics.md",
        "paper_tables/paper_table_core_metrics.md",
        "paper_tables/paper_table_accuracy.md",
    ]
    run("git", "add", *files)
    run("git", "diff", "--cached", "--check")
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], check=False
    ).returncode != 0
    if staged:
        run("git", "commit", "-m", "Treat unavailable metrics as missing in paper tables")
        print("Local commit created. The only remaining command is: git push origin main")
    else:
        print("No new commit was needed. The tables already match the repaired generator.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOPPED SAFELY: {exc}", file=sys.stderr)
        raise SystemExit(1)
