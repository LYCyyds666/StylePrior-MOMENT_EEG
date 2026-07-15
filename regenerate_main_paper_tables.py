#!/usr/bin/env python3
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
