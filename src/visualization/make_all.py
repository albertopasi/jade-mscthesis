"""Run all visualization scripts for both tasks in one shot.

Usage:
    uv run python -m src.visualization.make_all              # no titles (default)
    uv run python -m src.visualization.make_all --title      # with titles
    uv run python -m src.visualization.make_all --task 9-class
"""
from __future__ import annotations

import argparse
import subprocess
import sys

SCRIPTS = [
    "make_confusion_matrix",
    "make_subject_histogram",
    "make_per_class_table",
    "make_per_class_f1_bars",
    "make_paired_scatter",
]
TASKS = ["9-class", "binary"]
# Scripts that don't have a --title flag (tables, not figures)
NO_TITLE_SCRIPTS = {"make_per_class_table"}
# Scripts that also produce a --with-lp variant
WITH_LP_SCRIPTS = {"make_per_class_f1_bars", "make_paired_scatter"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS + ["all"], default="all")
    ap.add_argument("--title", action="store_true", help="Pass --title to figure scripts.")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    failed: list[str] = []

    for task in tasks:
        print(f"\n============ TASK: {task} ============")
        for script in SCRIPTS:
            variants = [[]]
            if script in WITH_LP_SCRIPTS:
                variants.append(["--with-lp"])
            for extra in variants:
                cmd = [sys.executable, "-m", f"src.visualization.{script}", "--task", task] + extra
                if args.title and script not in NO_TITLE_SCRIPTS:
                    cmd.append("--title")
                print(f"--- {' '.join(cmd[2:])} ---")
                r = subprocess.run(cmd)
                if r.returncode != 0:
                    failed.append(f"{script} {task} {' '.join(extra)}".strip())

    print("\n============ DONE ============")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All scripts completed.")


if __name__ == "__main__":
    main()
