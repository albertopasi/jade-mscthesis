"""Per-class precision/recall/F1/support table for JADE.

Writes CSV and a LaTeX booktabs-style table next to the figures.

Usage:
    uv run python -m src.visualization.make_per_class_table --task 9-class
    uv run python -m src.visualization.make_per_class_table --task binary
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.visualization._common import JADE_RUNS, PROJECT_ROOT, class_names, load_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["9-class", "binary"], required=True)
    args = ap.parse_args()

    stem = JADE_RUNS[args.task]
    summary, _ = load_run("jade", args.task, stem)
    names = class_names(args.task)
    pc = summary["classification_report"]["per_class"]
    macro = summary["classification_report"]["macro"]

    rows = []
    for i, name in enumerate(names):
        rows.append({
            "class": name,
            "precision": pc["precision"][i],
            "recall": pc["recall"][i],
            "f1": pc["f1"][i],
            "support": pc["support"][i],
        })
    rows.append({
        "class": "Macro avg",
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1"],
        "support": sum(pc["support"]),
    })

    out_dir = PROJECT_ROOT / "src" / "visualization" / f"jade_{args.task}" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "per_class_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        w.writeheader()
        w.writerows(rows)
    print(f"  saved → {csv_path.relative_to(PROJECT_ROOT)}")

    # LaTeX
    tex_path = out_dir / "per_class_metrics.tex"
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & Precision & Recall & F1 & Support \\",
        r"\midrule",
    ]
    for r in rows[:-1]:
        lines.append(
            f"{r['class']} & {r['precision']:.3f} & {r['recall']:.3f} "
            f"& {r['f1']:.3f} & {r['support']} \\\\"
        )
    lines.append(r"\midrule")
    r = rows[-1]
    lines.append(
        f"\\textbf{{{r['class']}}} & {r['precision']:.3f} & {r['recall']:.3f} "
        f"& {r['f1']:.3f} & {r['support']} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"  saved → {tex_path.relative_to(PROJECT_ROOT)}")

    # Pretty print
    print(f"\n  JADE per-class metrics ({args.task}):")
    print(f"  {'class':<14}  {'P':>6}  {'R':>6}  {'F1':>6}  {'support':>8}")
    for r in rows:
        print(f"  {r['class']:<14}  {r['precision']:>6.3f}  "
              f"{r['recall']:>6.3f}  {r['f1']:>6.3f}  {r['support']:>8}")


if __name__ == "__main__":
    main()
