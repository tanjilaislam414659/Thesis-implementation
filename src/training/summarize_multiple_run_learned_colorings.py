"""
Summarize multiple-run learned coloring results.

This script reads the learned coloring evaluation table for all repeated
GNN runs and computes simple aggregate statistics over color counts,
validity, and runtime.
"""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_learned_coloring_evaluation.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_learned_coloring_summary.csv"
)


def parse_bool(value: str) -> bool:
    """
    Parse a CSV boolean value.
    """

    return value.strip().lower() in {"true", "1", "yes"}


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    rows = []

    with INPUT_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("No rows found in learned coloring evaluation CSV.")

    graph_id = rows[0]["graph_id"]

    color_counts = [
        int(row["num_colors"])
        for row in rows
    ]

    valid_flags = [
        parse_bool(row["valid"])
        for row in rows
    ]

    runtimes = [
        float(row["runtime_seconds"])
        for row in rows
    ]

    summary_row = {
        "graph_id": graph_id,
        "num_runs": len(rows),
        "min_colors": min(color_counts),
        "max_colors": max(color_counts),
        "mean_colors": sum(color_counts) / len(color_counts),
        "all_colorings_valid": all(valid_flags),
        "mean_runtime_seconds": sum(runtimes) / len(runtimes),
        "min_runtime_seconds": min(runtimes),
        "max_runtime_seconds": max(runtimes),
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print("Multiple-run learned coloring summary")
    print("-------------------------------------")
    print(f"Graph ID: {graph_id}")
    print(f"Number of runs: {summary_row['num_runs']}")
    print(f"Min colors: {summary_row['min_colors']}")
    print(f"Max colors: {summary_row['max_colors']}")
    print(f"Mean colors: {summary_row['mean_colors']:.3f}")
    print(f"All colorings valid: {summary_row['all_colorings_valid']}")
    print(f"Mean runtime: {summary_row['mean_runtime_seconds']:.6f}s")
    print()
    print(f"Saved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
    