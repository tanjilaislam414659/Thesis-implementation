"""
Summarize the Week 12 method comparison table.

This creates a compact summary comparing:
- ColPack heuristic baselines
- GNN learned ordering across multiple seeds
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_method_comparison_jac_pat.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/method_comparison_summary_jac_pat.csv"
)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def method_group(row: dict[str, str]) -> str:
    if row["method_family"] == "colpack":
        return f"ColPack {row['ordering_name']}"
    if row["method_family"] == "gnn":
        return "GNN learned ordering"
    return row["method_family"]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    rows = []

    with INPUT_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("No rows found in method comparison table.")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped_rows[method_group(row)].append(row)

    summary_rows = []

    for group_name, group_rows in grouped_rows.items():
        color_counts = [
            int(row["num_colors"])
            for row in group_rows
        ]

        valid_flags = [
            parse_bool(row["valid"])
            for row in group_rows
        ]

        runtimes = [
            float(row["runtime_seconds"])
            for row in group_rows
        ]

        summary_rows.append(
            {
                "graph_id": group_rows[0]["graph_id"],
                "method_group": group_name,
                "num_runs": len(group_rows),
                "min_colors": min(color_counts),
                "max_colors": max(color_counts),
                "mean_colors": sum(color_counts) / len(color_counts),
                "all_valid": all(valid_flags),
                "mean_runtime_seconds": sum(runtimes) / len(runtimes),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "graph_id",
            "method_group",
            "num_runs",
            "min_colors",
            "max_colors",
            "mean_colors",
            "all_valid",
            "mean_runtime_seconds",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Method comparison summary")
    print("-------------------------")

    for row in summary_rows:
        print(
            f"{row['method_group']} | "
            f"runs={row['num_runs']} | "
            f"colors={row['mean_colors']:.3f} | "
            f"valid={row['all_valid']}"
        )

    print()
    print(f"Saved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()