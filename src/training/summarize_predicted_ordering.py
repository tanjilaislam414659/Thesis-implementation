"""
Summarize the predicted ordering comparison for the test graph.

This script reads the node-level predicted-vs-target ordering CSV and computes
simple ranking-quality summary statistics.
"""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/jac_pat_predicted_ordering_comparison.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/jac_pat_predicted_ordering_summary.csv"
)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    rows = []

    with INPUT_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("No rows found in predicted ordering comparison CSV.")

    graph_id = rows[0]["graph_id"]

    absolute_errors = [
        int(row["absolute_position_error"])
        for row in rows
    ]

    predicted_top_10 = {
        int(row["node_id"])
        for row in rows
        if int(row["predicted_position"]) < 10
    }

    target_top_10 = {
        int(row["node_id"])
        for row in rows
        if int(row["target_position"]) < 10
    }

    predicted_top_15 = {
        int(row["node_id"])
        for row in rows
        if int(row["predicted_position"]) < 15
    }

    target_top_15 = {
        int(row["node_id"])
        for row in rows
        if int(row["target_position"]) < 15
    }

    top_10_overlap = len(predicted_top_10.intersection(target_top_10))
    top_15_overlap = len(predicted_top_15.intersection(target_top_15))

    mean_absolute_position_error = sum(absolute_errors) / len(absolute_errors)
    max_absolute_position_error = max(absolute_errors)

    summary_row = {
        "graph_id": graph_id,
        "num_nodes": len(rows),
        "mean_absolute_position_error": mean_absolute_position_error,
        "max_absolute_position_error": max_absolute_position_error,
        "top_10_overlap": top_10_overlap,
        "top_15_overlap": top_15_overlap,
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print("Predicted ordering summary")
    print("--------------------------")
    print(f"Graph ID: {graph_id}")
    print(f"Number of nodes: {len(rows)}")
    print(f"Mean absolute position error: {mean_absolute_position_error:.3f}")
    print(f"Max absolute position error: {max_absolute_position_error}")
    print(f"Top-10 overlap: {top_10_overlap} / 10")
    print(f"Top-15 overlap: {top_15_overlap} / 15")
    print()
    print(f"Saved summary CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()