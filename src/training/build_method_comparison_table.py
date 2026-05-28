"""
Build a comparison table between ColPack heuristic colorings
and the learned GNN ordering result.

This creates a compact table for the current test graph comparison.
"""

from __future__ import annotations

import csv
from pathlib import Path


COLPACK_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/colpack_initial_benchmark.csv"
)

LEARNED_CSV = Path(
    "results/tables/gnn_node_scorer/learned_coloring_evaluation.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/method_comparison_jac_pat.csv"
)


TEST_GRAPH_ID = "jac_pat"


def load_colpack_rows() -> list[dict[str, object]]:
    """
    Load ColPack benchmark rows for the test graph.
    """

    if not COLPACK_CSV.exists():
        raise FileNotFoundError(f"ColPack benchmark CSV not found: {COLPACK_CSV}")

    rows = []

    with COLPACK_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["graph_id"] != TEST_GRAPH_ID:
                continue

            rows.append(
                {
                    "graph_id": row["graph_id"],
                    "method_family": row["method_family"],
                    "method_name": row["method_name"],
                    "ordering_name": row["ordering_name"],
                    "num_vertices": int(row["num_vertices"]),
                    "num_edges": int(row["num_edges"]),
                    "num_colors": int(row["num_colors"]),
                    "valid": row["valid"],
                }
            )

    if not rows:
        raise ValueError(f"No ColPack rows found for graph_id={TEST_GRAPH_ID}")

    return rows


def load_learned_rows() -> list[dict[str, object]]:
    """
    Load learned GNN coloring evaluation rows.
    """

    if not LEARNED_CSV.exists():
        raise FileNotFoundError(f"Learned evaluation CSV not found: {LEARNED_CSV}")

    rows = []

    with LEARNED_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["graph_id"] != TEST_GRAPH_ID:
                continue

            rows.append(
                {
                    "graph_id": row["graph_id"],
                    "method_family": row["method_family"],
                    "method_name": row["method_name"],
                    "ordering_name": row["ordering_name"],
                    "num_vertices": int(row["num_vertices"]),
                    "num_edges": int(row["num_edges"]),
                    "num_colors": int(row["num_colors"]),
                    "valid": row["valid"],
                }
            )

    if not rows:
        raise ValueError(f"No learned rows found for graph_id={TEST_GRAPH_ID}")

    return rows


def main() -> None:
    rows = load_colpack_rows() + load_learned_rows()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graph_id",
        "method_family",
        "method_name",
        "ordering_name",
        "num_vertices",
        "num_edges",
        "num_colors",
        "valid",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Method comparison table")
    print("-----------------------")

    for row in rows:
        print(
            f"{row['graph_id']} | "
            f"{row['method_family']} | "
            f"{row['ordering_name']} | "
            f"{row['num_colors']} colors | "
            f"valid={row['valid']}"
        )

    print()
    print(f"Saved comparison table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()