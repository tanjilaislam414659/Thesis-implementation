"""
Build a method comparison table for ColPack baselines and multiple GNN runs.

This table is used for Week 12 evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path


COLPACK_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/colpack_initial_benchmark.csv"
)

GNN_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_learned_coloring_evaluation.csv"
)

OUTPUT_CSV = Path(
    "results/tables/gnn_node_scorer/multiple_run_method_comparison_jac_pat.csv"
)

TEST_GRAPH_ID = "jac_pat"


def load_colpack_rows() -> list[dict[str, object]]:
    rows = []

    with COLPACK_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["graph_id"] != TEST_GRAPH_ID:
                continue

            rows.append(
                {
                    "graph_id": row["graph_id"],
                    "seed": "",
                    "method_family": row["method_family"],
                    "method_name": row["method_name"],
                    "ordering_name": row["ordering_name"],
                    "num_vertices": int(row["num_vertices"]),
                    "num_edges": int(row["num_edges"]),
                    "num_colors": int(row["num_colors"]),
                    "valid": row["valid"],
                    "runtime_seconds": row["runtime"],
                }
            )

    if not rows:
        raise ValueError(f"No ColPack rows found for graph_id={TEST_GRAPH_ID}")

    return rows


def load_gnn_rows() -> list[dict[str, object]]:
    rows = []

    with GNN_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["graph_id"] != TEST_GRAPH_ID:
                continue

            rows.append(
                {
                    "graph_id": row["graph_id"],
                    "seed": row["seed"],
                    "method_family": row["method_family"],
                    "method_name": row["method_name"],
                    "ordering_name": row["ordering_name"],
                    "num_vertices": int(row["num_vertices"]),
                    "num_edges": int(row["num_edges"]),
                    "num_colors": int(row["num_colors"]),
                    "valid": row["valid"],
                    "runtime_seconds": row["runtime_seconds"],
                }
            )

    if not rows:
        raise ValueError(f"No GNN rows found for graph_id={TEST_GRAPH_ID}")

    return rows


def main() -> None:
    rows = load_colpack_rows() + load_gnn_rows()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graph_id",
        "seed",
        "method_family",
        "method_name",
        "ordering_name",
        "num_vertices",
        "num_edges",
        "num_colors",
        "valid",
        "runtime_seconds",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Multiple-run method comparison table")
    print("------------------------------------")

    for row in rows:
        seed_text = f"seed={row['seed']}" if row["seed"] != "" else "baseline"
        print(
            f"{row['graph_id']} | "
            f"{seed_text} | "
            f"{row['method_family']} | "
            f"{row['ordering_name']} | "
            f"{row['num_colors']} colors | "
            f"valid={row['valid']}"
        )

    print()
    print(f"Saved comparison table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()