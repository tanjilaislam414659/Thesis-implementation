"""
Validate BEST_AVAILABLE_OF_5 ordering targets for the Week 16 larger-graph extension.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


TARGET_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/ordering_targets/"
    "best_available_of_5_ordering_targets_week16_larger_graphs.csv"
)

EXPECTED_VERTICES = {
    "bcsstk10": 1086,
    "bcsstk14": 1806,
    "bcsstk15": 3948,
}


def main() -> None:
    df = pd.read_csv(TARGET_CSV)

    print("Checking Week 16 larger-graph ordering targets")
    print("----------------------------------------------")

    for graph_id, expected_vertices in EXPECTED_VERTICES.items():
        graph_df = df[df["graph_id"] == graph_id]

        row_count = len(graph_df)
        unique_node_ids = graph_df["node_id"].nunique()
        unique_positions = graph_df["order_position"].nunique()

        score_min = graph_df["target_score"].min()
        score_max = graph_df["target_score"].max()

        rows_match = row_count == expected_vertices
        nodes_unique = unique_node_ids == expected_vertices
        positions_unique = unique_positions == expected_vertices
        scores_valid = score_min >= 0.0 and score_max <= 1.0

        print(f"{graph_id}:")
        print(f"  rows: {row_count} / expected {expected_vertices} -> {rows_match}")
        print(f"  unique node ids: {unique_node_ids} -> {nodes_unique}")
        print(f"  unique positions: {unique_positions} -> {positions_unique}")
        print(f"  score range: [{score_min:.6f}, {score_max:.6f}] -> {scores_valid}")

        if not (rows_match and nodes_unique and positions_unique and scores_valid):
            raise ValueError(f"Validation failed for {graph_id}")

    print()
    print("All larger-graph ordering target checks passed.")


if __name__ == "__main__":
    main()