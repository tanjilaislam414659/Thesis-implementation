"""
Utilities for loading node-level ordering targets for the GNN pipeline.

The current target source is the ColPack SMALLEST_LAST ordering target CSV.
Targets are returned in node-ID order so that they align with PyTorch Geometric
node feature matrices, where row i corresponds to node i.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


TARGET_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "ordering_targets/smallest_last_ordering_targets.csv"
)


def load_ordering_targets_by_graph(
    target_csv: str | Path = TARGET_CSV,
) -> dict[str, dict[int, float]]:
    """
    Load ordering target scores grouped by graph and node ID.

    Returns
    -------
    dict[str, dict[int, float]]
        Example:
        {
            "ash85": {
                0: 0.40,
                1: 0.72,
                ...
            }
        }
    """

    target_csv = Path(target_csv)

    if not target_csv.exists():
        raise FileNotFoundError(f"Ordering target CSV not found: {target_csv}")

    targets: dict[str, dict[int, float]] = {}

    with target_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "graph_id",
            "node_id",
            "target_score",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Target CSV must contain columns {required_columns}, "
                f"got {reader.fieldnames}."
            )

        for row in reader:
            graph_id = row["graph_id"]
            node_id = int(row["node_id"])
            target_score = float(row["target_score"])

            if graph_id not in targets:
                targets[graph_id] = {}

            if node_id in targets[graph_id]:
                raise ValueError(
                    f"Duplicate target score for graph '{graph_id}', node {node_id}."
                )

            targets[graph_id][node_id] = target_score

    return targets


def build_target_tensor_for_graph(
    graph_id: str,
    num_nodes: int,
    target_csv: str | Path = TARGET_CSV,
) -> torch.Tensor:
    """
    Build a node-level target tensor ordered by node ID.

    The returned tensor has shape [num_nodes, 1].
    """

    targets_by_graph = load_ordering_targets_by_graph(target_csv)

    if graph_id not in targets_by_graph:
        raise ValueError(f"No ordering targets found for graph: {graph_id}")

    graph_targets = targets_by_graph[graph_id]

    expected_node_ids = set(range(num_nodes))
    actual_node_ids = set(graph_targets.keys())

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{graph_id}: target node IDs do not match expected range "
            f"0 to {num_nodes - 1}."
        )

    ordered_scores = [
        graph_targets[node_id]
        for node_id in range(num_nodes)
    ]

    target_tensor = torch.tensor(ordered_scores, dtype=torch.float32).view(-1, 1)

    return target_tensor


def main() -> None:
    targets_by_graph = load_ordering_targets_by_graph()

    print("Ordering target tensor loading check")
    print("------------------------------------")

    for graph_id, node_targets in targets_by_graph.items():
        target_tensor = build_target_tensor_for_graph(
            graph_id=graph_id,
            num_nodes=len(node_targets),
        )

        print(f"Graph ID: {graph_id}")
        print(f"  number of targets: {len(node_targets)}")
        print(f"  tensor shape: {tuple(target_tensor.shape)}")
        print(f"  min target score: {float(target_tensor.min()):.1f}")
        print(f"  max target score: {float(target_tensor.max()):.1f}")
        print("  status: OK")
        print()

    print("All ordering target tensors loaded successfully.")


if __name__ == "__main__":
    main()