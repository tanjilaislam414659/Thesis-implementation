"""
Check the first GNN node-scoring model on all PyG graph objects
that currently have valid ordering targets attached.

The script verifies that:
- the model runs without error,
- the output shape is [num_nodes, 1],
- the output shape matches the target tensor shape data.y.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.gnn_node_scorer import GNNNodeScorer


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/pyg_data"
)

TARGET_ENABLED_GRAPHS = {
    "ash85",
    "can_24",
    "hess_pat",
    "hess_pat_small",
    "jac_pat",
}


def main() -> None:
    pyg_files = sorted(PYG_DATA_DIR.glob("*.pt"))

    if not pyg_files:
        raise FileNotFoundError(f"No .pt files found in {PYG_DATA_DIR}")

    print("GNN node scorer all-graph forward-pass check")
    print("--------------------------------------------")

    for pyg_file in pyg_files:
        data = torch.load(pyg_file, weights_only=False)

        if data.graph_id not in TARGET_ENABLED_GRAPHS:
            continue

        if not hasattr(data, "y") or data.y is None:
            raise ValueError(f"{data.graph_id}: expected data.y target, but it is missing.")

        model = GNNNodeScorer(
            in_channels=data.x.shape[1],
            hidden_channels=32,
            out_channels=1,
        )

        scores = model(data.x, data.edge_index)

        expected_shape = (data.num_nodes, 1)

        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"{data.graph_id}: expected score shape {expected_shape}, "
                f"got {tuple(scores.shape)}."
            )

        if tuple(scores.shape) != tuple(data.y.shape):
            raise ValueError(
                f"{data.graph_id}: score shape {tuple(scores.shape)} "
                f"does not match target shape {tuple(data.y.shape)}."
            )

        print(f"Graph ID: {data.graph_id}")
        print(f"  x shape: {tuple(data.x.shape)}")
        print(f"  edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  y shape: {tuple(data.y.shape)}")
        print(f"  score shape: {tuple(scores.shape)}")
        print("  status: OK")
        print()

    print("All GNN forward-pass checks passed.")


if __name__ == "__main__":
    main()