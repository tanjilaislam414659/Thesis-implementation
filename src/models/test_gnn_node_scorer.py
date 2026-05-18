"""
Basic forward-pass test for the first GNN node-scoring model.

This script loads one saved PyTorch Geometric graph object,
runs the GNNNodeScorer model, and checks that the model returns
one scalar score per node.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.gnn_node_scorer import GNNNodeScorer


def main() -> None:
    graph_path = Path(
        "data/processed/initial_graph_coloring_dataset/pyg_data/ash85.pt"
    )

    data = torch.load(graph_path, weights_only=False)

    model = GNNNodeScorer(
        in_channels=data.x.shape[1],
        hidden_channels=32,
        out_channels=1,
    )

    scores = model(data.x, data.edge_index)

    expected_shape = (data.num_nodes, 1)

    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            f"Expected score shape {expected_shape}, got {tuple(scores.shape)}."
        )

    print("GNN node scorer forward-pass check")
    print("----------------------------------")
    print(f"Graph ID: {data.graph_id}")
    print(f"Input x shape: {tuple(data.x.shape)}")
    print(f"Input edge_index shape: {tuple(data.edge_index.shape)}")
    print(f"Output score shape: {tuple(scores.shape)}")
    print("Status: OK")


if __name__ == "__main__":
    main()