"""
Attach node-level SMALLEST_LAST ordering targets to saved PyTorch Geometric Data objects.

For aligned square-matrix graphs, this script adds:

    data.y = target tensor of shape [num_nodes, 1]

The rectangular jac_pat graph is intentionally skipped because its current
ColPack ordering output was generated from a different graph representation
than the Python/PyTorch Geometric column-intersection graph.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.build_pyg_dataset import validate_pyg_data
from src.training.ordering_targets import build_target_tensor_for_graph


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/pyg_data"
)

SKIP_GRAPH_IDS = set()


def attach_targets_to_saved_pyg_data() -> None:
    """
    Load saved PyG graph files, attach ordering targets where valid,
    and save the updated graph objects back to disk.
    """

    pyg_files = sorted(PYG_DATA_DIR.glob("*.pt"))

    if not pyg_files:
        raise FileNotFoundError(f"No .pt files found in {PYG_DATA_DIR}")

    print("Attach ordering targets to PyG graph objects")
    print("--------------------------------------------")

    for pyg_file in pyg_files:
        data = torch.load(pyg_file, weights_only=False)
        validate_pyg_data(data)

        graph_id = data.graph_id

        if graph_id in SKIP_GRAPH_IDS:
            print(f"Skipped {graph_id}")
            print("  reason: ColPack/PyG graph representation mismatch")
            print()
            continue

        target_tensor = build_target_tensor_for_graph(
            graph_id=graph_id,
            num_nodes=data.num_nodes,
        )

        data.y = target_tensor

        torch.save(data, pyg_file)

        print(f"Updated {graph_id}")
        print(f"  saved file: {pyg_file}")
        print(f"  y shape: {tuple(data.y.shape)}")
        print(f"  y score range: [{float(data.y.min()):.1f}, {float(data.y.max()):.1f}]")
        print()

    print("Ordering-target attachment step completed.")


def main() -> None:
    attach_targets_to_saved_pyg_data()


if __name__ == "__main__":
    main()