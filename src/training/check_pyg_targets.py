"""
Check whether saved PyTorch Geometric graph objects contain
ordering target tensors where expected.

Aligned graphs should contain:
    data.y with shape [num_nodes, 1]

jac_pat should currently have no data.y target attached because
its ColPack graph representation is not aligned with the PyG graph.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.build_pyg_dataset import validate_pyg_data


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/pyg_data"
)

GRAPHS_EXPECTED_TO_HAVE_TARGETS = {
    "ash85",
    "can_24",
    "hess_pat",
    "hess_pat_small",
}

GRAPHS_EXPECTED_WITHOUT_TARGETS = {
    "jac_pat",
}


def main() -> None:
    pyg_files = sorted(PYG_DATA_DIR.glob("*.pt"))

    if not pyg_files:
        raise FileNotFoundError(f"No .pt files found in {PYG_DATA_DIR}")

    print("PyG ordering-target attachment check")
    print("------------------------------------")

    for pyg_file in pyg_files:
        data = torch.load(pyg_file, weights_only=False)
        validate_pyg_data(data)

        graph_id = data.graph_id

        print(f"Graph ID: {graph_id}")

        if graph_id in GRAPHS_EXPECTED_TO_HAVE_TARGETS:
            if not hasattr(data, "y") or data.y is None:
                raise ValueError(f"{graph_id}: expected data.y, but it is missing.")

            if data.y.shape != (data.num_nodes, 1):
                raise ValueError(
                    f"{graph_id}: expected y shape {(data.num_nodes, 1)}, "
                    f"got {tuple(data.y.shape)}."
                )

            print(f"  y shape: {tuple(data.y.shape)}")
            print(f"  y score range: [{float(data.y.min()):.1f}, {float(data.y.max()):.1f}]")
            print("  target status: attached")

        elif graph_id in GRAPHS_EXPECTED_WITHOUT_TARGETS:
            if hasattr(data, "y") and data.y is not None:
                raise ValueError(
                    f"{graph_id}: data.y should not be attached yet, but it exists."
                )

            print("  target status: intentionally not attached")

        else:
            raise ValueError(f"Unexpected graph_id found: {graph_id}")

        print()

    print("All PyG target attachment checks passed.")


if __name__ == "__main__":
    main()