"""
Check that saved PyTorch Geometric Data objects can be loaded correctly.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.build_pyg_dataset import validate_pyg_data


def main() -> None:
    pyg_dir = Path("data/processed/initial_graph_coloring_dataset/pyg_data")

    if not pyg_dir.exists():
        raise FileNotFoundError(f"PyG data directory not found: {pyg_dir}")

    pyg_files = sorted(pyg_dir.glob("*.pt"))

    if not pyg_files:
        raise FileNotFoundError(f"No .pt files found in {pyg_dir}")

    print("Saved PyG Data loading check")
    print("----------------------------")

    for pyg_file in pyg_files:
        data = torch.load(pyg_file, weights_only=False)

        validate_pyg_data(data)

        print(f"Loaded file: {pyg_file.name}")
        print(f"  graph_id: {data.graph_id}")
        print(f"  split: {data.split}")
        print(f"  x shape: {tuple(data.x.shape)}")
        print(f"  edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  num_nodes: {data.num_nodes}")
        print("  status: OK")
        print()

    print("All saved PyG Data files loaded successfully.")


if __name__ == "__main__":
    main()