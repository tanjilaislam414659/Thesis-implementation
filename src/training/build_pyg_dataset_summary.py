"""
Create a CSV summary of the saved PyTorch Geometric dataset files.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.training.build_pyg_dataset import validate_pyg_data


def main() -> None:
    pyg_dir = Path("data/processed/initial_graph_coloring_dataset/pyg_data")
    output_path = Path(
        "data/processed/initial_graph_coloring_dataset/pyg_data/pyg_dataset_summary.csv"
    )

    if not pyg_dir.exists():
        raise FileNotFoundError(f"PyG data directory not found: {pyg_dir}")

    pyg_files = sorted(pyg_dir.glob("*.pt"))

    if not pyg_files:
        raise FileNotFoundError(f"No .pt files found in {pyg_dir}")

    rows = []

    for pyg_file in pyg_files:
        data = torch.load(pyg_file, weights_only=False)
        validate_pyg_data(data)

        rows.append(
            {
                "graph_id": data.graph_id,
                "split": data.split,
                "num_nodes": data.num_nodes,
                "num_directed_edges": data.edge_index.shape[1],
                "num_features": data.x.shape[1],
                "pyg_file": str(pyg_file),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "graph_id",
            "split",
            "num_nodes",
            "num_directed_edges",
            "num_features",
            "pyg_file",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved PyG dataset summary to: {output_path}")
    print()
    print("Summary rows:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()