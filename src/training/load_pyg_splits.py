"""
Load saved PyTorch Geometric graph objects and group them by dataset split.

The expected split setup is:
- train: ash85, can_24, hess_pat
- validation: hess_pat_small
- test: jac_pat
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data


PYG_DATA_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/pyg_data"
)

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test",
}


def load_all_pyg_graphs(
    pyg_data_dir: str | Path = PYG_DATA_DIR,
) -> list[Data]:
    """
    Load all saved PyTorch Geometric graph objects.
    """

    pyg_data_dir = Path(pyg_data_dir)

    if not pyg_data_dir.exists():
        raise FileNotFoundError(f"PyG data directory not found: {pyg_data_dir}")

    pyg_files = sorted(pyg_data_dir.glob("*.pt"))

    if not pyg_files:
        raise FileNotFoundError(f"No .pt files found in: {pyg_data_dir}")

    dataset = []

    for pyg_file in pyg_files:
        data = torch.load(pyg_file, weights_only=False)

        if not hasattr(data, "graph_id"):
            raise ValueError(f"{pyg_file.name}: missing graph_id.")

        if not hasattr(data, "split"):
            raise ValueError(f"{data.graph_id}: missing split.")

        if data.split not in EXPECTED_SPLITS:
            raise ValueError(
                f"{data.graph_id}: unexpected split '{data.split}'."
            )

        if not hasattr(data, "y") or data.y is None:
            raise ValueError(f"{data.graph_id}: missing ordering target tensor data.y.")

        dataset.append(data)

    return dataset


def group_dataset_by_split(
    dataset: list[Data],
) -> dict[str, list[Data]]:
    """
    Group loaded PyG graphs into train, validation, and test splits.
    """

    grouped = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for data in dataset:
        grouped[data.split].append(data)

    return grouped


def main() -> None:
    dataset = load_all_pyg_graphs()
    grouped = group_dataset_by_split(dataset)

    print("PyG split loader check")
    print("----------------------")

    for split_name in ["train", "validation", "test"]:
        graphs = grouped[split_name]

        print(f"{split_name.capitalize()} graphs: {len(graphs)}")

        for data in graphs:
            print(
                f"  {data.graph_id}: "
                f"x={tuple(data.x.shape)}, "
                f"edge_index={tuple(data.edge_index.shape)}, "
                f"y={tuple(data.y.shape)}"
            )

        print()

    print("All PyG split loading checks passed.")


if __name__ == "__main__":
    main()