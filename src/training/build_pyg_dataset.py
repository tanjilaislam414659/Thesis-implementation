"""
Build PyTorch Geometric Data objects from the initial graph-coloring dataset.

This module converts each sparse matrix graph into a PyG Data object containing:
1. x           - node feature matrix
2. edge_index  - graph connectivity
3. graph_id    - graph identifier
4. split       - train / validation / test assignment
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.node_features import extract_node_features


GRAPH_ID_TO_MATRIX = {
    "ash85": "ash85.mtx",
    "can_24": "can_24.mtx",
    "hess_pat": "hess_pat.mtx",
    "hess_pat_small": "hess_pat_small.mtx",
    "jac_pat": "jac_pat.mtx",
}


def graph_to_edge_index(graph) -> torch.Tensor:
    """
    Convert a NetworkX graph into a PyTorch Geometric edge_index tensor.

    For an undirected graph, both directions are added:
    u -> v and v -> u.

    Returns
    -------
    torch.Tensor
        edge_index tensor with shape [2, num_directed_edges].
    """

    directed_edges = []

    for u, v in graph.edges():
        directed_edges.append((int(u), int(v)))
        directed_edges.append((int(v), int(u)))

    if not directed_edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    return edge_index


def load_split_file(split_path: str | Path) -> dict[str, str]:
    """
    Load the graph-level train/validation/test split file.

    Expected CSV format:
    graph_id,split
    ash85,train
    ...
    """

    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    graph_to_split = {}

    with split_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {"graph_id", "split"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Split file must contain columns {required_columns}, "
                f"got {reader.fieldnames}."
            )

        for row in reader:
            graph_id = row["graph_id"]
            split = row["split"]

            graph_to_split[graph_id] = split

    return graph_to_split


def build_pyg_data_from_graph_id(
    graph_id: str,
    matrix_dir: str | Path = "data/raw/matrices",
    split: str | None = None,
) -> Data:
    """
    Build one PyTorch Geometric Data object from a graph ID.
    """

    if graph_id not in GRAPH_ID_TO_MATRIX:
        raise ValueError(f"Unknown graph_id: {graph_id}")

    matrix_dir = Path(matrix_dir)
    matrix_path = matrix_dir / GRAPH_ID_TO_MATRIX[graph_id]

    graph = load_graph_from_mtx(matrix_path)
    features = extract_node_features(graph)

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = graph_to_edge_index(graph)

    data = Data(
        x=x,
        edge_index=edge_index,
        graph_id=graph_id,
        split=split,
        num_nodes=graph.number_of_nodes(),
    )

    return data


def build_dataset_from_split(
    split_path: str | Path = "data/processed/initial_graph_coloring_dataset/splits/initial_graph_split.csv",
    matrix_dir: str | Path = "data/raw/matrices",
) -> list[Data]:
    """
    Build PyG Data objects for all graphs listed in the split file.
    """

    graph_to_split = load_split_file(split_path)

    dataset = []

    for graph_id, split in graph_to_split.items():
        data = build_pyg_data_from_graph_id(
            graph_id=graph_id,
            matrix_dir=matrix_dir,
            split=split,
        )
        dataset.append(data)

    return dataset


def get_dataset_by_split(dataset: list[Data], split_name: str) -> list[Data]:
    """
    Filter PyG Data objects by split name.
    """

    return [data for data in dataset if data.split == split_name]


def validate_pyg_data(data: Data) -> None:
    """
    Validate basic consistency properties of one PyTorch Geometric Data object.
    """

    if data.x.ndim != 2:
        raise ValueError(f"{data.graph_id}: x must be 2-dimensional, got {data.x.shape}.")

    if data.x.shape[0] != data.num_nodes:
        raise ValueError(
            f"{data.graph_id}: x has {data.x.shape[0]} rows, "
            f"but num_nodes is {data.num_nodes}."
        )

    if data.edge_index.ndim != 2:
        raise ValueError(
            f"{data.graph_id}: edge_index must be 2-dimensional, "
            f"got {data.edge_index.shape}."
        )

    if data.edge_index.shape[0] != 2:
        raise ValueError(
            f"{data.graph_id}: edge_index must have shape [2, num_edges], "
            f"got {data.edge_index.shape}."
        )

    if data.edge_index.numel() > 0:
        min_index = int(data.edge_index.min())
        max_index = int(data.edge_index.max())

        if min_index < 0:
            raise ValueError(f"{data.graph_id}: edge_index contains negative node indices.")

        if max_index >= data.num_nodes:
            raise ValueError(
                f"{data.graph_id}: edge_index contains node index {max_index}, "
                f"but num_nodes is {data.num_nodes}."
            )

    if not hasattr(data, "graph_id") or data.graph_id is None:
        raise ValueError("PyG Data object is missing graph_id.")

    if not hasattr(data, "split") or data.split is None:
        raise ValueError(f"{data.graph_id}: PyG Data object is missing split.")
    

def save_pyg_dataset(
    dataset: list[Data],
    output_dir: str | Path = "data/processed/initial_graph_coloring_dataset/pyg_data",
) -> None:
    """
    Save each PyTorch Geometric Data object as a .pt file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for data in dataset:
        validate_pyg_data(data)

        output_path = output_dir / f"{data.graph_id}.pt"
        torch.save(data, output_path)

        print(f"Saved {data.graph_id} to {output_path}")


def main() -> None:
    dataset = build_dataset_from_split()

    train_data = get_dataset_by_split(dataset, "train")
    validation_data = get_dataset_by_split(dataset, "validation")
    test_data = get_dataset_by_split(dataset, "test")

    print("PyTorch Geometric split-aware dataset check")
    print("------------------------------------------")

    print(f"Total graphs: {len(dataset)}")
    print(f"Train graphs: {len(train_data)}")
    print(f"Validation graphs: {len(validation_data)}")
    print(f"Test graphs: {len(test_data)}")
    print()

    for data in dataset:
        validate_pyg_data(data)

        print(f"Graph ID: {data.graph_id}")
        print(f"  split: {data.split}")
        print(f"  x shape: {tuple(data.x.shape)}")
        print(f"  edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  num_nodes: {data.num_nodes}")
        print("  status: OK")
        print()
    
    print("Saving PyG Data objects")
    print("-----------------------")
    save_pyg_dataset(dataset)
    print()

    
    print("All split-aware PyG Data checks passed.")


if __name__ == "__main__":
    main()