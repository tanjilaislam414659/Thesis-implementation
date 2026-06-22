"""
Build expanded PyTorch Geometric Data objects for Week 15.

This script uses:
1. the expanded 15-graph split from Week 15,
2. the Week 14 SMALLEST_LAST ordering targets,
3. the corrected column-intersection graph for jac_pat.

Each PyG Data object contains:
- x: node feature matrix
- edge_index: graph connectivity
- y: node-level ordering target score
- graph_id
- split
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.build_pyg_dataset import graph_to_edge_index, validate_pyg_data
from src.training.node_features import extract_node_features


MATRIX_DIR = Path("data/raw/matrices")

JAC_PAT_GRAPH_PATH = Path(
    "data/processed/initial_graph_coloring_dataset/colpack_graph_inputs/"
    "jac_pat_column_intersection_graph.mtx"
)

SPLIT_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/splits/"
    "expanded_graph_split_week15.csv"
)

TARGET_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/ordering_targets/"
    "smallest_last_ordering_targets_week14_expanded.csv"
)

OUTPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "pyg_data_week15_expanded"
)


GRAPH_ID_TO_MATRIX = {
    "ash85": "ash85.mtx",
    "can_24": "can_24.mtx",
    "hess_pat": "hess_pat.mtx",
    "hess_pat_small": "hess_pat_small.mtx",
    "jac_pat": JAC_PAT_GRAPH_PATH,
    "bcsstk01": "bcsstk01.mtx",
    "bcsstk03": "bcsstk03.mtx",
    "bcsstk04": "bcsstk04.mtx",
    "bcsstk05": "bcsstk05.mtx",
    "bcsstk06": "bcsstk06.mtx",
    "dwt_234": "dwt_234.mtx",
    "dwt_361": "dwt_361.mtx",
    "dwt_419": "dwt_419.mtx",
    "west0479": "west0479.mtx",
    "sherman1": "sherman1.mtx",
}


def resolve_matrix_path(graph_id: str) -> Path:
    matrix_entry = GRAPH_ID_TO_MATRIX[graph_id]

    if isinstance(matrix_entry, Path):
        return matrix_entry

    return MATRIX_DIR / matrix_entry


def load_split() -> dict[str, str]:
    split_df = pd.read_csv(SPLIT_CSV)
    return dict(zip(split_df["graph_id"], split_df["split"]))


def load_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGET_CSV)

    required_columns = {
        "graph_id",
        "node_id",
        "order_position",
        "target_score",
    }

    if not required_columns.issubset(targets.columns):
        raise ValueError(
            f"Target file must contain columns {required_columns}, "
            f"got {list(targets.columns)}."
        )

    return targets


def build_data_for_graph(
    graph_id: str,
    split: str,
    targets: pd.DataFrame,
) -> Data:
    matrix_path = resolve_matrix_path(graph_id)

    graph = load_graph_from_mtx(matrix_path)
    features = extract_node_features(graph)

    graph_targets = (
        targets[targets["graph_id"] == graph_id]
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    if len(graph_targets) != graph.number_of_nodes():
        raise ValueError(
            f"{graph_id}: target rows ({len(graph_targets)}) do not match "
            f"number of graph nodes ({graph.number_of_nodes()})."
        )

    expected_node_ids = list(range(graph.number_of_nodes()))
    actual_node_ids = graph_targets["node_id"].tolist()

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{graph_id}: node IDs in target file do not match "
            f"expected node IDs 0..{graph.number_of_nodes() - 1}."
        )

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = graph_to_edge_index(graph)
    y = torch.tensor(graph_targets["target_score"].values, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        graph_id=graph_id,
        split=split,
        num_nodes=graph.number_of_nodes(),
    )

    validate_pyg_data(data)

    if data.y.shape[0] != data.num_nodes:
        raise ValueError(
            f"{graph_id}: y has {data.y.shape[0]} rows, "
            f"but num_nodes is {data.num_nodes}."
        )

    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph_to_split = load_split()
    targets = load_targets()

    dataset = []

    for graph_id, split in graph_to_split.items():
        data = build_data_for_graph(
            graph_id=graph_id,
            split=split,
            targets=targets,
        )

        output_path = OUTPUT_DIR / f"{graph_id}.pt"
        torch.save(data, output_path)

        dataset.append(data)

        print(f"Saved {graph_id} to {output_path}")
        print(f"  split: {split}")
        print(f"  x shape: {tuple(data.x.shape)}")
        print(f"  edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  y shape: {tuple(data.y.shape)}")
        print(f"  num_nodes: {data.num_nodes}")
        print()

    print("Expanded Week 15 PyG dataset built successfully.")
    print(f"Total graphs: {len(dataset)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()