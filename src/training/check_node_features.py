"""
Check node feature extraction on all graphs in the initial graph-coloring dataset.

This script verifies that:
1. each matrix can be loaded as a graph,
2. node features can be extracted,
3. the feature matrix has the expected shape,
4. no NaN or infinite values occur,
5. all graphs have the same feature dimension.
"""

from __future__ import annotations

from pathlib import Path

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.node_features import (
    extract_node_features,
    get_feature_names,
    validate_feature_matrix,
)


MATRIX_FILES = [
    "ash85.mtx",
    "can_24.mtx",
    "hess_pat.mtx",
    "hess_pat_small.mtx",
    "jac_pat.mtx",
]


def main() -> None:
    matrix_dir = Path("data/raw/matrices")
    feature_names = get_feature_names()
    expected_num_features = len(feature_names)

    print("Node feature consistency check")
    print("------------------------------")
    print(f"Feature names: {feature_names}")
    print(f"Expected number of features: {expected_num_features}")
    print()

    for matrix_file in MATRIX_FILES:
        matrix_path = matrix_dir / matrix_file

        graph = load_graph_from_mtx(matrix_path)
        features = extract_node_features(graph)

        validate_feature_matrix(
            features=features,
            expected_num_nodes=graph.number_of_nodes(),
        )

        if features.shape[1] != expected_num_features:
            raise ValueError(
                f"{matrix_file}: expected {expected_num_features} features, "
                f"got {features.shape[1]}."
            )

        print(f"Graph from matrix: {matrix_file}")
        print(f"  nodes: {graph.number_of_nodes()}")
        print(f"  edges: {graph.number_of_edges()}")
        print(f"  feature shape: {features.shape}")
        print("  status: OK")
        print()

    print("All node feature checks passed.")


if __name__ == "__main__":
    main()