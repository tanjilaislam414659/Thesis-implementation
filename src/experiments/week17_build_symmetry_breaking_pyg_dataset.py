from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.build_pyg_dataset import graph_to_edge_index, validate_pyg_data
from src.training.node_features_week17_symmetry_breaking import (
    extract_node_features,
    validate_feature_matrix,
)


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

MATRIX_DIR = PROJECT_ROOT / "data" / "raw" / "matrices"

JAC_PAT_GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_graph_inputs"
    / "jac_pat_column_intersection_graph.mtx"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_best_available_of_5_split.csv"
)

TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week17_best_available_of_5_ordering_targets.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week17_best_available_of_5_symmetry_breaking_features"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_best_available_of_5_symmetry_breaking_pyg_summary.csv"
)


GRAPH_ID_TO_MATRIX = {
    # Original sparse matrices
    "ash85": MATRIX_DIR / "ash85.mtx",
    "can_24": MATRIX_DIR / "can_24.mtx",
    "hess_pat": MATRIX_DIR / "hess_pat.mtx",
    "hess_pat_small": MATRIX_DIR / "hess_pat_small.mtx",
    "jac_pat": JAC_PAT_GRAPH_PATH,
    "bcsstk01": MATRIX_DIR / "bcsstk01.mtx",
    "bcsstk03": MATRIX_DIR / "bcsstk03.mtx",
    "bcsstk04": MATRIX_DIR / "bcsstk04.mtx",
    "bcsstk05": MATRIX_DIR / "bcsstk05.mtx",
    "bcsstk06": MATRIX_DIR / "bcsstk06.mtx",
    "dwt_234": MATRIX_DIR / "dwt_234.mtx",
    "dwt_361": MATRIX_DIR / "dwt_361.mtx",
    "dwt_419": MATRIX_DIR / "dwt_419.mtx",
    "west0479": MATRIX_DIR / "west0479.mtx",
    "sherman1": MATRIX_DIR / "sherman1.mtx",

    # Bickle hard-case graphs
    "week17_bickle_g10": MATRIX_DIR / "week17_bickle_hard_cases" / "week17_bickle_g10.mtx",
    "week17_cycle_square_c8": MATRIX_DIR / "week17_bickle_hard_cases" / "week17_cycle_square_c8.mtx",
    "week17_cycle_square_c11": MATRIX_DIR / "week17_bickle_hard_cases" / "week17_cycle_square_c11.mtx",
    "week17_cycle_square_c14": MATRIX_DIR / "week17_bickle_hard_cases" / "week17_cycle_square_c14.mtx",
    "week17_cycle_square_c17": MATRIX_DIR / "week17_bickle_hard_cases" / "week17_cycle_square_c17.mtx",
    "week17_cycle_square_c20": MATRIX_DIR / "week17_bickle_hard_cases" / "week17_cycle_square_c20.mtx",

    # Structured matrices
    "week17_nos1": MATRIX_DIR / "nos1.mtx",
    "week17_gr_30_30": MATRIX_DIR / "gr_30_30.mtx",
    "week17_bwm200": MATRIX_DIR / "bwm200.mtx",
    "week17_bcsstk08": MATRIX_DIR / "bcsstk08.mtx",
    "week17_lshp_265": MATRIX_DIR / "lshp_265.mtx",
    "week17_arrowhead_100": MATRIX_DIR / "week17_arrowhead_100.mtx",
}


def load_split() -> pd.DataFrame:
    split_df = pd.read_csv(SPLIT_CSV)

    required_columns = {"graph_id", "split", "group", "reason"}
    missing = required_columns - set(split_df.columns)

    if missing:
        raise ValueError(f"Split CSV missing columns: {missing}")

    return split_df


def load_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGET_CSV)

    required_columns = {
        "graph_id",
        "node_id",
        "order_position",
        "target_score",
        "selected_ordering",
        "selected_num_colors",
    }

    missing = required_columns - set(targets.columns)

    if missing:
        raise ValueError(f"Target CSV missing columns: {missing}")

    return targets


def resolve_matrix_path(graph_id: str) -> Path:
    if graph_id not in GRAPH_ID_TO_MATRIX:
        raise KeyError(f"No matrix mapping found for graph_id: {graph_id}")

    matrix_path = GRAPH_ID_TO_MATRIX[graph_id]

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found for {graph_id}: {matrix_path}")

    return matrix_path


def build_data_for_graph(
    graph_id: str,
    split: str,
    group: str,
    reason: str,
    targets: pd.DataFrame,
) -> Data:
    matrix_path = resolve_matrix_path(graph_id)
    graph = load_graph_from_mtx(matrix_path)

    features = extract_node_features(graph)
    validate_feature_matrix(features, graph.number_of_nodes())

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
    actual_node_ids = graph_targets["node_id"].astype(int).tolist()

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{graph_id}: node IDs in target file do not match "
            f"expected node IDs 0..{graph.number_of_nodes() - 1}."
        )

    selected_ordering_values = graph_targets["selected_ordering"].unique()
    selected_color_values = graph_targets["selected_num_colors"].unique()

    if len(selected_ordering_values) != 1:
        raise ValueError(f"{graph_id}: multiple selected_ordering values found.")

    if len(selected_color_values) != 1:
        raise ValueError(f"{graph_id}: multiple selected_num_colors values found.")

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = graph_to_edge_index(graph)
    y = torch.tensor(graph_targets["target_score"].values, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        graph_id=graph_id,
        split=split,
        group=group,
        reason=reason,
        selected_ordering=str(selected_ordering_values[0]),
        selected_num_colors=int(selected_color_values[0]),
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

    split_df = load_split()
    targets = load_targets()

    dataset = []
    summary_rows = []

    for row in split_df.itertuples(index=False):
        data = build_data_for_graph(
            graph_id=row.graph_id,
            split=row.split,
            group=row.group,
            reason=row.reason,
            targets=targets,
        )

        output_path = OUTPUT_DIR / f"{row.graph_id}.pt"
        torch.save(data, output_path)

        dataset.append(data)

        summary_rows.append(
            {
                "graph_id": row.graph_id,
                "split": row.split,
                "group": row.group,
                "num_nodes": int(data.num_nodes),
                "num_directed_edges": int(data.edge_index.shape[1]),
                "num_features": int(data.x.shape[1]),
                "selected_ordering": data.selected_ordering,
                "selected_num_colors": data.selected_num_colors,
                "path": str(output_path),
            }
        )

        print(f"Saved {row.graph_id} to {output_path}")
        print(f"  split: {row.split}")
        print(f"  group: {row.group}")
        print(f"  x shape: {tuple(data.x.shape)}")
        print(f"  edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  y shape: {tuple(data.y.shape)}")
        print(f"  selected_ordering: {data.selected_ordering}")
        print(f"  selected_num_colors: {data.selected_num_colors}")
        print()

    summary_df = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("Week 17 symmetry-breaking PyG dataset built successfully.")
    print(f"Total graphs: {len(dataset)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Saved summary to: {SUMMARY_CSV}")
    print()

    print("Split counts:")
    print(summary_df["split"].value_counts())
    print()

    print("Group counts:")
    print(summary_df["group"].value_counts())
    print()

    print("Feature count check:")
    print(summary_df["num_features"].value_counts())


if __name__ == "__main__":
    main()