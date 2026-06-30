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

MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week17_bickle_exact_family"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_bickle_exact_split.csv"
)

TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week17_bickle_exact_optimal_ordering_targets.csv"
)

EXACT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_exact_family_summary.csv"
)

COLPACK_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_exact_family_colpack_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week17_bickle_exact_symmetry_breaking"
)

PYG_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week17_bickle_exact_symmetry_breaking_pyg_summary.csv"
)


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
        "known_chromatic_number",
    }

    missing = required_columns - set(targets.columns)

    if missing:
        raise ValueError(f"Target CSV missing columns: {missing}")

    return targets


def load_exact_summary() -> pd.DataFrame:
    summary = pd.read_csv(EXACT_SUMMARY_CSV)

    required_columns = {
        "graph_id",
        "cycle_size",
        "known_chromatic_number",
        "exact_greedy_colors",
    }

    missing = required_columns - set(summary.columns)

    if missing:
        raise ValueError(f"Exact summary missing columns: {missing}")

    return summary


def load_colpack_summary() -> pd.DataFrame:
    summary = pd.read_csv(COLPACK_SUMMARY_CSV)

    required_columns = {
        "graph_id",
        "best_colpack5_colors",
        "best_colpack5_gap_from_chromatic",
        "colpack5_stuck_above_optimum",
    }

    missing = required_columns - set(summary.columns)

    if missing:
        raise ValueError(f"ColPack summary missing columns: {missing}")

    return summary


def resolve_matrix_path(graph_id: str) -> Path:
    matrix_path = MATRIX_DIR / f"{graph_id}.mtx"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found for {graph_id}: {matrix_path}")

    return matrix_path


def build_data_for_graph(
    graph_id: str,
    split: str,
    group: str,
    reason: str,
    targets: pd.DataFrame,
    exact_summary: pd.DataFrame,
    colpack_summary: pd.DataFrame,
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

    exact_rows = exact_summary[exact_summary["graph_id"] == graph_id]

    if len(exact_rows) != 1:
        raise ValueError(f"{graph_id}: expected exactly one exact summary row.")

    colpack_rows = colpack_summary[colpack_summary["graph_id"] == graph_id]

    if len(colpack_rows) != 1:
        raise ValueError(f"{graph_id}: expected exactly one ColPack summary row.")

    exact_row = exact_rows.iloc[0]
    colpack_row = colpack_rows.iloc[0]

    selected_ordering_values = graph_targets["selected_ordering"].unique()
    selected_color_values = graph_targets["selected_num_colors"].unique()
    chromatic_values = graph_targets["known_chromatic_number"].unique()

    if len(selected_ordering_values) != 1:
        raise ValueError(f"{graph_id}: multiple selected_ordering values found.")

    if len(selected_color_values) != 1:
        raise ValueError(f"{graph_id}: multiple selected_num_colors values found.")

    if len(chromatic_values) != 1:
        raise ValueError(f"{graph_id}: multiple known_chromatic_number values found.")

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
        known_chromatic_number=int(chromatic_values[0]),
        cycle_size=int(exact_row["cycle_size"]),
        exact_greedy_colors=int(exact_row["exact_greedy_colors"]),
        best_colpack5_colors=int(colpack_row["best_colpack5_colors"]),
        best_colpack5_gap_from_chromatic=int(
            colpack_row["best_colpack5_gap_from_chromatic"]
        ),
        colpack5_stuck_above_optimum=bool(
            colpack_row["colpack5_stuck_above_optimum"]
        ),
        num_nodes=graph.number_of_nodes(),
    )

    validate_pyg_data(data)

    if data.y.shape[0] != data.num_nodes:
        raise ValueError(
            f"{graph_id}: y has {data.y.shape[0]} rows, "
            f"but num_nodes is {data.num_nodes}."
        )

    if data.x.shape[1] != 25:
        raise ValueError(
            f"{graph_id}: expected 25 symmetry-breaking features, "
            f"got {data.x.shape[1]}."
        )

    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_df = load_split()
    targets = load_targets()
    exact_summary = load_exact_summary()
    colpack_summary = load_colpack_summary()

    dataset = []
    summary_rows = []

    for row in split_df.itertuples(index=False):
        data = build_data_for_graph(
            graph_id=row.graph_id,
            split=row.split,
            group=row.group,
            reason=row.reason,
            targets=targets,
            exact_summary=exact_summary,
            colpack_summary=colpack_summary,
        )

        output_path = OUTPUT_DIR / f"{row.graph_id}.pt"
        torch.save(data, output_path)

        dataset.append(data)

        summary_rows.append(
            {
                "graph_id": row.graph_id,
                "split": row.split,
                "group": row.group,
                "cycle_size": int(data.cycle_size),
                "num_nodes": int(data.num_nodes),
                "num_directed_edges": int(data.edge_index.shape[1]),
                "num_features": int(data.x.shape[1]),
                "target_colors": int(data.selected_num_colors),
                "known_chromatic_number": int(data.known_chromatic_number),
                "best_colpack5_colors": int(data.best_colpack5_colors),
                "best_colpack5_gap_from_chromatic": int(
                    data.best_colpack5_gap_from_chromatic
                ),
                "colpack5_stuck_above_optimum": bool(
                    data.colpack5_stuck_above_optimum
                ),
                "selected_ordering": data.selected_ordering,
                "path": str(output_path),
            }
        )

        print(f"Saved {row.graph_id} to {output_path}")
        print(f"  split: {row.split}")
        print(f"  cycle size: {data.cycle_size}")
        print(f"  x shape: {tuple(data.x.shape)}")
        print(f"  edge_index shape: {tuple(data.edge_index.shape)}")
        print(f"  y shape: {tuple(data.y.shape)}")
        print(f"  exact target colors: {data.selected_num_colors}")
        print(f"  best ColPack-5 colors: {data.best_colpack5_colors}")
        print(f"  ColPack-5 stuck above optimum: {data.colpack5_stuck_above_optimum}")
        print()

    summary_df = pd.DataFrame(summary_rows)
    PYG_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PYG_SUMMARY_CSV, index=False)

    print("Week 17 exact-optimal Bickle PyG dataset built successfully.")
    print(f"Total graphs: {len(dataset)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Saved summary to: {PYG_SUMMARY_CSV}")
    print()

    print("Split counts:")
    print(summary_df["split"].value_counts())
    print()

    print("Feature count check:")
    print(summary_df["num_features"].value_counts())
    print()

    print("ColPack-5 stuck above optimum counts:")
    print(summary_df["colpack5_stuck_above_optimum"].value_counts())


if __name__ == "__main__":
    main()