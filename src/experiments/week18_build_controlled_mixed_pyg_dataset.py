from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.build_pyg_dataset import (
    graph_to_edge_index,
    validate_pyg_data,
)
from src.training.node_features_week17_symmetry_breaking import (
    extract_node_features,
    validate_feature_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week18_controlled_mixed_joins"
)

TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week18_controlled_mixed_exact_ordering_targets.csv"
)

TARGET_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_exact_target_summary.csv"
)

COLPACK_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_colpack_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week18_controlled_mixed_exact"
)

PYG_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_mixed_exact_pyg_summary.csv"
)

EXPECTED_GRAPH_COUNT = 105
EXPECTED_FEATURE_COUNT = 25


def load_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGET_CSV)

    required_columns = {
        "graph_id",
        "node_id",
        "order_position",
        "target_score",
        "known_color",
        "selected_ordering",
        "selected_num_colors",
        "known_chromatic_number",
        "split",
        "graph_family",
        "num_components_joined",
        "component_cycle_sizes",
        "score_convention",
    }

    missing = required_columns - set(targets.columns)

    if missing:
        raise ValueError(
            f"Target CSV missing columns: {sorted(missing)}"
        )

    return targets


def load_target_summary() -> pd.DataFrame:
    summary = pd.read_csv(TARGET_SUMMARY_CSV)

    required_columns = {
        "graph_id",
        "split",
        "graph_family",
        "num_components_joined",
        "component_cycle_sizes",
        "num_unique_component_sizes",
        "num_vertices",
        "num_edges",
        "known_chromatic_number",
        "best_colpack5_colors",
        "verified_gap",
        "selected_ordering",
        "selected_num_colors",
        "target_coloring_valid",
        "score_convention",
    }

    missing = required_columns - set(summary.columns)

    if missing:
        raise ValueError(
            f"Target summary missing columns: {sorted(missing)}"
        )

    if len(summary) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} target-summary rows, "
            f"found {len(summary)}."
        )

    if summary["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in target summary."
        )

    if not bool(summary["target_coloring_valid"].all()):
        raise ValueError(
            "At least one exact target coloring is invalid."
        )

    return summary


def load_colpack_summary() -> pd.DataFrame:
    summary = pd.read_csv(COLPACK_SUMMARY_CSV)

    required_columns = {
        "graph_id",
        "best_colpack5_colors",
        "best_colpack5_gap_from_known",
        "colpack5_stuck_above_known",
    }

    missing = required_columns - set(summary.columns)

    if missing:
        raise ValueError(
            f"ColPack summary missing columns: {sorted(missing)}"
        )

    if len(summary) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} ColPack rows, "
            f"found {len(summary)}."
        )

    if summary["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in ColPack summary."
        )

    return summary


def resolve_matrix_path(
    graph_id: str,
) -> Path:
    matrix_path = (
        MATRIX_DIR
        / f"{graph_id}.mtx"
    )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Matrix file not found for {graph_id}: "
            f"{matrix_path}"
        )

    return matrix_path


def validate_input_graph_ids(
    targets: pd.DataFrame,
    target_summary: pd.DataFrame,
    colpack_summary: pd.DataFrame,
) -> None:
    target_ids = set(
        targets["graph_id"].unique()
    )

    target_summary_ids = set(
        target_summary["graph_id"]
    )

    colpack_ids = set(
        colpack_summary["graph_id"]
    )

    if target_ids != target_summary_ids:
        raise ValueError(
            "Node-target and target-summary graph IDs differ."
        )

    if target_summary_ids != colpack_ids:
        raise ValueError(
            "Target-summary and ColPack-summary graph IDs differ."
        )

    if len(target_summary_ids) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} unique graphs, "
            f"found {len(target_summary_ids)}."
        )


def build_data_for_graph(
    graph_id: str,
    targets: pd.DataFrame,
    target_summary: pd.DataFrame,
    colpack_summary: pd.DataFrame,
) -> Data:
    matrix_path = resolve_matrix_path(
        graph_id
    )

    graph = load_graph_from_mtx(
        matrix_path
    )

    features = extract_node_features(
        graph
    )

    validate_feature_matrix(
        features,
        graph.number_of_nodes(),
    )

    graph_targets = (
        targets[
            targets["graph_id"]
            == graph_id
        ]
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    if len(graph_targets) != graph.number_of_nodes():
        raise ValueError(
            f"{graph_id}: target rows "
            f"({len(graph_targets)}) do not match "
            f"graph nodes ({graph.number_of_nodes()})."
        )

    expected_node_ids = list(
        range(
            graph.number_of_nodes()
        )
    )

    actual_node_ids = (
        graph_targets["node_id"]
        .astype(int)
        .tolist()
    )

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{graph_id}: target node IDs do not match "
            f"0..{graph.number_of_nodes() - 1}."
        )

    target_rows = target_summary[
        target_summary["graph_id"]
        == graph_id
    ]

    if len(target_rows) != 1:
        raise ValueError(
            f"{graph_id}: expected one target-summary row."
        )

    colpack_rows = colpack_summary[
        colpack_summary["graph_id"]
        == graph_id
    ]

    if len(colpack_rows) != 1:
        raise ValueError(
            f"{graph_id}: expected one ColPack-summary row."
        )

    target_row = target_rows.iloc[0]
    colpack_row = colpack_rows.iloc[0]

    expected_nodes = int(
        target_row["num_vertices"]
    )

    expected_edges = int(
        target_row["num_edges"]
    )

    if graph.number_of_nodes() != expected_nodes:
        raise ValueError(
            f"{graph_id}: graph has "
            f"{graph.number_of_nodes()} nodes, "
            f"expected {expected_nodes}."
        )

    if graph.number_of_edges() != expected_edges:
        raise ValueError(
            f"{graph_id}: graph has "
            f"{graph.number_of_edges()} edges, "
            f"expected {expected_edges}."
        )

    selected_ordering_values = (
        graph_targets[
            "selected_ordering"
        ].unique()
    )

    selected_color_values = (
        graph_targets[
            "selected_num_colors"
        ].unique()
    )

    chromatic_values = (
        graph_targets[
            "known_chromatic_number"
        ].unique()
    )

    score_conventions = (
        graph_targets[
            "score_convention"
        ].unique()
    )

    if len(selected_ordering_values) != 1:
        raise ValueError(
            f"{graph_id}: multiple selected orderings found."
        )

    if len(selected_color_values) != 1:
        raise ValueError(
            f"{graph_id}: multiple selected color counts found."
        )

    if len(chromatic_values) != 1:
        raise ValueError(
            f"{graph_id}: multiple chromatic numbers found."
        )

    if len(score_conventions) != 1:
        raise ValueError(
            f"{graph_id}: multiple score conventions found."
        )

    num_components = int(
        target_row[
            "num_components_joined"
        ]
    )

    known_chromatic_number = int(
        chromatic_values[0]
    )

    selected_num_colors = int(
        selected_color_values[0]
    )

    best_colpack5_colors = int(
        colpack_row[
            "best_colpack5_colors"
        ]
    )

    best_colpack5_gap = int(
        colpack_row[
            "best_colpack5_gap_from_known"
        ]
    )

    verified_gap = int(
        target_row[
            "verified_gap"
        ]
    )

    if selected_num_colors != known_chromatic_number:
        raise ValueError(
            f"{graph_id}: exact ordering uses "
            f"{selected_num_colors} colors, but target is "
            f"{known_chromatic_number}."
        )

    if known_chromatic_number != 4 * num_components:
        raise ValueError(
            f"{graph_id}: expected target "
            f"{4 * num_components}, found "
            f"{known_chromatic_number}."
        )

    if best_colpack5_gap != verified_gap:
        raise ValueError(
            f"{graph_id}: ColPack gap {best_colpack5_gap} "
            f"does not match verified gap {verified_gap}."
        )

    if best_colpack5_gap != num_components:
        raise ValueError(
            f"{graph_id}: expected gap {num_components}, "
            f"found {best_colpack5_gap}."
        )

    x = torch.tensor(
        features,
        dtype=torch.float32,
    )

    edge_index = graph_to_edge_index(
        graph
    )

    y = torch.tensor(
        graph_targets[
            "target_score"
        ].values,
        dtype=torch.float32,
    )

    known_colors = torch.tensor(
        graph_targets[
            "known_color"
        ].values,
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        known_colors=known_colors,
        graph_id=graph_id,
        split="train",
        group=(
            "controlled_mixed_"
            "cycle_square_joins"
        ),
        graph_family=str(
            target_row[
                "graph_family"
            ]
        ),
        num_components_joined=(
            num_components
        ),
        component_cycle_sizes=str(
            target_row[
                "component_cycle_sizes"
            ]
        ),
        num_unique_component_sizes=int(
            target_row[
                "num_unique_component_sizes"
            ]
        ),
        gap_level=(
            best_colpack5_gap
        ),
        selected_ordering=str(
            selected_ordering_values[0]
        ),
        selected_num_colors=(
            selected_num_colors
        ),
        known_chromatic_number=(
            known_chromatic_number
        ),
        best_colpack5_colors=(
            best_colpack5_colors
        ),
        best_colpack5_gap_from_known=(
            best_colpack5_gap
        ),
        colpack5_stuck_above_known=bool(
            colpack_row[
                "colpack5_stuck_above_known"
            ]
        ),
        score_convention=str(
            score_conventions[0]
        ),
        num_nodes=(
            graph.number_of_nodes()
        ),
    )

    validate_pyg_data(
        data
    )

    if data.y.shape[0] != data.num_nodes:
        raise ValueError(
            f"{graph_id}: y has {data.y.shape[0]} rows, "
            f"but graph has {data.num_nodes} nodes."
        )

    if data.known_colors.shape[0] != data.num_nodes:
        raise ValueError(
            f"{graph_id}: known_colors has "
            f"{data.known_colors.shape[0]} rows, "
            f"but graph has {data.num_nodes} nodes."
        )

    if data.x.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{graph_id}: expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"found {data.x.shape[1]}."
        )

    return data


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = load_targets()
    target_summary = (
        load_target_summary()
    )
    colpack_summary = (
        load_colpack_summary()
    )

    validate_input_graph_ids(
        targets=targets,
        target_summary=target_summary,
        colpack_summary=colpack_summary,
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    sorted_graph_ids = sorted(
        target_summary[
            "graph_id"
        ].tolist()
    )

    for graph_index, graph_id in enumerate(
        sorted_graph_ids,
        start=1,
    ):
        data = build_data_for_graph(
            graph_id=graph_id,
            targets=targets,
            target_summary=target_summary,
            colpack_summary=colpack_summary,
        )

        output_path = (
            OUTPUT_DIR
            / f"{graph_id}.pt"
        )

        torch.save(
            data,
            output_path,
        )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "split": data.split,
                "graph_family": (
                    data.graph_family
                ),
                "num_components_joined": int(
                    data.num_components_joined
                ),
                "component_cycle_sizes": (
                    data.component_cycle_sizes
                ),
                "num_unique_component_sizes": int(
                    data.num_unique_component_sizes
                ),
                "gap_level": int(
                    data.gap_level
                ),
                "num_nodes": int(
                    data.num_nodes
                ),
                "num_directed_edges": int(
                    data.edge_index.shape[1]
                ),
                "num_features": int(
                    data.x.shape[1]
                ),
                "target_colors": int(
                    data.selected_num_colors
                ),
                "known_chromatic_number": int(
                    data.known_chromatic_number
                ),
                "best_colpack5_colors": int(
                    data.best_colpack5_colors
                ),
                "best_colpack5_gap_from_known": int(
                    data.best_colpack5_gap_from_known
                ),
                "selected_ordering": (
                    data.selected_ordering
                ),
                "score_convention": (
                    data.score_convention
                ),
                "path": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

        print(
            f"[{graph_index}/{EXPECTED_GRAPH_COUNT}] "
            f"Saved {graph_id}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "num_components_joined",
            "num_nodes",
            "graph_id",
        ]
    ).reset_index(drop=True)

    if len(summary_df) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} PyG graphs, "
            f"created {len(summary_df)}."
        )

    saved_paths = sorted(
        OUTPUT_DIR.glob(
            "week18_controlled_mixed_*.pt"
        )
    )

    if len(saved_paths) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} saved .pt files, "
            f"found {len(saved_paths)}."
        )

    if not (
        summary_df[
            "num_features"
        ]
        == EXPECTED_FEATURE_COUNT
    ).all():
        raise ValueError(
            "At least one graph has an incorrect "
            "feature count."
        )

    PYG_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        PYG_SUMMARY_CSV,
        index=False,
    )

    print()
    print(
        "Week 18 controlled mixed PyG "
        "dataset built successfully."
    )
    print(
        "---------------------------------------"
    )
    print(
        f"Total graphs: {len(summary_df)}"
    )
    print(
        f"Total nodes: "
        f"{summary_df['num_nodes'].sum()}"
    )
    print(
        f"Features per node: "
        f"{EXPECTED_FEATURE_COUNT}"
    )
    print()

    print(
        "Graphs by gap level:"
    )

    print(
        summary_df[
            "gap_level"
        ]
        .value_counts()
        .sort_index()
        .rename_axis("gap_level")
        .rename("num_graphs")
        .to_string()
    )

    print()
    print(
        "Node totals by gap level:"
    )

    print(
        summary_df
        .groupby("gap_level")[
            "num_nodes"
        ]
        .sum()
        .to_string()
    )

    print()
    print(
        f"Saved PyG graphs to: "
        f"{OUTPUT_DIR}"
    )
    print(
        f"Saved PyG summary to: "
        f"{PYG_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()