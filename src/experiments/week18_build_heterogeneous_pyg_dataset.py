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

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_heterogeneous_balanced_split.csv"
)

TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week18_balanced_best_of_5_ordering_targets.csv"
)

TARGET_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_balanced_best_of_5_target_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week18_heterogeneous_generalization"
)

PYG_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_pyg_summary.csv"
)


EXPECTED_SPLIT_COUNTS = {
    "train": 48,
    "validation": 8,
    "test": 12,
}

EXPECTED_TEACHERS = [
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def load_split() -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Week 18 split CSV not found: {SPLIT_CSV}"
        )

    split_df = pd.read_csv(SPLIT_CSV)

    required_columns = {
        "graph_id",
        "split",
        "split_group_id",
        "split_grouping_mode",
        "family",
        "num_vertices",
        "num_edges",
        "density",
        "matrix_path",
        "selected_teacher_ordering",
        "best_colpack5_colors",
        "worst_colpack5_colors",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
        "teacher_selection_reason",
    }

    missing_columns = required_columns - set(split_df.columns)

    if missing_columns:
        raise ValueError(
            f"Split CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if split_df["graph_id"].duplicated().any():
        duplicates = split_df.loc[
            split_df["graph_id"].duplicated(keep=False),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph IDs in split CSV: {duplicates}"
        )

    if len(split_df) != 68:
        raise ValueError(
            f"Expected 68 split rows, found {len(split_df)}."
        )

    split_counts = split_df["split"].value_counts()

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        actual_count = int(split_counts.get(split, 0))

        if actual_count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} graphs, "
                f"found {actual_count}."
            )

    return split_df


def load_targets() -> pd.DataFrame:
    if not TARGET_CSV.exists():
        raise FileNotFoundError(
            f"Week 18 target CSV not found: {TARGET_CSV}"
        )

    targets = pd.read_csv(TARGET_CSV)

    required_columns = {
        "graph_id",
        "family",
        "selected_teacher_ordering",
        "selected_teacher_num_colors",
        "best_colpack5_orderings",
        "num_best_orderings",
        "node_id",
        "order_position",
        "target_score",
        "num_vertices",
        "matrix_path",
        "source_file",
    }

    missing_columns = required_columns - set(targets.columns)

    if missing_columns:
        raise ValueError(
            f"Target CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return targets


def load_target_summary() -> pd.DataFrame:
    if not TARGET_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Week 18 target summary not found: "
            f"{TARGET_SUMMARY_CSV}"
        )

    summary = pd.read_csv(TARGET_SUMMARY_CSV)

    required_columns = {
        "graph_id",
        "family",
        "num_vertices",
        "selected_teacher_ordering",
        "selected_teacher_num_colors",
        "best_colpack5_orderings",
        "num_best_orderings",
        "target_rows",
        "source_file",
        "matrix_path",
    }

    missing_columns = required_columns - set(summary.columns)

    if missing_columns:
        raise ValueError(
            f"Target summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if summary["graph_id"].duplicated().any():
        duplicates = summary.loc[
            summary["graph_id"].duplicated(keep=False),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph IDs in target summary: "
            f"{duplicates}"
        )

    return summary


def resolve_matrix_path(matrix_path_value: str) -> Path:
    matrix_path = Path(matrix_path_value)

    if not matrix_path.is_absolute():
        matrix_path = PROJECT_ROOT / matrix_path

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Matrix file not found: {matrix_path}"
        )

    return matrix_path


def verify_single_value(
    graph_id: str,
    values: list[object],
    field_name: str,
) -> object:
    unique_values = pd.Series(values).drop_duplicates().tolist()

    if len(unique_values) != 1:
        raise ValueError(
            f"{graph_id}: expected one {field_name} value, "
            f"found {unique_values}."
        )

    return unique_values[0]


def build_data_for_graph(
    split_row: object,
    targets: pd.DataFrame,
    target_summary: pd.DataFrame,
) -> Data:
    graph_id = str(split_row.graph_id)
    split = str(split_row.split)
    family = str(split_row.family)

    matrix_path = resolve_matrix_path(
        str(split_row.matrix_path)
    )

    graph = load_graph_from_mtx(matrix_path)

    expected_num_vertices = int(split_row.num_vertices)
    expected_num_edges = int(split_row.num_edges)

    if graph.number_of_nodes() != expected_num_vertices:
        raise ValueError(
            f"{graph_id}: graph has "
            f"{graph.number_of_nodes()} vertices, but split CSV "
            f"expects {expected_num_vertices}."
        )

    if graph.number_of_edges() != expected_num_edges:
        raise ValueError(
            f"{graph_id}: graph has "
            f"{graph.number_of_edges()} edges, but split CSV "
            f"expects {expected_num_edges}."
        )

    features = extract_node_features(graph)

    validate_feature_matrix(
        features,
        graph.number_of_nodes(),
    )

    if features.shape[1] != 25:
        raise ValueError(
            f"{graph_id}: expected 25 symmetry-breaking "
            f"features, found {features.shape[1]}."
        )

    graph_targets = (
        targets[targets["graph_id"] == graph_id]
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    if len(graph_targets) != graph.number_of_nodes():
        raise ValueError(
            f"{graph_id}: target rows "
            f"({len(graph_targets)}) do not match graph nodes "
            f"({graph.number_of_nodes()})."
        )

    expected_node_ids = list(
        range(graph.number_of_nodes())
    )

    actual_node_ids = (
        graph_targets["node_id"]
        .astype(int)
        .tolist()
    )

    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"{graph_id}: target node IDs are not exactly "
            f"0 to {graph.number_of_nodes() - 1}."
        )

    order_positions = (
        graph_targets["order_position"]
        .astype(int)
        .tolist()
    )

    if sorted(order_positions) != expected_node_ids:
        raise ValueError(
            f"{graph_id}: ordering positions are not exactly "
            f"0 to {graph.number_of_nodes() - 1}."
        )

    target_summary_rows = target_summary[
        target_summary["graph_id"] == graph_id
    ]

    if len(target_summary_rows) != 1:
        raise ValueError(
            f"{graph_id}: expected exactly one target-summary row, "
            f"found {len(target_summary_rows)}."
        )

    target_summary_row = target_summary_rows.iloc[0]

    target_teacher = str(
        verify_single_value(
            graph_id=graph_id,
            values=graph_targets[
                "selected_teacher_ordering"
            ].tolist(),
            field_name="selected teacher",
        )
    )

    target_num_colors = int(
        verify_single_value(
            graph_id=graph_id,
            values=graph_targets[
                "selected_teacher_num_colors"
            ].tolist(),
            field_name="selected teacher color count",
        )
    )

    target_family = str(
        verify_single_value(
            graph_id=graph_id,
            values=graph_targets["family"].tolist(),
            field_name="family",
        )
    )

    if target_teacher != str(
        split_row.selected_teacher_ordering
    ):
        raise ValueError(
            f"{graph_id}: split teacher "
            f"{split_row.selected_teacher_ordering} does not match "
            f"target teacher {target_teacher}."
        )

    if target_num_colors != int(
        split_row.best_colpack5_colors
    ):
        raise ValueError(
            f"{graph_id}: target color count "
            f"{target_num_colors} does not match best ColPack-5 "
            f"count {split_row.best_colpack5_colors}."
        )

    if target_family != family:
        raise ValueError(
            f"{graph_id}: target family {target_family} "
            f"does not match split family {family}."
        )

    if int(target_summary_row["target_rows"]) != graph.number_of_nodes():
        raise ValueError(
            f"{graph_id}: target summary reports "
            f"{target_summary_row['target_rows']} rows, expected "
            f"{graph.number_of_nodes()}."
        )

    x = torch.tensor(
        features,
        dtype=torch.float32,
    )

    edge_index = graph_to_edge_index(graph)

    y = torch.tensor(
        graph_targets["target_score"].values,
        dtype=torch.float32,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        graph_id=graph_id,
        split=split,
        group=family,
        graph_family=family,
        split_group_id=str(split_row.split_group_id),
        split_grouping_mode=str(
            split_row.split_grouping_mode
        ),
        selected_ordering=target_teacher,
        selected_teacher_ordering=target_teacher,
        selected_num_colors=target_num_colors,
        selected_teacher_num_colors=target_num_colors,
        best_colpack5_colors=int(
            split_row.best_colpack5_colors
        ),
        worst_colpack5_colors=int(
            split_row.worst_colpack5_colors
        ),
        ordering_gap=int(split_row.ordering_gap),
        best_colpack5_orderings=str(
            split_row.best_colpack5_orderings
        ),
        num_best_orderings=int(
            split_row.num_best_orderings
        ),
        teacher_selection_reason=str(
            split_row.teacher_selection_reason
        ),
        density=float(split_row.density),
        matrix_path=str(split_row.matrix_path),
        num_nodes=graph.number_of_nodes(),
    )

    validate_pyg_data(data)

    if data.x.shape != (
        graph.number_of_nodes(),
        25,
    ):
        raise ValueError(
            f"{graph_id}: unexpected x shape "
            f"{tuple(data.x.shape)}."
        )

    if data.y.shape[0] != data.num_nodes:
        raise ValueError(
            f"{graph_id}: y has {data.y.shape[0]} rows, "
            f"but graph has {data.num_nodes} nodes."
        )

    if data.edge_index.shape[0] != 2:
        raise ValueError(
            f"{graph_id}: edge_index has unexpected shape "
            f"{tuple(data.edge_index.shape)}."
        )

    return data


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_df = load_split()
    targets = load_targets()
    target_summary = load_target_summary()

    split_graph_ids = set(
        split_df["graph_id"].astype(str)
    )

    target_graph_ids = set(
        targets["graph_id"].astype(str)
    )

    summary_graph_ids = set(
        target_summary["graph_id"].astype(str)
    )

    if split_graph_ids != target_graph_ids:
        raise ValueError(
            "Graph IDs differ between split and target files.\n"
            f"Only in split: "
            f"{sorted(split_graph_ids - target_graph_ids)}\n"
            f"Only in targets: "
            f"{sorted(target_graph_ids - split_graph_ids)}"
        )

    if split_graph_ids != summary_graph_ids:
        raise ValueError(
            "Graph IDs differ between split and target summary.\n"
            f"Only in split: "
            f"{sorted(split_graph_ids - summary_graph_ids)}\n"
            f"Only in summary: "
            f"{sorted(summary_graph_ids - split_graph_ids)}"
        )

    dataset: list[Data] = []
    summary_rows: list[dict[str, object]] = []

    for row in split_df.itertuples(index=False):
        data = build_data_for_graph(
            split_row=row,
            targets=targets,
            target_summary=target_summary,
        )

        output_path = (
            OUTPUT_DIR
            / f"{row.graph_id}.pt"
        )

        torch.save(
            data,
            output_path,
        )

        dataset.append(data)

        summary_rows.append(
            {
                "graph_id": row.graph_id,
                "split": row.split,
                "split_group_id": row.split_group_id,
                "graph_family": data.graph_family,
                "num_nodes": int(data.num_nodes),
                "num_undirected_edges": int(
                    row.num_edges
                ),
                "num_directed_edges": int(
                    data.edge_index.shape[1]
                ),
                "num_features": int(
                    data.x.shape[1]
                ),
                "selected_teacher_ordering": (
                    data.selected_teacher_ordering
                ),
                "selected_teacher_num_colors": int(
                    data.selected_teacher_num_colors
                ),
                "best_colpack5_colors": int(
                    data.best_colpack5_colors
                ),
                "worst_colpack5_colors": int(
                    data.worst_colpack5_colors
                ),
                "ordering_gap": int(
                    data.ordering_gap
                ),
                "num_best_orderings": int(
                    data.num_best_orderings
                ),
                "density": float(data.density),
                "path": str(output_path),
            }
        )

        print(
            f"Saved {row.graph_id}: "
            f"split={row.split}, "
            f"family={row.family}, "
            f"nodes={data.num_nodes}, "
            f"features={data.x.shape[1]}, "
            f"teacher={data.selected_teacher_ordering}, "
            f"colors={data.selected_teacher_num_colors}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    PYG_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        PYG_SUMMARY_CSV,
        index=False,
    )

    saved_files = list(
        OUTPUT_DIR.glob("*.pt")
    )

    if len(saved_files) != 68:
        raise ValueError(
            f"Expected 68 saved PyG files, "
            f"found {len(saved_files)}."
        )

    print()
    print(
        "Week 18 heterogeneous PyG dataset "
        "built successfully."
    )
    print("------------------------------------")
    print(f"Total graphs: {len(dataset)}")
    print(f"Total saved .pt files: {len(saved_files)}")
    print()

    print("Split counts:")
    print(
        summary_df["split"]
        .value_counts()
        .reindex(
            ["train", "validation", "test"]
        )
        .to_string()
    )
    print()

    print("Teacher counts by split:")
    print(
        pd.crosstab(
            summary_df["split"],
            summary_df[
                "selected_teacher_ordering"
            ],
        )
        .reindex(
            index=[
                "train",
                "validation",
                "test",
            ],
            columns=EXPECTED_TEACHERS,
            fill_value=0,
        )
        .to_string()
    )
    print()

    print("Family counts by split:")
    print(
        pd.crosstab(
            summary_df["split"],
            summary_df["graph_family"],
        )
        .reindex(
            index=[
                "train",
                "validation",
                "test",
            ],
            fill_value=0,
        )
        .to_string()
    )
    print()

    print("Feature-count check:")
    print(
        summary_df["num_features"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print(
        f"Saved PyG summary to: "
        f"{PYG_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()