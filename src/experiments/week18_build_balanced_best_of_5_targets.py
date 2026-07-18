from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_balanced_best_of_5_teacher_assignments.csv"
)

COLPACK_OUTPUT_DIRS = [
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_heterogeneous_generalization",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_additional_heterogeneous_generalization",
]

TARGET_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week18_balanced_best_of_5_ordering_targets.csv"
)

TARGET_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_balanced_best_of_5_target_summary.csv"
)

ORDER_LINE_PATTERN = re.compile(
    r"Order Position\s+(\d+)\s+->\s+Vertex\s+(\d+)"
)


def output_filename(
    graph_id: str,
    ordering_name: str,
) -> str:
    return f"{graph_id}_{ordering_name.lower()}.txt"


def find_colpack_output(
    graph_id: str,
    ordering_name: str,
) -> Path:
    filename = output_filename(
        graph_id=graph_id,
        ordering_name=ordering_name,
    )

    matches = [
        directory / filename
        for directory in COLPACK_OUTPUT_DIRS
        if (directory / filename).exists()
    ]

    if not matches:
        raise FileNotFoundError(
            f"No ColPack output found for graph={graph_id}, "
            f"ordering={ordering_name}, filename={filename}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple ColPack outputs found for {graph_id}, "
            f"{ordering_name}: {matches}"
        )

    return matches[0]


def parse_ordering(path: Path) -> list[tuple[int, int]]:
    ordering: list[tuple[int, int]] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            match = ORDER_LINE_PATTERN.search(line)

            if match:
                order_position = int(match.group(1))
                node_id = int(match.group(2))

                ordering.append(
                    (order_position, node_id)
                )

    if not ordering:
        raise ValueError(
            f"No ordering lines found in: {path}"
        )

    ordering.sort(
        key=lambda item: item[0]
    )

    return ordering


def validate_ordering(
    graph_id: str,
    ordering: list[tuple[int, int]],
    expected_num_vertices: int,
) -> None:
    if len(ordering) != expected_num_vertices:
        raise ValueError(
            f"{graph_id}: parsed {len(ordering)} ordering rows, "
            f"expected {expected_num_vertices}."
        )

    positions = [
        position
        for position, _ in ordering
    ]

    node_ids = [
        node_id
        for _, node_id in ordering
    ]

    expected_values = list(
        range(expected_num_vertices)
    )

    if sorted(positions) != expected_values:
        raise ValueError(
            f"{graph_id}: ordering positions are not exactly "
            f"0 to {expected_num_vertices - 1}."
        )

    if sorted(node_ids) != expected_values:
        raise ValueError(
            f"{graph_id}: node IDs are not exactly "
            f"0 to {expected_num_vertices - 1}."
        )


def build_graph_targets(
    graph_id: str,
    family: str,
    selected_teacher: str,
    selected_num_colors: int,
    best_orderings: str,
    num_best_orderings: int,
    num_vertices: int,
    matrix_path: str,
) -> tuple[list[dict[str, object]], Path]:
    source_file = find_colpack_output(
        graph_id=graph_id,
        ordering_name=selected_teacher,
    )

    ordering = parse_ordering(source_file)

    validate_ordering(
        graph_id=graph_id,
        ordering=ordering,
        expected_num_vertices=num_vertices,
    )

    rows: list[dict[str, object]] = []

    for order_position, node_id in ordering:
        if num_vertices > 1:
            target_score = (
                num_vertices - 1 - order_position
            ) / (num_vertices - 1)
        else:
            target_score = 1.0

        rows.append(
            {
                "graph_id": graph_id,
                "family": family,
                "selected_teacher_ordering": selected_teacher,
                "selected_teacher_num_colors": (
                    selected_num_colors
                ),
                "best_colpack5_orderings": best_orderings,
                "num_best_orderings": num_best_orderings,
                "node_id": node_id,
                "order_position": order_position,
                "target_score": target_score,
                "num_vertices": num_vertices,
                "matrix_path": matrix_path,
                "source_file": str(
                    source_file.relative_to(PROJECT_ROOT)
                ),
            }
        )

    return rows, source_file


def main() -> None:
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            f"Teacher-assignment file not found: "
            f"{ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH
    )

    required_columns = {
        "graph_id",
        "family",
        "num_vertices",
        "matrix_path",
        "best_colpack5_colors",
        "best_colpack5_orderings",
        "num_best_orderings",
        "selected_teacher_ordering",
        "selected_teacher_is_colpack_best",
    }

    missing_columns = (
        required_columns
        - set(assignments.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if assignments["graph_id"].duplicated().any():
        duplicates = assignments.loc[
            assignments["graph_id"].duplicated(
                keep=False
            ),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph assignments: {duplicates}"
        )

    if not assignments[
        "selected_teacher_is_colpack_best"
    ].astype(bool).all():
        invalid_graphs = assignments.loc[
            ~assignments[
                "selected_teacher_is_colpack_best"
            ].astype(bool),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Unverified teacher assignments found: "
            f"{invalid_graphs}"
        )

    all_target_rows: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    for row in assignments.itertuples(
        index=False
    ):
        graph_id = str(row.graph_id)
        family = str(row.family)
        selected_teacher = str(
            row.selected_teacher_ordering
        )

        selected_num_colors = int(
            row.best_colpack5_colors
        )

        num_vertices = int(
            row.num_vertices
        )

        num_best_orderings = int(
            row.num_best_orderings
        )

        graph_target_rows, source_file = (
            build_graph_targets(
                graph_id=graph_id,
                family=family,
                selected_teacher=selected_teacher,
                selected_num_colors=(
                    selected_num_colors
                ),
                best_orderings=str(
                    row.best_colpack5_orderings
                ),
                num_best_orderings=(
                    num_best_orderings
                ),
                num_vertices=num_vertices,
                matrix_path=str(row.matrix_path),
            )
        )

        all_target_rows.extend(
            graph_target_rows
        )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "family": family,
                "num_vertices": num_vertices,
                "selected_teacher_ordering": (
                    selected_teacher
                ),
                "selected_teacher_num_colors": (
                    selected_num_colors
                ),
                "best_colpack5_orderings": str(
                    row.best_colpack5_orderings
                ),
                "num_best_orderings": (
                    num_best_orderings
                ),
                "target_rows": len(
                    graph_target_rows
                ),
                "source_file": str(
                    source_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "matrix_path": str(
                    row.matrix_path
                ),
            }
        )

    target_df = pd.DataFrame(
        all_target_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    expected_target_rows = int(
        assignments["num_vertices"].sum()
    )

    if len(target_df) != expected_target_rows:
        raise ValueError(
            f"Expected {expected_target_rows} target rows, "
            f"but created {len(target_df)}."
        )

    if target_df["graph_id"].nunique() != len(
        assignments
    ):
        raise ValueError(
            "Target graph count does not match "
            "assignment graph count."
        )

    target_counts = (
        target_df
        .groupby("graph_id")
        .size()
    )

    expected_counts = (
        assignments
        .set_index("graph_id")[
            "num_vertices"
        ]
        .astype(int)
    )

    mismatched_graphs = [
        graph_id
        for graph_id in expected_counts.index
        if int(target_counts[graph_id])
        != int(expected_counts[graph_id])
    ]

    if mismatched_graphs:
        raise ValueError(
            f"Target-row count mismatch for: "
            f"{mismatched_graphs}"
        )

    TARGET_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TARGET_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_df.to_csv(
        TARGET_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        TARGET_SUMMARY_PATH,
        index=False,
    )

    teacher_distribution = (
        summary_df[
            "selected_teacher_ordering"
        ]
        .value_counts()
        .sort_index()
    )

    family_distribution = (
        summary_df["family"]
        .value_counts()
        .sort_index()
    )

    print(
        "Week 18 balanced best-of-five "
        "node-level targets"
    )
    print("----------------------------------")
    print(
        f"Graphs processed: "
        f"{summary_df['graph_id'].nunique()}"
    )
    print(
        f"Total target rows: "
        f"{len(target_df)}"
    )
    print(
        f"Expected target rows: "
        f"{expected_target_rows}"
    )
    print()

    print("Teacher distribution:")

    for ordering, count in (
        teacher_distribution.items()
    ):
        print(
            f"  {ordering}: {int(count)}"
        )

    print()
    print("Graph-family distribution:")

    for family, count in (
        family_distribution.items()
    ):
        print(
            f"  {family}: {int(count)}"
        )

    print()
    print(
        "Every graph has one target row per vertex."
    )
    print(
        "All node IDs and ordering positions were "
        "validated."
    )
    print()
    print(
        f"Saved node targets to: "
        f"{TARGET_OUTPUT_PATH}"
    )
    print(
        f"Saved target summary to: "
        f"{TARGET_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()