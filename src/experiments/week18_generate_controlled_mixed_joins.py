from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week18_controlled_mixed_joins"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_candidate_summary.csv"
)


# Only cycle sizes already present in the original training set.
# Validation size 38 and test size 41 remain completely frozen.
TRAINING_BASE_SIZES = [
    20,
    26,
    29,
    35,
]

COMPONENT_COUNTS = [
    2,
    3,
    4,
    5,
]

EXPECTED_CANDIDATE_COUNT = 105


def build_cycle_square_graph(
    n: int,
) -> nx.Graph:
    """
    Build C_n^2, the square of a cycle on n vertices.

    Vertices at cycle distance one or two are adjacent.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for node in range(n):
        graph.add_edge(
            node,
            (node + 1) % n,
        )

        graph.add_edge(
            node,
            (node + 2) % n,
        )

    return graph


def exact_coloring_cycle_square_3r_plus_2(
    n: int,
) -> dict[int, int]:
    """
    Construct the known four-coloring for C_n^2 when n = 3r + 2.
    """
    if (n - 2) % 3 != 0:
        raise ValueError(
            f"Expected n = 3r + 2, but received n={n}."
        )

    colors: list[int] = []

    for index in range(n - 5):
        colors.append(index % 3)

    colors.extend(
        [
            0,
            3,
            1,
            2,
            3,
        ]
    )

    if len(colors) != n:
        raise ValueError(
            f"Generated {len(colors)} colors for a "
            f"{n}-vertex graph."
        )

    return {
        node: colors[node]
        for node in range(n)
    }


def validate_coloring(
    graph: nx.Graph,
    coloring: dict[int, int],
) -> bool:
    """
    Check that every edge has differently colored endpoints.
    """
    if set(coloring) != set(graph.nodes()):
        return False

    for source, target in graph.edges():
        if coloring[source] == coloring[target]:
            return False

    return True


def join_graphs_with_colorings(
    component_sizes: tuple[int, ...],
) -> tuple[nx.Graph, dict[int, int]]:
    """
    Build the join of cycle-square components of potentially
    different sizes.

    Every component receives its own four-color range because every
    vertex of one component is adjacent to every vertex of all other
    components.
    """
    joined_graph = nx.Graph()
    joined_coloring: dict[int, int] = {}

    component_node_sets: list[list[int]] = []

    next_node_id = 0

    for component_index, cycle_size in enumerate(
        component_sizes
    ):
        component_graph = build_cycle_square_graph(
            cycle_size
        )

        component_coloring = (
            exact_coloring_cycle_square_3r_plus_2(
                cycle_size
            )
        )

        if not validate_coloring(
            component_graph,
            component_coloring,
        ):
            raise ValueError(
                f"Invalid base coloring for C_{cycle_size}^2."
            )

        node_mapping = {
            old_node: next_node_id + old_node
            for old_node in component_graph.nodes()
        }

        relabeled_graph = nx.relabel_nodes(
            component_graph,
            node_mapping,
        )

        joined_graph.add_nodes_from(
            relabeled_graph.nodes()
        )

        joined_graph.add_edges_from(
            relabeled_graph.edges()
        )

        relabeled_nodes = sorted(
            relabeled_graph.nodes()
        )

        component_node_sets.append(
            relabeled_nodes
        )

        color_offset = (
            4 * component_index
        )

        for old_node, old_color in (
            component_coloring.items()
        ):
            joined_coloring[
                node_mapping[old_node]
            ] = (
                old_color
                + color_offset
            )

        next_node_id += cycle_size

    # Add all edges between different components.
    for first_index in range(
        len(component_node_sets)
    ):
        for second_index in range(
            first_index + 1,
            len(component_node_sets),
        ):
            for first_node in (
                component_node_sets[
                    first_index
                ]
            ):
                for second_node in (
                    component_node_sets[
                        second_index
                    ]
                ):
                    joined_graph.add_edge(
                        first_node,
                        second_node,
                    )

    return (
        joined_graph,
        joined_coloring,
    )


def save_graph_as_mtx(
    graph: nx.Graph,
    output_path: Path,
) -> None:
    """
    Save the undirected adjacency pattern in Matrix Market format.
    """
    sorted_nodes = sorted(
        graph.nodes()
    )

    node_to_index = {
        node: index
        for index, node in enumerate(
            sorted_nodes
        )
    }

    rows: list[int] = []
    columns: list[int] = []

    for source, target in graph.edges():
        source_index = node_to_index[
            source
        ]

        target_index = node_to_index[
            target
        ]

        rows.extend(
            [
                source_index,
                target_index,
            ]
        )

        columns.extend(
            [
                target_index,
                source_index,
            ]
        )

    values = [1] * len(rows)

    matrix = coo_matrix(
        (
            values,
            (
                rows,
                columns,
            ),
        ),
        shape=(
            len(sorted_nodes),
            len(sorted_nodes),
        ),
    )

    mmwrite(
        output_path,
        matrix,
    )


def build_graph_id(
    component_sizes: tuple[int, ...],
) -> str:
    """
    Build a deterministic ID that records every component size.
    """
    component_part = "_".join(
        f"c{cycle_size}"
        for cycle_size in component_sizes
    )

    return (
        "week18_controlled_mixed_"
        f"join_{len(component_sizes)}_"
        f"{component_part}"
    )


def generate_component_specs() -> list[
    tuple[int, ...]
]:
    """
    Generate all mixed multisets of two to five training component
    sizes, excluding homogeneous joins already available in Week 17.
    """
    specs: list[tuple[int, ...]] = []

    for component_count in (
        COMPONENT_COUNTS
    ):
        for component_sizes in (
            combinations_with_replacement(
                TRAINING_BASE_SIZES,
                component_count,
            )
        ):
            # Homogeneous joins are already present in Week 17.
            if len(set(component_sizes)) == 1:
                continue

            specs.append(
                tuple(component_sizes)
            )

    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} mixed joins, "
            f"generated {len(specs)}."
        )

    return specs


def main() -> None:
    OUTPUT_MATRIX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    component_specs = (
        generate_component_specs()
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    seen_graph_ids: set[str] = set()

    for component_sizes in (
        component_specs
    ):
        graph_id = build_graph_id(
            component_sizes
        )

        if graph_id in seen_graph_ids:
            raise ValueError(
                f"Duplicate graph ID generated: "
                f"{graph_id}"
            )

        seen_graph_ids.add(
            graph_id
        )

        graph, known_coloring = (
            join_graphs_with_colorings(
                component_sizes
            )
        )

        coloring_valid = validate_coloring(
            graph,
            known_coloring,
        )

        if not coloring_valid:
            raise ValueError(
                f"{graph_id}: known coloring is invalid."
            )

        known_chromatic_number = (
            max(
                known_coloring.values()
            )
            + 1
        )

        expected_target = (
            4 * len(component_sizes)
        )

        if (
            known_chromatic_number
            != expected_target
        ):
            raise ValueError(
                f"{graph_id}: expected target "
                f"{expected_target}, found "
                f"{known_chromatic_number}."
            )

        matrix_path = (
            OUTPUT_MATRIX_DIR
            / f"{graph_id}.mtx"
        )

        save_graph_as_mtx(
            graph,
            matrix_path,
        )

        component_size_text = ";".join(
            str(size)
            for size in component_sizes
        )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "family": (
                    "mixed_join_of_"
                    "cycle_square_hard_cases"
                ),
                "construction": (
                    "join of cycle-square components "
                    + ", ".join(
                        f"C_{size}^2"
                        for size in component_sizes
                    )
                ),
                "num_components_joined": (
                    len(component_sizes)
                ),
                "component_cycle_sizes": (
                    component_size_text
                ),
                "num_unique_component_sizes": (
                    len(
                        set(component_sizes)
                    )
                ),
                "minimum_component_size": (
                    min(component_sizes)
                ),
                "maximum_component_size": (
                    max(component_sizes)
                ),
                "num_vertices": (
                    graph.number_of_nodes()
                ),
                "num_edges": (
                    graph.number_of_edges()
                ),
                "known_chromatic_number": (
                    known_chromatic_number
                ),
                "known_coloring_valid": (
                    coloring_valid
                ),
                "matrix_path": str(
                    matrix_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if len(summary_df) != (
        EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} rows, "
            f"found {len(summary_df)}."
        )

    if summary_df[
        "graph_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in summary."
        )

    if not bool(
        summary_df[
            "known_coloring_valid"
        ].all()
    ):
        raise ValueError(
            "At least one known coloring is invalid."
        )

    summary_df = summary_df.sort_values(
        [
            "num_components_joined",
            "num_vertices",
            "graph_id",
        ]
    ).reset_index(drop=True)

    summary_df.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    print(
        "Week 18 controlled mixed joins generated."
    )
    print(
        "-----------------------------------------"
    )
    print(
        f"Total candidates: {len(summary_df)}"
    )
    print()

    print(
        "Candidates by component count:"
    )

    count_summary = (
        summary_df
        .groupby(
            "num_components_joined"
        )
        .agg(
            num_graphs=(
                "graph_id",
                "count",
            ),
            minimum_vertices=(
                "num_vertices",
                "min",
            ),
            maximum_vertices=(
                "num_vertices",
                "max",
            ),
            known_target_colors=(
                "known_chromatic_number",
                "first",
            ),
        )
    )

    print(
        count_summary.to_string()
    )
    print()

    print(
        "Unique component-size diversity:"
    )

    print(
        pd.crosstab(
            summary_df[
                "num_components_joined"
            ],
            summary_df[
                "num_unique_component_sizes"
            ],
        ).to_string()
    )
    print()

    print(
        "All known target colorings are valid."
    )
    print(
        "Cycle sizes 38 and 41 were not used."
    )
    print(
        f"Saved matrices to: "
        f"{OUTPUT_MATRIX_DIR}"
    )
    print(
        f"Saved candidate summary to: "
        f"{OUTPUT_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()