from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx
from src.training.learned_ordering import scores_to_ordering
from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week18_controlled_mixed_joins"
)

CANDIDATE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_candidate_summary.csv"
)

COLPACK_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_join_colpack_summary.csv"
)

TARGET_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week18_controlled_mixed_exact_ordering_targets.csv"
)

TARGET_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_controlled_mixed_exact_target_summary.csv"
)

EXPECTED_GRAPH_COUNT = 105

SELECTED_ORDERING_NAME = (
    "EXACT_KNOWN_COLOR_CLASS_ORDERING"
)


def parse_component_sizes(
    value: object,
) -> tuple[int, ...]:
    """
    Parse component sizes stored as strings such as '20;26;35'.
    """
    parts = [
        part.strip()
        for part in str(value).split(";")
        if part.strip()
    ]

    component_sizes = tuple(
        int(part)
        for part in parts
    )

    if not component_sizes:
        raise ValueError(
            f"Could not parse component sizes from: {value}"
        )

    return component_sizes


def exact_coloring_cycle_square_3r_plus_2(
    n: int,
) -> dict[int, int]:
    """
    Construct the known four-coloring of C_n^2 for n = 3r + 2.
    """
    if (n - 2) % 3 != 0:
        raise ValueError(
            f"Expected n = 3r + 2, received n={n}."
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
            f"Generated {len(colors)} colors "
            f"for a graph with {n} vertices."
        )

    return {
        node: colors[node]
        for node in range(n)
    }


def build_join_known_coloring(
    component_sizes: tuple[int, ...],
) -> dict[int, int]:
    """
    Reconstruct the exact coloring used when the mixed join was built.

    Each component has four colors. Since all vertices in different
    components are adjacent, each component receives a disjoint
    four-color range.
    """
    joined_coloring: dict[int, int] = {}

    node_offset = 0

    for component_index, cycle_size in enumerate(
        component_sizes
    ):
        component_coloring = (
            exact_coloring_cycle_square_3r_plus_2(
                cycle_size
            )
        )

        color_offset = (
            4 * component_index
        )

        for local_node, local_color in (
            component_coloring.items()
        ):
            global_node = (
                node_offset
                + local_node
            )

            joined_coloring[
                global_node
            ] = (
                color_offset
                + local_color
            )

        node_offset += cycle_size

    return joined_coloring


def validate_known_coloring(
    graph: nx.Graph,
    coloring: dict[int, int],
) -> bool:
    if set(coloring) != set(graph.nodes()):
        return False

    for source, target in graph.edges():
        if coloring[source] == coloring[target]:
            return False

    return True


def build_exact_ordering(
    coloring: dict[int, int],
) -> list[int]:
    """
    Order vertices by known color class and then by node ID.

    Processing all vertices of color 0 first, followed by color 1,
    and so on guarantees that greedy coloring uses no more colors
    than the known coloring.
    """
    return sorted(
        coloring,
        key=lambda node: (
            coloring[node],
            node,
        ),
    )


def normalize_ordering(
    ordering: object,
) -> list[int]:
    if isinstance(ordering, torch.Tensor):
        values = ordering.detach().cpu().tolist()
    else:
        values = list(ordering)

    normalized: list[int] = []

    for value in values:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(
                    "Unexpected nested ordering value: "
                    f"{value}"
                )

            value = value[0]

        normalized.append(int(value))

    return normalized


def build_target_scores(
    ordering: list[int],
    num_nodes: int,
) -> tuple[list[float], str]:
    """
    Build node scores that reproduce the exact ordering through the
    repository's existing scores_to_ordering function.

    Both possible conventions are tested:
    - high score means earlier;
    - low score means earlier.
    """
    if len(ordering) != num_nodes:
        raise ValueError(
            "Ordering length does not match graph size."
        )

    if set(ordering) != set(range(num_nodes)):
        raise ValueError(
            "Ordering is not a permutation of node IDs."
        )

    denominator = max(
        num_nodes - 1,
        1,
    )

    high_score_first = [
        0.0
        for _ in range(num_nodes)
    ]

    low_score_first = [
        0.0
        for _ in range(num_nodes)
    ]

    for position, node in enumerate(ordering):
        high_score_first[node] = (
            1.0
            - position / denominator
        )

        low_score_first[node] = (
            position / denominator
        )

    candidates = [
        (
            high_score_first,
            "higher_score_processed_earlier",
        ),
        (
            low_score_first,
            "lower_score_processed_earlier",
        ),
    ]

    for scores, convention in candidates:
        score_tensor = torch.tensor(
            scores,
            dtype=torch.float32,
        ).view(-1, 1)

        recovered_ordering = normalize_ordering(
            scores_to_ordering(
                score_tensor
            )
        )

        if recovered_ordering == ordering:
            return (
                scores,
                convention,
            )

    raise ValueError(
        "Could not create target scores that reproduce "
        "the exact ordering through scores_to_ordering."
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not CANDIDATE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Candidate summary not found: "
            f"{CANDIDATE_SUMMARY_PATH}"
        )

    if not COLPACK_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "ColPack summary not found: "
            f"{COLPACK_SUMMARY_PATH}"
        )

    candidate_df = pd.read_csv(
        CANDIDATE_SUMMARY_PATH
    )

    colpack_df = pd.read_csv(
        COLPACK_SUMMARY_PATH
    )

    if len(candidate_df) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} candidates, "
            f"found {len(candidate_df)}."
        )

    if len(colpack_df) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} ColPack rows, "
            f"found {len(colpack_df)}."
        )

    if candidate_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs in candidate summary."
        )

    if colpack_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs in ColPack summary."
        )

    candidate_ids = set(
        candidate_df["graph_id"]
    )

    colpack_ids = set(
        colpack_df["graph_id"]
    )

    if candidate_ids != colpack_ids:
        missing_from_colpack = sorted(
            candidate_ids - colpack_ids
        )

        missing_from_candidates = sorted(
            colpack_ids - candidate_ids
        )

        raise ValueError(
            "Candidate and ColPack graph IDs differ.\n"
            f"Missing from ColPack: {missing_from_colpack}\n"
            f"Missing from candidates: "
            f"{missing_from_candidates}"
        )

    return (
        candidate_df,
        colpack_df,
    )


def main() -> None:
    candidate_df, colpack_df = (
        load_inputs()
    )

    colpack_lookup = (
        colpack_df
        .set_index("graph_id")
        .to_dict(orient="index")
    )

    target_rows: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    score_conventions: set[str] = set()

    for row in candidate_df.itertuples(
        index=False
    ):
        graph_id = str(
            row.graph_id
        )

        matrix_path = (
            MATRIX_DIR
            / f"{graph_id}.mtx"
        )

        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Matrix not found for {graph_id}: "
                f"{matrix_path}"
            )

        graph = load_graph_from_mtx(
            matrix_path
        )

        expected_nodes = int(
            row.num_vertices
        )

        expected_edges = int(
            row.num_edges
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

        expected_node_ids = set(
            range(expected_nodes)
        )

        if set(graph.nodes()) != expected_node_ids:
            raise ValueError(
                f"{graph_id}: graph nodes are not "
                f"0..{expected_nodes - 1}."
            )

        component_sizes = (
            parse_component_sizes(
                row.component_cycle_sizes
            )
        )

        num_components = int(
            row.num_components_joined
        )

        if len(component_sizes) != num_components:
            raise ValueError(
                f"{graph_id}: component-size count "
                f"{len(component_sizes)} does not match "
                f"num_components_joined={num_components}."
            )

        if sum(component_sizes) != expected_nodes:
            raise ValueError(
                f"{graph_id}: component sizes sum to "
                f"{sum(component_sizes)}, expected "
                f"{expected_nodes} vertices."
            )

        known_coloring = (
            build_join_known_coloring(
                component_sizes
            )
        )

        if not validate_known_coloring(
            graph,
            known_coloring,
        ):
            raise ValueError(
                f"{graph_id}: reconstructed known "
                "coloring is invalid."
            )

        known_target = int(
            row.known_chromatic_number
        )

        reconstructed_target = (
            max(
                known_coloring.values()
            )
            + 1
        )

        expected_target = (
            4 * num_components
        )

        if known_target != expected_target:
            raise ValueError(
                f"{graph_id}: summary target "
                f"{known_target} does not match "
                f"expected target {expected_target}."
            )

        if reconstructed_target != known_target:
            raise ValueError(
                f"{graph_id}: reconstructed coloring "
                f"uses {reconstructed_target} colors, "
                f"expected {known_target}."
            )

        exact_ordering = (
            build_exact_ordering(
                known_coloring
            )
        )

        greedy_coloring = (
            greedy_color_with_ordering(
                graph=graph,
                ordering=exact_ordering,
            )
        )

        selected_num_colors = int(
            count_colors(
                greedy_coloring
            )
        )

        target_coloring_valid = bool(
            is_valid_coloring(
                graph,
                greedy_coloring,
            )
        )

        if not target_coloring_valid:
            raise ValueError(
                f"{graph_id}: greedy coloring from "
                "the exact ordering is invalid."
            )

        if selected_num_colors != known_target:
            raise ValueError(
                f"{graph_id}: exact ordering produced "
                f"{selected_num_colors} colors, "
                f"expected {known_target}."
            )

        target_scores, score_convention = (
            build_target_scores(
                ordering=exact_ordering,
                num_nodes=expected_nodes,
            )
        )

        score_conventions.add(
            score_convention
        )

        order_position_by_node = {
            node: position
            for position, node
            in enumerate(exact_ordering)
        }

        colpack_row = (
            colpack_lookup[
                graph_id
            ]
        )

        best_colpack5_colors = int(
            colpack_row[
                "best_colpack5_colors"
            ]
        )

        verified_gap = int(
            colpack_row[
                "best_colpack5_gap_from_known"
            ]
        )

        if (
            best_colpack5_colors
            - known_target
            != verified_gap
        ):
            raise ValueError(
                f"{graph_id}: inconsistent "
                "ColPack gap."
            )

        if verified_gap != num_components:
            raise ValueError(
                f"{graph_id}: expected heuristic gap "
                f"{num_components}, found "
                f"{verified_gap}."
            )

        graph_family = str(
            row.family
        )

        for node_id in range(
            expected_nodes
        ):
            target_rows.append(
                {
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "order_position": (
                        order_position_by_node[
                            node_id
                        ]
                    ),
                    "target_score": (
                        target_scores[
                            node_id
                        ]
                    ),
                    "known_color": (
                        known_coloring[
                            node_id
                        ]
                    ),
                    "selected_ordering": (
                        SELECTED_ORDERING_NAME
                    ),
                    "selected_num_colors": (
                        selected_num_colors
                    ),
                    "known_chromatic_number": (
                        known_target
                    ),
                    "split": "train",
                    "graph_family": (
                        graph_family
                    ),
                    "num_components_joined": (
                        num_components
                    ),
                    "component_cycle_sizes": (
                        ";".join(
                            str(size)
                            for size
                            in component_sizes
                        )
                    ),
                    "score_convention": (
                        score_convention
                    ),
                }
            )

        summary_rows.append(
            {
                "graph_id": graph_id,
                "split": "train",
                "graph_family": graph_family,
                "num_components_joined": (
                    num_components
                ),
                "component_cycle_sizes": (
                    ";".join(
                        str(size)
                        for size
                        in component_sizes
                    )
                ),
                "num_unique_component_sizes": int(
                    row.num_unique_component_sizes
                ),
                "num_vertices": (
                    expected_nodes
                ),
                "num_edges": (
                    expected_edges
                ),
                "known_chromatic_number": (
                    known_target
                ),
                "best_colpack5_colors": (
                    best_colpack5_colors
                ),
                "verified_gap": (
                    verified_gap
                ),
                "selected_ordering": (
                    SELECTED_ORDERING_NAME
                ),
                "selected_num_colors": (
                    selected_num_colors
                ),
                "target_coloring_valid": (
                    target_coloring_valid
                ),
                "score_convention": (
                    score_convention
                ),
            }
        )

    if len(summary_rows) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} "
            f"summary rows, found "
            f"{len(summary_rows)}."
        )

    if len(score_conventions) != 1:
        raise ValueError(
            "Different target-score conventions "
            f"were detected: {score_conventions}"
        )

    targets_df = pd.DataFrame(
        target_rows
    ).sort_values(
        [
            "graph_id",
            "node_id",
        ]
    ).reset_index(drop=True)

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "num_components_joined",
            "num_vertices",
            "graph_id",
        ]
    ).reset_index(drop=True)

    expected_target_rows = int(
        candidate_df[
            "num_vertices"
        ].sum()
    )

    if len(targets_df) != expected_target_rows:
        raise ValueError(
            f"Expected {expected_target_rows} "
            f"node target rows, found "
            f"{len(targets_df)}."
        )

    if targets_df[
        [
            "graph_id",
            "node_id",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate graph-node target rows found."
        )

    TARGET_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TARGET_SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets_df.to_csv(
        TARGET_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        TARGET_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print(
        "Week 18 controlled mixed exact "
        "targets built successfully."
    )
    print(
        "------------------------------------------"
    )
    print(
        f"Graphs processed: {len(summary_df)}"
    )
    print(
        f"Node target rows: {len(targets_df)}"
    )
    print(
        "All reconstructed exact colorings: valid"
    )
    print(
        "All exact orderings reached known targets: yes"
    )
    print(
        "Target-score convention: "
        f"{next(iter(score_conventions))}"
    )
    print()

    print(
        "Target verification by component count:"
    )

    verification_summary = (
        summary_df
        .groupby(
            "num_components_joined"
        )
        .agg(
            num_graphs=(
                "graph_id",
                "count",
            ),
            total_target_colors=(
                "selected_num_colors",
                "sum",
            ),
            total_colpack_colors=(
                "best_colpack5_colors",
                "sum",
            ),
            mean_verified_gap=(
                "verified_gap",
                "mean",
            ),
            all_valid=(
                "target_coloring_valid",
                "all",
            ),
        )
    )

    print(
        verification_summary.to_string()
    )
    print()

    print(
        f"Saved node targets to: "
        f"{TARGET_OUTPUT_PATH}"
    )
    print(
        f"Saved target summary to: "
        f"{TARGET_SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()