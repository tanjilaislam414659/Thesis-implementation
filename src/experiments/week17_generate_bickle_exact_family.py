from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

from src.training.ordered_greedy_coloring import (
    count_colors,
    greedy_color_with_ordering,
    is_valid_coloring,
)


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

OUTPUT_MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week17_bickle_exact_family"
)

TARGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "ordering_targets"
    / "week17_bickle_exact_optimal_ordering_targets.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_bickle_exact_split.csv"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_bickle_exact_family_summary.csv"
)


CYCLE_SQUARE_SIZES = [8, 11, 14, 17, 20, 23, 26, 29, 32]


SPLIT_BY_SIZE = {
    8: "train",
    11: "train",
    14: "train",
    23: "train",
    26: "train",
    29: "train",
    32: "validation",
    17: "test",
    20: "test",
}


def make_cycle_square_graph(n: int) -> nx.Graph:
    """
    Build the square of the cycle C_n.

    Vertices i and j are adjacent if their circular distance is 1 or 2.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for node in range(n):
        graph.add_edge(node, (node + 1) % n)
        graph.add_edge(node, (node + 2) % n)

    return graph


def exact_four_coloring_for_cycle_square_3r_plus_2(n: int) -> dict[int, int]:
    """
    Construct a known 4-coloring for C_n^2 when n = 3r + 2.

    Pattern:
        repeat 0,1,2 for n-5 positions,
        then append 0,3,1,2,3.

    This gives a valid 4-coloring for the cycle-square graphs used here.
    """

    if n % 3 != 2:
        raise ValueError(f"This construction expects n = 3r + 2, got n={n}.")

    if n < 8:
        raise ValueError("This construction is intended for n >= 8.")

    prefix_length = n - 5

    if prefix_length % 3 != 0:
        raise ValueError(
            f"Internal construction error: prefix length {prefix_length} "
            "is not divisible by 3."
        )

    color_sequence = [0, 1, 2] * (prefix_length // 3)
    color_sequence += [0, 3, 1, 2, 3]

    if len(color_sequence) != n:
        raise ValueError(
            f"Expected color sequence length {n}, got {len(color_sequence)}."
        )

    return {node: int(color_sequence[node]) for node in range(n)}


def validate_coloring(graph: nx.Graph, coloring: dict[int, int]) -> None:
    for source, target in graph.edges():
        if coloring[source] == coloring[target]:
            raise ValueError(
                f"Invalid coloring: edge ({source}, {target}) has same color "
                f"{coloring[source]}."
            )


def ordering_from_color_classes(coloring: dict[int, int]) -> list[int]:
    """
    Convert an exact coloring into a deterministic greedy-compatible ordering.

    Vertices are ordered by color class, then by node id.
    Greedy coloring with this ordering should use the exact number of colors.
    """

    return [
        node
        for node, _color in sorted(
            coloring.items(),
            key=lambda item: (item[1], item[0]),
        )
    ]


def write_graph_as_matrix_market(graph: nx.Graph, output_path: Path) -> None:
    """
    Save an undirected graph as a symmetric adjacency-pattern Matrix Market file.
    """

    rows = []
    cols = []

    for source, target in sorted(graph.edges()):
        rows.append(source)
        cols.append(target)

        rows.append(target)
        cols.append(source)

    values = [1] * len(rows)

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(graph.number_of_nodes(), graph.number_of_nodes()),
        dtype=int,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mmwrite(output_path, matrix)


def normalized_target_score(order_position: int, num_nodes: int) -> float:
    """
    Convert an ordering position into a supervised node score.

    Earlier positions receive larger target scores.
    """

    if num_nodes <= 1:
        return 1.0

    return 1.0 - (order_position / (num_nodes - 1))


def main() -> None:
    OUTPUT_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    target_rows = []
    split_rows = []

    for n in CYCLE_SQUARE_SIZES:
        graph_id = f"week17_cycle_square_c{n}"
        split = SPLIT_BY_SIZE[n]

        graph = make_cycle_square_graph(n)

        exact_coloring = exact_four_coloring_for_cycle_square_3r_plus_2(n)
        validate_coloring(graph, exact_coloring)

        exact_ordering = ordering_from_color_classes(exact_coloring)

        greedy_coloring = greedy_color_with_ordering(
            graph=graph,
            ordering=exact_ordering,
        )

        exact_greedy_colors = count_colors(greedy_coloring)
        exact_greedy_valid = is_valid_coloring(graph, greedy_coloring)

        if exact_greedy_colors != 4:
            raise ValueError(
                f"{graph_id}: expected exact greedy coloring with 4 colors, "
                f"got {exact_greedy_colors}."
            )

        if not exact_greedy_valid:
            raise ValueError(f"{graph_id}: exact greedy coloring is invalid.")

        matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"
        write_graph_as_matrix_market(graph, matrix_path)

        num_vertices = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        summary_rows.append(
            {
                "graph_id": graph_id,
                "cycle_size": n,
                "split": split,
                "graph_family": "cycle_square_3r_plus_2",
                "num_vertices": num_vertices,
                "num_edges": num_edges,
                "matrix_nnz": 2 * num_edges,
                "known_chromatic_number": 4,
                "exact_target_ordering": "EXACT_OPTIMAL_COLOR_CLASS_ORDER",
                "exact_greedy_colors": exact_greedy_colors,
                "exact_greedy_valid": exact_greedy_valid,
                "matrix_path": str(matrix_path),
            }
        )

        split_rows.append(
            {
                "graph_id": graph_id,
                "split": split,
                "group": "bickle_cycle_square_exact",
                "reason": (
                    "held_out_test_hard_case"
                    if split == "test"
                    else "exact_optimal_bickle_training_family"
                    if split == "train"
                    else "exact_optimal_bickle_validation_family"
                ),
            }
        )

        for order_position, node_id in enumerate(exact_ordering):
            target_rows.append(
                {
                    "graph_id": graph_id,
                    "node_id": int(node_id),
                    "order_position": int(order_position),
                    "target_score": normalized_target_score(
                        order_position=order_position,
                        num_nodes=num_vertices,
                    ),
                    "selected_ordering": "EXACT_OPTIMAL_COLOR_CLASS_ORDER",
                    "selected_num_colors": 4,
                    "known_chromatic_number": 4,
                    "split": split,
                    "graph_family": "cycle_square_3r_plus_2",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    target_df = pd.DataFrame(target_rows)
    split_df = pd.DataFrame(split_rows)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    target_df.to_csv(TARGET_CSV, index=False)
    split_df.to_csv(SPLIT_CSV, index=False)

    print("Week 17 exact-optimal Bickle cycle-square family generated.")
    print()
    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Split counts:")
    print(split_df["split"].value_counts())
    print()
    print("Target rows:", len(target_df))
    print()
    print(f"Saved matrices to: {OUTPUT_MATRIX_DIR}")
    print(f"Saved exact target CSV to: {TARGET_CSV}")
    print(f"Saved split CSV to: {SPLIT_CSV}")
    print(f"Saved summary CSV to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()