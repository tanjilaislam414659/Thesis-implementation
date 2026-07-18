from __future__ import annotations

import random
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
    / "week18_heterogeneous_generalization"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_candidate_graph_summary.csv"
)


def save_graph_as_mtx(graph: nx.Graph, path: Path) -> None:
    """
    Save an undirected graph as a symmetric Matrix Market adjacency pattern.
    """
    nodes = sorted(graph.nodes())
    node_to_idx = {node: index for index, node in enumerate(nodes)}

    rows: list[int] = []
    cols: list[int] = []

    for u, v in graph.edges():
        if u == v:
            continue

        i = node_to_idx[u]
        j = node_to_idx[v]

        rows.extend([i, j])
        cols.extend([j, i])

    data = [1] * len(rows)

    matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(nodes), len(nodes)),
    )

    mmwrite(path, matrix)


def build_cycle_square_graph(n: int) -> nx.Graph:
    """
    Build C_n^2: vertices at cycle distance 1 or 2 are adjacent.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for node in range(n):
        graph.add_edge(node, (node + 1) % n)
        graph.add_edge(node, (node + 2) % n)

    return graph


def build_crown_graph(partition_size: int) -> nx.Graph:
    """
    Build a crown graph.

    Start from K_{m,m} and remove the perfect matching.
    The graph has 2m vertices and is bipartite.
    """
    graph = nx.Graph()

    left = list(range(partition_size))
    right = list(range(partition_size, 2 * partition_size))

    graph.add_nodes_from(left, bipartite=0)
    graph.add_nodes_from(right, bipartite=1)

    for i in range(partition_size):
        for j in range(partition_size):
            # Remove the matching edge between corresponding vertices.
            if i != j:
                graph.add_edge(left[i], right[j])

    return graph


def randomly_relabel_graph(graph: nx.Graph, seed: int) -> nx.Graph:
    """
    Randomly permute node labels while preserving graph structure.
    """
    rng = random.Random(seed)

    old_nodes = sorted(graph.nodes())
    new_labels = list(range(len(old_nodes)))
    rng.shuffle(new_labels)

    mapping = {
        old_node: new_label
        for old_node, new_label in zip(old_nodes, new_labels)
    }

    return nx.relabel_nodes(graph, mapping, copy=True)


def alternating_crown_relabel(
    graph: nx.Graph,
    partition_size: int,
) -> nx.Graph:
    """
    Relabel crown vertices so left/right vertices alternate:

    left_0, right_0, left_1, right_1, ...
    """
    mapping: dict[int, int] = {}

    for i in range(partition_size):
        left_node = i
        right_node = partition_size + i

        mapping[left_node] = 2 * i
        mapping[right_node] = 2 * i + 1

    return nx.relabel_nodes(graph, mapping, copy=True)


def build_erdos_renyi_graph(
    num_vertices: int,
    probability: float,
    seed: int,
) -> nx.Graph:
    """
    Build a reproducible Erdos-Renyi random graph G(n,p).
    """
    graph = nx.gnp_random_graph(
        n=num_vertices,
        p=probability,
        seed=seed,
        directed=False,
    )

    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def validate_graph(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError("Generated graph has no vertices.")

    if nx.number_of_selfloops(graph) != 0:
        raise ValueError("Generated graph contains self-loops.")


def create_summary_row(
    graph_id: str,
    graph: nx.Graph,
    family: str,
    construction: str,
    labeling: str,
    generation_seed: int | None,
    known_chromatic_number: int | None,
    parameter_1_name: str,
    parameter_1_value: int | float,
    parameter_2_name: str = "",
    parameter_2_value: int | float | str = "",
) -> dict[str, object]:
    matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"

    validate_graph(graph)
    save_graph_as_mtx(graph, matrix_path)

    num_vertices = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    return {
        "graph_id": graph_id,
        "family": family,
        "construction": construction,
        "labeling": labeling,
        "generation_seed": generation_seed,
        "parameter_1_name": parameter_1_name,
        "parameter_1_value": parameter_1_value,
        "parameter_2_name": parameter_2_name,
        "parameter_2_value": parameter_2_value,
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "density": nx.density(graph),
        "num_components": nx.number_connected_components(graph),
        "known_chromatic_number": known_chromatic_number,
        "matrix_path": str(matrix_path.relative_to(PROJECT_ROOT)),
    }


def generate_cycle_square_candidates() -> list[dict[str, object]]:
    rows = []

    # All sizes satisfy n = 3r + 2 and therefore have a known 4-coloring.
    cycle_sizes = [
        17, 20, 23, 26, 29,
        32, 35, 38, 41, 44,
        47, 50, 53, 56, 59,
    ]

    for n in cycle_sizes:
        base_graph = build_cycle_square_graph(n)

        variants = [
            ("identity", base_graph, None),
            ("random_0", randomly_relabel_graph(base_graph, seed=1000 + n), 1000 + n),
            ("random_1", randomly_relabel_graph(base_graph, seed=2000 + n), 2000 + n),
        ]

        for labeling, graph, seed in variants:
            graph_id = f"week18_cycle_square_c{n}_{labeling}"

            rows.append(
                create_summary_row(
                    graph_id=graph_id,
                    graph=graph,
                    family="cycle_square",
                    construction=f"C_{n}^2",
                    labeling=labeling,
                    generation_seed=seed,
                    known_chromatic_number=4,
                    parameter_1_name="cycle_size",
                    parameter_1_value=n,
                )
            )

    return rows


def generate_crown_candidates() -> list[dict[str, object]]:
    rows = []

    partition_sizes = list(range(10, 31, 2))

    for partition_size in partition_sizes:
        base_graph = build_crown_graph(partition_size)

        variants = [
            ("blocked", base_graph, None),
            (
                "alternating",
                alternating_crown_relabel(base_graph, partition_size),
                None,
            ),
            (
                "random",
                randomly_relabel_graph(
                    base_graph,
                    seed=3000 + partition_size,
                ),
                3000 + partition_size,
            ),
        ]

        for labeling, graph, seed in variants:
            graph_id = (
                f"week18_crown_m{partition_size}_{labeling}"
            )

            rows.append(
                create_summary_row(
                    graph_id=graph_id,
                    graph=graph,
                    family="crown",
                    construction=(
                        f"K_{{{partition_size},{partition_size}}} "
                        "minus perfect matching"
                    ),
                    labeling=labeling,
                    generation_seed=seed,
                    known_chromatic_number=2,
                    parameter_1_name="partition_size",
                    parameter_1_value=partition_size,
                )
            )

    return rows


def generate_random_candidates() -> list[dict[str, object]]:
    rows = []

    vertex_counts = [40, 60, 80, 100, 120]
    probabilities = [0.06, 0.09, 0.12, 0.16, 0.22, 0.30]
    replicate_seeds = [0, 1, 2, 3]

    for num_vertices in vertex_counts:
        for probability in probabilities:
            probability_code = int(round(probability * 100))

            for replicate in replicate_seeds:
                seed = (
                    100_000
                    + num_vertices * 100
                    + probability_code * 10
                    + replicate
                )

                graph = build_erdos_renyi_graph(
                    num_vertices=num_vertices,
                    probability=probability,
                    seed=seed,
                )

                graph_id = (
                    f"week18_er_n{num_vertices}"
                    f"_p{probability_code:02d}"
                    f"_s{replicate}"
                )

                rows.append(
                    create_summary_row(
                        graph_id=graph_id,
                        graph=graph,
                        family="erdos_renyi",
                        construction=(
                            f"G({num_vertices}, {probability:.2f})"
                        ),
                        labeling="generated",
                        generation_seed=seed,
                        known_chromatic_number=None,
                        parameter_1_name="num_vertices",
                        parameter_1_value=num_vertices,
                        parameter_2_name="edge_probability",
                        parameter_2_value=probability,
                    )
                )

    return rows


def main() -> None:
    OUTPUT_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    summary_rows.extend(generate_cycle_square_candidates())
    summary_rows.extend(generate_crown_candidates())
    summary_rows.extend(generate_random_candidates())

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("Generated Week 18 heterogeneous candidate graph pool.")
    print(f"Saved matrices to: {OUTPUT_MATRIX_DIR}")
    print(f"Saved metadata to: {OUTPUT_SUMMARY_PATH}")
    print()

    print("Candidate count by family:")
    print(summary_df.groupby("family").size().to_string())
    print()

    print(f"Total candidate graphs: {len(summary_df)}")
    print(
        f"Vertex range: "
        f"{summary_df['num_vertices'].min()}–"
        f"{summary_df['num_vertices'].max()}"
    )
    print(
        f"Edge range: "
        f"{summary_df['num_edges'].min()}–"
        f"{summary_df['num_edges'].max()}"
    )


if __name__ == "__main__":
    main()