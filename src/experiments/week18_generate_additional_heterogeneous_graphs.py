from __future__ import annotations

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
    / "week18_additional_heterogeneous_candidate_graph_summary.csv"
)


def save_graph_as_mtx(graph: nx.Graph, path: Path) -> None:
    """
    Save an undirected graph as a symmetric Matrix Market adjacency matrix.
    """
    nodes = sorted(graph.nodes())
    node_to_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    rows: list[int] = []
    cols: list[int] = []

    for u, v in graph.edges():
        if u == v:
            continue

        i = node_to_index[u]
        j = node_to_index[v]

        rows.extend([i, j])
        cols.extend([j, i])

    data = [1] * len(rows)

    matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(nodes), len(nodes)),
    )

    mmwrite(path, matrix)


def validate_graph(graph: nx.Graph, graph_id: str) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{graph_id}: graph has no vertices.")

    if nx.number_of_selfloops(graph) != 0:
        raise ValueError(f"{graph_id}: graph contains self-loops.")


def save_candidate(
    graph_id: str,
    graph: nx.Graph,
    family: str,
    construction: str,
    generation_seed: int,
    parameters: dict[str, object],
) -> dict[str, object]:
    validate_graph(graph, graph_id)

    matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"
    save_graph_as_mtx(graph, matrix_path)

    degrees = [
        degree
        for _, degree in graph.degree()
    ]

    return {
        "graph_id": graph_id,
        "family": family,
        "construction": construction,
        "generation_seed": generation_seed,
        "num_vertices": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "num_components": nx.number_connected_components(graph),
        "minimum_degree": min(degrees),
        "maximum_degree": max(degrees),
        "average_degree": sum(degrees) / len(degrees),
        "parameters": "; ".join(
            f"{key}={value}"
            for key, value in parameters.items()
        ),
        "matrix_path": str(
            matrix_path.relative_to(PROJECT_ROOT)
        ),
    }


def generate_barabasi_albert_candidates() -> list[dict[str, object]]:
    rows = []

    vertex_counts = [60, 80, 100, 120]
    attachment_counts = [2, 4, 6]
    replicates = [0, 1, 2]

    for n in vertex_counts:
        for m in attachment_counts:
            for replicate in replicates:
                seed = 200_000 + n * 100 + m * 10 + replicate

                graph = nx.barabasi_albert_graph(
                    n=n,
                    m=m,
                    seed=seed,
                )

                graph_id = (
                    f"week18_ba_n{n}_m{m}_s{replicate}"
                )

                rows.append(
                    save_candidate(
                        graph_id=graph_id,
                        graph=graph,
                        family="barabasi_albert",
                        construction=(
                            f"Barabasi-Albert graph with n={n}, m={m}"
                        ),
                        generation_seed=seed,
                        parameters={
                            "n": n,
                            "m": m,
                            "replicate": replicate,
                        },
                    )
                )

    return rows


def generate_watts_strogatz_candidates() -> list[dict[str, object]]:
    rows = []

    vertex_counts = [60, 80, 100, 120]
    neighbour_counts = [4, 8]
    rewiring_probabilities = [0.05, 0.20, 0.50]
    replicates = [0, 1]

    for n in vertex_counts:
        for k in neighbour_counts:
            for probability in rewiring_probabilities:
                probability_code = int(round(probability * 100))

                for replicate in replicates:
                    seed = (
                        300_000
                        + n * 100
                        + k * 10
                        + probability_code
                        + replicate
                    )

                    graph = nx.watts_strogatz_graph(
                        n=n,
                        k=k,
                        p=probability,
                        seed=seed,
                    )

                    graph_id = (
                        f"week18_ws_n{n}"
                        f"_k{k}"
                        f"_p{probability_code:02d}"
                        f"_s{replicate}"
                    )

                    rows.append(
                        save_candidate(
                            graph_id=graph_id,
                            graph=graph,
                            family="watts_strogatz",
                            construction=(
                                f"Watts-Strogatz graph with "
                                f"n={n}, k={k}, p={probability}"
                            ),
                            generation_seed=seed,
                            parameters={
                                "n": n,
                                "k": k,
                                "rewiring_probability": probability,
                                "replicate": replicate,
                            },
                        )
                    )

    return rows


def build_probability_matrix(
    number_of_blocks: int,
    within_probability: float,
    between_probability: float,
) -> list[list[float]]:
    return [
        [
            (
                within_probability
                if row == column
                else between_probability
            )
            for column in range(number_of_blocks)
        ]
        for row in range(number_of_blocks)
    ]


def generate_stochastic_block_candidates() -> list[dict[str, object]]:
    rows = []

    block_specs = [
        ("two_blocks_60", [30, 30]),
        ("two_blocks_80", [40, 40]),
        ("three_blocks_90", [30, 30, 30]),
        ("three_blocks_120", [40, 40, 40]),
    ]

    probability_profiles = [
        ("strong", 0.35, 0.02),
        ("moderate", 0.25, 0.05),
        ("weak", 0.18, 0.10),
    ]

    replicates = [0, 1]

    for specification_name, block_sizes in block_specs:
        number_of_blocks = len(block_sizes)
        total_vertices = sum(block_sizes)

        for profile_name, p_in, p_out in probability_profiles:
            probability_matrix = build_probability_matrix(
                number_of_blocks=number_of_blocks,
                within_probability=p_in,
                between_probability=p_out,
            )

            for replicate in replicates:
                seed = (
                    400_000
                    + total_vertices * 100
                    + number_of_blocks * 10
                    + replicate
                    + int(p_in * 1000)
                    + int(p_out * 100)
                )

                graph = nx.stochastic_block_model(
                    sizes=block_sizes,
                    p=probability_matrix,
                    seed=seed,
                    directed=False,
                    selfloops=False,
                )

                graph_id = (
                    f"week18_sbm_{specification_name}"
                    f"_{profile_name}"
                    f"_s{replicate}"
                )

                rows.append(
                    save_candidate(
                        graph_id=graph_id,
                        graph=graph,
                        family="stochastic_block_model",
                        construction=(
                            f"SBM blocks={block_sizes}, "
                            f"p_in={p_in}, p_out={p_out}"
                        ),
                        generation_seed=seed,
                        parameters={
                            "block_sizes": block_sizes,
                            "profile": profile_name,
                            "p_in": p_in,
                            "p_out": p_out,
                            "replicate": replicate,
                        },
                    )
                )

    return rows


def main() -> None:
    OUTPUT_MATRIX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    rows.extend(generate_barabasi_albert_candidates())
    rows.extend(generate_watts_strogatz_candidates())
    rows.extend(generate_stochastic_block_candidates())

    summary_df = pd.DataFrame(rows)

    if summary_df["graph_id"].duplicated().any():
        duplicates = summary_df.loc[
            summary_df["graph_id"].duplicated(),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph IDs generated: {duplicates}"
        )

    summary_df.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    print("Generated additional Week 18 heterogeneous candidates.")
    print("------------------------------------------------------")
    print(f"Total additional candidates: {len(summary_df)}")
    print()

    print("Candidate count by family:")
    print(
        summary_df
        .groupby("family")
        .size()
        .to_string()
    )
    print()

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

    print(
        f"Disconnected candidates: "
        f"{int((summary_df['num_components'] > 1).sum())}"
    )

    print()
    print(f"Saved matrices to: {OUTPUT_MATRIX_DIR}")
    print(f"Saved metadata to: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()