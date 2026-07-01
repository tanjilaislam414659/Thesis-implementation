from pathlib import Path
import pandas as pd
import networkx as nx
from scipy.io import mmwrite
from scipy.sparse import coo_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week17_heuristic_gap_extension"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_heuristic_gap_graph_family_summary.csv"
)


EXTRA_TEST_BASE_SIZES = [44, 47]
GAP_LEVELS = [1, 2, 3, 4, 5]


def build_cycle_square_graph(n: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        graph.add_edge(i, (i + 1) % n)
        graph.add_edge(i, (i + 2) % n)

    return graph


def exact_coloring_cycle_square_3r_plus_2(n: int) -> dict[int, int]:
    if (n - 2) % 3 != 0:
        raise ValueError(f"Expected n = 3r + 2, but got n={n}")

    colors = []

    for i in range(n - 5):
        colors.append(i % 3)

    colors.extend([0, 3, 1, 2, 3])

    return {node: colors[node] for node in range(n)}


def validate_coloring(graph: nx.Graph, coloring: dict[int, int]) -> bool:
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False
    return True


def save_graph_as_mtx(graph: nx.Graph, path: Path) -> None:
    nodes = sorted(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    rows = []
    cols = []

    for u, v in graph.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]

        rows.extend([i, j])
        cols.extend([j, i])

    data = [1] * len(rows)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))

    mmwrite(path, matrix)


def join_graphs_with_colorings(
    base_graphs: list[nx.Graph],
    base_colorings: list[dict[int, int]],
    color_offset_step: int = 4,
) -> tuple[nx.Graph, dict[int, int]]:
    joined = nx.Graph()
    joined_coloring = {}

    component_nodes = []
    next_node = 0

    for component_id, graph in enumerate(base_graphs):
        mapping = {old_node: next_node + old_node for old_node in graph.nodes()}
        relabeled = nx.relabel_nodes(graph, mapping)

        joined.add_nodes_from(relabeled.nodes())
        joined.add_edges_from(relabeled.edges())

        new_nodes = sorted(relabeled.nodes())
        component_nodes.append(new_nodes)

        for old_node, old_color in base_colorings[component_id].items():
            new_node = mapping[old_node]
            joined_coloring[new_node] = old_color + component_id * color_offset_step

        next_node += graph.number_of_nodes()

    for i in range(len(component_nodes)):
        for j in range(i + 1, len(component_nodes)):
            for u in component_nodes[i]:
                for v in component_nodes[j]:
                    joined.add_edge(u, v)

    return joined, joined_coloring


def build_graph_and_coloring(base_n: int, gap_level: int) -> tuple[nx.Graph, dict[int, int], str]:
    if gap_level == 1:
        graph = build_cycle_square_graph(base_n)
        coloring = exact_coloring_cycle_square_3r_plus_2(base_n)
        graph_id = f"week17_gap_cycle_square_c{base_n}"
        return graph, coloring, graph_id

    base_graphs = []
    base_colorings = []

    for _ in range(gap_level):
        base_graphs.append(build_cycle_square_graph(base_n))
        base_colorings.append(exact_coloring_cycle_square_3r_plus_2(base_n))

    graph, coloring = join_graphs_with_colorings(
        base_graphs=base_graphs,
        base_colorings=base_colorings,
    )

    graph_id = f"week17_gap_join_c{base_n}_join_{gap_level}"
    return graph, coloring, graph_id


def main() -> None:
    OUTPUT_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SUMMARY_PATH.exists():
        summary_df = pd.read_csv(SUMMARY_PATH)
        existing_graph_ids = set(summary_df["graph_id"].tolist())
    else:
        summary_df = pd.DataFrame()
        existing_graph_ids = set()

    new_rows = []

    for base_n in EXTRA_TEST_BASE_SIZES:
        for gap_level in GAP_LEVELS:
            graph, coloring, graph_id = build_graph_and_coloring(
                base_n=base_n,
                gap_level=gap_level,
            )

            if graph_id in existing_graph_ids:
                print(f"Skipping existing graph: {graph_id}")
                continue

            is_valid = validate_coloring(graph, coloring)
            known_colors = max(coloring.values()) + 1

            matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"
            save_graph_as_mtx(graph, matrix_path)

            if gap_level == 1:
                family = "cycle_square_3r_plus_2"
                construction = f"C_{base_n}^2"
                num_components_joined = 1
            else:
                family = "join_of_cycle_square_hard_cases"
                construction = f"join of {gap_level} copies of C_{base_n}^2"
                num_components_joined = gap_level

            new_rows.append(
                {
                    "graph_id": graph_id,
                    "family": family,
                    "construction": construction,
                    "num_components_joined": num_components_joined,
                    "base_cycle_size": base_n,
                    "num_vertices": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "known_chromatic_number": known_colors,
                    "known_coloring_valid": is_valid,
                    "matrix_path": str(matrix_path.relative_to(PROJECT_ROOT)),
                }
            )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([summary_df, new_df], ignore_index=True)
        combined_df.to_csv(SUMMARY_PATH, index=False)

        print("Generated extra unseen test heuristic-gap graphs.")
        print()
        print(
            new_df[
                [
                    "graph_id",
                    "base_cycle_size",
                    "num_components_joined",
                    "num_vertices",
                    "num_edges",
                    "known_chromatic_number",
                    "known_coloring_valid",
                ]
            ].to_string(index=False)
        )
    else:
        print("No new extra test graphs were generated. They may already exist.")

    print()
    print(f"Updated summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()