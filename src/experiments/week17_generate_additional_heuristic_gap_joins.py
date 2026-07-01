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

    # Use only cycle-square sizes that were verified as hard against best-of-5 ColPack,
    # excluding C23 because DYNAMIC_LARGEST_FIRST already reached the 4-color optimum there.
    hard_base_sizes = [26, 29, 32, 35, 38, 41]
    copy_counts = [2, 3]

    for base_n in hard_base_sizes:
        for copies in copy_counts:
            graph_id = f"week17_gap_join_c{base_n}_join_{copies}"

            if graph_id in existing_graph_ids:
                print(f"Skipping existing graph: {graph_id}")
                continue

            base_graphs = []
            base_colorings = []

            for _ in range(copies):
                base_graphs.append(build_cycle_square_graph(base_n))
                base_colorings.append(exact_coloring_cycle_square_3r_plus_2(base_n))

            joined_graph, joined_coloring = join_graphs_with_colorings(
                base_graphs=base_graphs,
                base_colorings=base_colorings,
            )

            is_valid = validate_coloring(joined_graph, joined_coloring)
            known_colors = max(joined_coloring.values()) + 1

            matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"
            save_graph_as_mtx(joined_graph, matrix_path)

            new_rows.append(
                {
                    "graph_id": graph_id,
                    "family": "join_of_cycle_square_hard_cases",
                    "construction": f"join of {copies} copies of C_{base_n}^2",
                    "num_components_joined": copies,
                    "base_cycle_size": base_n,
                    "num_vertices": joined_graph.number_of_nodes(),
                    "num_edges": joined_graph.number_of_edges(),
                    "known_chromatic_number": known_colors,
                    "known_coloring_valid": is_valid,
                    "matrix_path": str(matrix_path.relative_to(PROJECT_ROOT)),
                }
            )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([summary_df, new_df], ignore_index=True)
        combined_df.to_csv(SUMMARY_PATH, index=False)

        print("Generated additional heuristic-gap join graphs.")
        print()
        print(new_df[[
            "graph_id",
            "base_cycle_size",
            "num_components_joined",
            "num_vertices",
            "num_edges",
            "known_chromatic_number",
            "known_coloring_valid",
        ]].to_string(index=False))
    else:
        print("No new graphs were generated. All requested graphs already exist.")

    print()
    print(f"Updated summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()