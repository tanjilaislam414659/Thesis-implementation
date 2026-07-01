from pathlib import Path
import pandas as pd
import networkx as nx
from scipy.io import mmwrite
from scipy.sparse import coo_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_MATRIX_DIR = PROJECT_ROOT / "data" / "raw" / "matrices" / "week17_heuristic_gap_extension"
OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_heuristic_gap_graph_family_summary.csv"
)


def build_cycle_square_graph(n: int) -> nx.Graph:
    """
    Build C_n^2, the square of a cycle on n vertices.
    Two vertices are adjacent if their cycle distance is 1 or 2.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        graph.add_edge(i, (i + 1) % n)
        graph.add_edge(i, (i + 2) % n)

    return graph


def exact_coloring_cycle_square_3r_plus_2(n: int) -> dict[int, int]:
    """
    Construct a known 4-coloring for C_n^2 where n = 3r + 2.

    Pattern:
    repeat 0,1,2 for n-5 vertices,
    then append 0,3,1,2,3.
    """
    if (n - 2) % 3 != 0:
        raise ValueError(f"Expected n = 3r + 2, but got n={n}")

    prefix_length = n - 5
    colors = []

    for i in range(prefix_length):
        colors.append(i % 3)

    colors.extend([0, 3, 1, 2, 3])

    return {node: colors[node] for node in range(n)}


def validate_coloring(graph: nx.Graph, coloring: dict[int, int]) -> bool:
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False
    return True


def save_graph_as_mtx(graph: nx.Graph, path: Path) -> None:
    """
    Save graph adjacency pattern as Matrix Market file.
    Self-loops are not added.
    """
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
    """
    Build the join of several graphs.

    In the join, every vertex of one component is connected to every vertex
    of every other component.

    Since all components are fully connected to each other, their color sets
    must be disjoint. Therefore, each component coloring receives a color offset.
    """
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

    # Add complete connections between different components.
    for i in range(len(component_nodes)):
        for j in range(i + 1, len(component_nodes)):
            for u in component_nodes[i]:
                for v in component_nodes[j]:
                    joined.add_edge(u, v)

    return joined, joined_coloring


def main() -> None:
    OUTPUT_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # Part 1: expanded Bickle cycle-square candidates.
    # All n here satisfy n = 3r + 2, so known chromatic number is 4.
    cycle_sizes = [17, 20, 23, 26, 29, 32, 35, 38, 41]

    for n in cycle_sizes:
        graph = build_cycle_square_graph(n)
        coloring = exact_coloring_cycle_square_3r_plus_2(n)

        is_valid = validate_coloring(graph, coloring)
        known_colors = max(coloring.values()) + 1

        graph_id = f"week17_gap_cycle_square_c{n}"
        matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"
        save_graph_as_mtx(graph, matrix_path)

        summary_rows.append(
            {
                "graph_id": graph_id,
                "family": "cycle_square_3r_plus_2",
                "construction": f"C_{n}^2",
                "num_components_joined": 1,
                "base_cycle_size": n,
                "num_vertices": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "known_chromatic_number": known_colors,
                "known_coloring_valid": is_valid,
                "matrix_path": str(matrix_path.relative_to(PROJECT_ROOT)),
            }
        )

    # Part 2: join-of-Bickle candidates.
    # We start with C17^2 and C20^2 because these are confirmed hard cases
    # where the tested ColPack orderings miss the known 4-color result.
    join_specs = [
        ("c17_join_2", 17, 2),
        ("c17_join_3", 17, 3),
        ("c20_join_2", 20, 2),
        ("c20_join_3", 20, 3),
    ]

    for name, base_n, copies in join_specs:
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

        graph_id = f"week17_gap_join_{name}"
        matrix_path = OUTPUT_MATRIX_DIR / f"{graph_id}.mtx"
        save_graph_as_mtx(joined_graph, matrix_path)

        summary_rows.append(
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

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("Generated Week 17 heuristic-gap graph candidates.")
    print(f"Saved matrices to: {OUTPUT_MATRIX_DIR}")
    print(f"Saved summary to: {OUTPUT_SUMMARY_PATH}")
    print()
    print(summary_df[[
        "graph_id",
        "family",
        "num_vertices",
        "num_edges",
        "known_chromatic_number",
        "known_coloring_valid",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()