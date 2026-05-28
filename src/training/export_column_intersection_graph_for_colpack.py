"""
Export the Python column-intersection graph of a rectangular matrix
as a square Matrix Market adjacency-pattern file for ColPack.

This is needed for jac_pat.mtx, where:
- the Python/PyG pipeline uses a column intersection graph,
- the earlier ColPack run interpreted the rectangular matrix differently.

The exported file allows ColPack to process the same 43-vertex,
121-edge graph used by the Python/PyG pipeline.
"""

from __future__ import annotations

from pathlib import Path

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx


INPUT_MATRIX = Path("data/raw/matrices/jac_pat.mtx")

OUTPUT_MATRIX = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_graph_inputs/jac_pat_column_intersection_graph.mtx"
)


def export_graph_as_matrix_market_adjacency() -> None:
    """
    Export the jac_pat column-intersection graph as a square
    Matrix Market coordinate file.

    One entry is written for each undirected edge (u, v), using
    1-based indexing as required by Matrix Market format.
    """

    graph = load_graph_from_mtx(INPUT_MATRIX)

    num_nodes = graph.number_of_nodes()
    edges = sorted(graph.edges())

    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_MATRIX.open("w", encoding="utf-8") as file:
        file.write("%%MatrixMarket matrix coordinate integer general\n")
        file.write("% Column intersection graph exported for ColPack alignment\n")
        file.write(f"{num_nodes} {num_nodes} {len(edges)}\n")

        for u, v in edges:
            # Matrix Market uses 1-based indexing.
            file.write(f"{u + 1} {v + 1} 1\n")

    print("Exported column-intersection graph for ColPack")
    print("------------------------------------------------")
    print(f"Input matrix: {INPUT_MATRIX}")
    print(f"Output graph file: {OUTPUT_MATRIX}")
    print(f"Vertices: {num_nodes}")
    print(f"Undirected edges written: {len(edges)}")


def main() -> None:
    export_graph_as_matrix_market_adjacency()


if __name__ == "__main__":
    main()