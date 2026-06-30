from pathlib import Path
import csv

import scipy.io
import scipy.sparse as sp
import networkx as nx


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

MATRIX_DIR = PROJECT_ROOT / "data" / "raw" / "matrices"

ARROWHEAD_PATH = MATRIX_DIR / "week17_arrowhead_100.mtx"

METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_structured_matrix_metadata.csv"
)


STRUCTURED_MATRICES = [
    {
        "graph_id": "week17_nos1",
        "matrix_file": "nos1.mtx",
        "source_type": "real_matrix",
        "source_name": "HB/nos1",
        "structure_family": "banded",
        "description": "Harwell-Boeing finite-element beam matrix; narrow banded structure",
    },
    {
        "graph_id": "week17_gr_30_30",
        "matrix_file": "gr_30_30.mtx",
        "source_type": "real_matrix",
        "source_name": "HB/gr_30_30",
        "structure_family": "pde_grid_banded",
        "description": "Grid/stencil-like structured matrix",
    },
    {
        "graph_id": "week17_bwm200",
        "matrix_file": "bwm200.mtx",
        "source_type": "real_matrix",
        "source_name": "Bai/bwm200",
        "structure_family": "narrow_banded",
        "description": "Small chemical-process style matrix; narrow structured sparsity",
    },
    {
        "graph_id": "week17_bcsstk08",
        "matrix_file": "bcsstk08.mtx",
        "source_type": "real_matrix",
        "source_name": "HB/bcsstk08",
        "structure_family": "block_structured",
        "description": "Structural stiffness matrix extending the bcsstk family",
    },
    {
        "graph_id": "week17_lshp_265",
        "matrix_file": "lshp_265.mtx",
        "source_type": "real_matrix",
        "source_name": "HB/lshp_265",
        "structure_family": "finite_element_block_like",
        "description": "L-shaped finite-element style matrix",
    },
    {
        "graph_id": "week17_arrowhead_100",
        "matrix_file": "week17_arrowhead_100.mtx",
        "source_type": "constructed",
        "source_name": "synthetic_arrowhead",
        "structure_family": "arrowhead",
        "description": "Constructed diagonal-plus-border arrowhead sparsity pattern",
    },
]


def create_arrowhead_matrix(n: int = 100) -> None:
    """
    Create a pure arrowhead sparsity pattern:
    diagonal entries plus dense last row/last column.
    """
    rows = []
    cols = []
    data = []

    # Diagonal
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(1)

    # Last row and last column border
    hub = n - 1
    for i in range(n - 1):
        rows.append(i)
        cols.append(hub)
        data.append(1)

        rows.append(hub)
        cols.append(i)
        data.append(1)

    matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    scipy.io.mmwrite(ARROWHEAD_PATH, matrix)

    print(f"Created synthetic arrowhead matrix: {ARROWHEAD_PATH}")


def matrix_to_graph_stats(matrix_path: Path):
    matrix = scipy.io.mmread(matrix_path).tocoo()

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: {matrix_path}")

    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))

    for i, j in zip(matrix.row, matrix.col):
        i = int(i)
        j = int(j)
        if i != j:
            graph.add_edge(i, j)

    return {
        "matrix_rows": matrix.shape[0],
        "matrix_cols": matrix.shape[1],
        "matrix_nnz": matrix.nnz,
        "graph_vertices": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }


def main() -> None:
    create_arrowhead_matrix(n=100)

    rows = []

    for item in STRUCTURED_MATRICES:
        matrix_path = MATRIX_DIR / item["matrix_file"]

        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing matrix file: {matrix_path}")

        stats = matrix_to_graph_stats(matrix_path)

        rows.append(
            {
                "graph_id": item["graph_id"],
                "matrix_file": item["matrix_file"],
                "source_type": item["source_type"],
                "source_name": item["source_name"],
                "structure_family": item["structure_family"],
                "matrix_rows": stats["matrix_rows"],
                "matrix_cols": stats["matrix_cols"],
                "matrix_nnz": stats["matrix_nnz"],
                "graph_vertices": stats["graph_vertices"],
                "graph_edges": stats["graph_edges"],
                "description": item["description"],
            }
        )

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Structured matrix metadata:")
    print()

    for row in rows:
        print(
            f"{row['graph_id']}: "
            f"family={row['structure_family']}, "
            f"matrix={row['matrix_rows']}x{row['matrix_cols']}, "
            f"nnz={row['matrix_nnz']}, "
            f"graph_vertices={row['graph_vertices']}, "
            f"graph_edges={row['graph_edges']}"
        )

    print()
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()