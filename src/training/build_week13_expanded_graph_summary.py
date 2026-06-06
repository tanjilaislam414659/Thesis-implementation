from pathlib import Path

import pandas as pd
from scipy.io import mmread

from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx


rows = []
folder = Path("data/raw/matrices")

for p in sorted(folder.glob("*.mtx")):
    matrix = mmread(p)
    graph = load_graph_from_mtx(p)

    rows.append(
        {
            "graph_id": p.stem,
            "matrix_file": p.name,
            "matrix_rows": matrix.shape[0],
            "matrix_cols": matrix.shape[1],
            "matrix_nnz": matrix.nnz if hasattr(matrix, "nnz") else None,
            "graph_vertices": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
        }
    )

df = pd.DataFrame(rows)

out = Path(
    "data/processed/initial_graph_coloring_dataset/graph_metadata/"
    "week13_expanded_graph_summary.csv"
)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(df.to_string(index=False))
print(f"\nSaved to {out}")