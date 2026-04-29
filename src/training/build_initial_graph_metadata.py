import csv

from pathlib import Path


from src.graphs.sparse_matrix_to_graph import load_graph_from_mtx


MATRIX_DIR = Path("data/raw/matrices")

OUTPUT_CSV = Path("data/processed/initial_graph_coloring_dataset/graph_metadata/graph_metadata.csv")


def build_metadata_row(path: Path) -> dict:
    """Build one graph metadata row from a Matrix Market file."""
    try:
        graph = load_graph_from_mtx(path)

        return {
            "graph_id": path.stem,
            "source_type": "sparse_matrix",
            "source_name": path.name,
            "matrix_path": str(path),
            "num_vertices": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "supported_by_python_graph_loader": "true",
            "notes": "",
        }

    except ValueError as error:
        return {
            "graph_id": path.stem,
            "source_type": "sparse_matrix",
            "source_name": path.name,
            "matrix_path": str(path),
            "num_vertices": "",
            "num_edges": "",
            "supported_by_python_graph_loader": "false",
            "notes": str(error),
        }
    

def save_rows_to_csv(rows: list[dict], output_path: Path) -> None:
    """Save metadata rows to a CSV file."""
    if not rows:
        raise ValueError("No metadata rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)    


def main() -> None:
    matrix_files = sorted(MATRIX_DIR.glob("*.mtx"))
    rows = [build_metadata_row(path) for path in matrix_files]

    save_rows_to_csv(rows, OUTPUT_CSV)

    print(f"Saved {len(rows)} graph metadata rows to:")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()