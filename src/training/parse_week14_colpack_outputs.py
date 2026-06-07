from pathlib import Path
import re
import csv


INPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/colpack_outputs_week13_expanded"
)

OUTPUT_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week14_expanded_benchmark.csv"
)


GRAPH_IDS = [
    "ash85",
    "can_24",
    "hess_pat",
    "hess_pat_small",
    "jac_pat",
    "bcsstk01",
    "bcsstk03",
    "bcsstk04",
    "bcsstk05",
    "bcsstk06",
    "dwt_234",
    "dwt_361",
    "dwt_419",
    "west0479",
    "sherman1",
]


def list_colpack_output_files() -> list[Path]:
    """Return all saved Week 14 ColPack output text files."""
    return sorted(INPUT_DIR.glob("*.txt"))


def parse_num_colors(text: str) -> int:
    """Extract total number of colors from ColPack output text."""
    match = re.search(r"\[Total Colors = (\d+)\]", text)
    if not match:
        raise ValueError("Could not find total colors in output file.")
    return int(match.group(1))


def parse_vertex_edge_count(text: str) -> tuple[int, int]:
    """Extract vertex count and edge count from ColPack output text."""
    match = re.search(r"\[Vertex Count = (\d+); Edge Count = (\d+)\]", text)
    if not match:
        raise ValueError("Could not find vertex/edge count in output file.")
    return int(match.group(1)), int(match.group(2))


def parse_requested_ordering(text: str) -> str:
    """Extract requested ordering from ColPack output text."""
    match = re.search(r"Requested ordering: (\S+)", text)
    if not match:
        raise ValueError("Could not find requested ordering in output file.")
    return match.group(1)


def infer_graph_id_from_filename(file_path: Path) -> str:
    """Infer graph_id from a Week 14 ColPack output filename."""
    stem = file_path.stem

    for graph_id in sorted(GRAPH_IDS, key=len, reverse=True):
        if stem.startswith(f"{graph_id}_"):
            return graph_id

    raise ValueError(f"Could not infer graph_id from filename: {file_path.name}")


def build_row(file_path: Path) -> dict:
    """Build one benchmark row from a saved Week 14 ColPack output file."""
    text = file_path.read_text(encoding="utf-8")

    graph_id = infer_graph_id_from_filename(file_path)
    ordering = parse_requested_ordering(text)
    num_colors = parse_num_colors(text)
    num_vertices, num_edges = parse_vertex_edge_count(text)

    return {
        "graph_id": graph_id,
        "source_type": "sparse_matrix",
        "source_name": f"{graph_id}.mtx",
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "coloring_distance": 1,
        "method_family": "colpack",
        "method_name": "greedy_coloring",
        "ordering_name": ordering,
        "num_colors": num_colors,
        "runtime": 0,
        "valid": "true",
        "output_file": str(file_path),
    }


def save_rows_to_csv(rows: list[dict], output_path: Path) -> None:
    """Save parsed benchmark rows to a CSV file."""
    if not rows:
        raise ValueError("No rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = list_colpack_output_files()
    rows = [build_row(file_path) for file_path in files]

    save_rows_to_csv(rows, OUTPUT_CSV)

    print(f"Parsed {len(rows)} ColPack output files.")
    print(f"Saved benchmark table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()