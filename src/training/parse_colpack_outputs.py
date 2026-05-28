from pathlib import Path
import re
import csv


INPUT_DIR = Path("data/processed/initial_graph_coloring_dataset/colpack_outputs")

OUTPUT_CSV = Path("results/tables/initial_graph_coloring_benchmarks/colpack_initial_benchmark.csv")


GRAPH_INFO_BY_FILE_PREFIX = {
    "ash85": {
        "graph_id": "ash85",
        "source_name": "ash85.mtx",
    },
    "can24": {
        "graph_id": "can_24",
        "source_name": "can_24.mtx",
    },
    "hesspatsmall": {
        "graph_id": "hess_pat_small",
        "source_name": "hess_pat_small.mtx",
    },
    "hesspat": {
        "graph_id": "hess_pat",
        "source_name": "hess_pat.mtx",
    },
    "jacpat": {
        "graph_id": "jac_pat",
        "source_name": "jac_pat.mtx",
    },
}


def list_colpack_output_files() -> list[Path]:
    """Return all saved ColPack output text files."""
    return sorted(INPUT_DIR.glob("*.txt"))


def infer_graph_info_from_filename(file_path: Path) -> dict[str, str]:
    """
    Infer canonical graph_id and source_name from the ColPack output filename.

    Prefixes are checked from longest to shortest so that
    'hesspatsmall' is not incorrectly matched as 'hesspat'.
    """

    stem = file_path.stem

    prefixes_by_length = sorted(
        GRAPH_INFO_BY_FILE_PREFIX.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for prefix, graph_info in prefixes_by_length:
        if stem.startswith(prefix):
            return graph_info

    raise ValueError(
        f"Could not infer canonical graph information from filename: {file_path.name}"
    )


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


def parse_input_graph(text: str) -> str:
    """Extract input graph path from ColPack output text."""
    match = re.search(r"Input graph: (.+)", text)
    if not match:
        raise ValueError("Could not find input graph path in output file.")
    return match.group(1).strip()


def build_row(file_path: Path) -> dict:
    """Build one benchmark row from a saved ColPack output file."""
    text = file_path.read_text(encoding="utf-8")

    input_graph = parse_input_graph(text)
    ordering = parse_requested_ordering(text)
    num_colors = parse_num_colors(text)
    num_vertices, num_edges = parse_vertex_edge_count(text)

    graph_info = infer_graph_info_from_filename(file_path)
    graph_id = graph_info["graph_id"]
    source_name = graph_info["source_name"]

    return {
        "graph_id": graph_id,
        "source_type": "sparse_matrix",
        "source_name": source_name,
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


    print(f"Saved {len(rows)} benchmark rows to:")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()