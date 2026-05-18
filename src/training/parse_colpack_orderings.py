"""
Parse ColPack vertex orderings from saved output files.

For the first GNN learning prototype, this script extracts
SMALLEST_LAST vertex orderings and stores one row per node:

graph_id, node_id, order_position, target_score, ordering_name, output_file

The target_score is defined so that vertices appearing earlier in the
heuristic ordering receive a larger score. This will later align with
sorting predicted GNN scores in descending order.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


INPUT_DIR = Path("data/processed/initial_graph_coloring_dataset/colpack_outputs")

OUTPUT_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "ordering_targets/smallest_last_ordering_targets.csv"
)

TARGET_ORDERING = "SMALLEST_LAST"


GRAPH_ID_BY_FILE_PREFIX = {
    "ash85": "ash85",
    "can24": "can_24",
    "hesspat": "hess_pat",
    "hesspatsmall": "hess_pat_small",
    "jacpat": "jac_pat",
}


def list_smallest_last_output_files() -> list[Path]:
    """
    Return saved ColPack output files for SMALLEST_LAST only.
    """

    return sorted(INPUT_DIR.glob("*_smallest_last.txt"))


def infer_graph_id_from_filename(file_path: Path) -> str:
    """
    Infer graph_id from ColPack output filename.

    Prefixes are checked from longest to shortest so that
    'hesspatsmall' is not incorrectly matched as 'hesspat'.
    """

    stem = file_path.stem

    prefixes_by_length = sorted(
        GRAPH_ID_BY_FILE_PREFIX.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for prefix, graph_id in prefixes_by_length:
        if stem.startswith(prefix):
            return graph_id

    raise ValueError(f"Could not infer graph_id from filename: {file_path.name}")

def parse_ordered_vertices(text: str) -> list[int]:
    """
    Extract ordered vertices from ColPack output text.

    Expected lines:
    Order Position 0 -> Vertex 13
    Order Position 1 -> Vertex 14
    ...
    """

    matches = re.findall(
        r"Order Position\s+(\d+)\s+->\s+Vertex\s+(\d+)",
        text,
    )

    if not matches:
        raise ValueError("Could not find vertex ordering in ColPack output.")

    ordered_pairs = [(int(position), int(vertex)) for position, vertex in matches]
    ordered_pairs.sort(key=lambda pair: pair[0])

    ordered_vertices = [vertex for _, vertex in ordered_pairs]

    expected_positions = list(range(len(ordered_vertices)))
    actual_positions = [position for position, _ in ordered_pairs]

    if actual_positions != expected_positions:
        raise ValueError(
            "Ordering positions are not consecutive starting at 0. "
            f"Found: {actual_positions[:10]}..."
        )

    if len(set(ordered_vertices)) != len(ordered_vertices):
        raise ValueError("Ordering contains duplicate vertices.")

    return ordered_vertices


def build_target_rows(file_path: Path) -> list[dict[str, object]]:
    """
    Build one target row per node from a ColPack ordering output file.
    """

    text = file_path.read_text(encoding="utf-8")
    graph_id = infer_graph_id_from_filename(file_path)

    ordered_vertices = parse_ordered_vertices(text)
    num_vertices = len(ordered_vertices)

    rows = []

    for order_position, node_id in enumerate(ordered_vertices):
        # Earlier heuristic positions should correspond to larger target scores.
        # The score is normalized to [0, 1] so targets are comparable
        # across graphs with different numbers of vertices.
        if num_vertices > 1:
            target_score = float(num_vertices - 1 - order_position) / float(num_vertices - 1)
        else:
            target_score = 1.0

        rows.append(
            {
                "graph_id": graph_id,
                "node_id": node_id,
                "order_position": order_position,
                "target_score": target_score,
                "ordering_name": TARGET_ORDERING,
                "output_file": str(file_path),
            }
        )

    return rows


def save_rows_to_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """
    Save ordering-target rows to CSV.
    """

    if not rows:
        raise ValueError("No ordering target rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graph_id",
        "node_id",
        "order_position",
        "target_score",
        "ordering_name",
        "output_file",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_files = list_smallest_last_output_files()

    if not output_files:
        raise FileNotFoundError(
            f"No SMALLEST_LAST ColPack output files found in {INPUT_DIR}."
        )

    all_rows = []

    for file_path in output_files:
        graph_rows = build_target_rows(file_path)
        all_rows.extend(graph_rows)

        print(
            f"Parsed {len(graph_rows)} ordering targets from "
            f"{file_path.name}"
        )

    save_rows_to_csv(all_rows, OUTPUT_CSV)

    print()
    print(f"Saved {len(all_rows)} node-level ordering target rows to:")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()