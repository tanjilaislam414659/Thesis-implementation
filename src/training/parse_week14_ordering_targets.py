from __future__ import annotations

import csv
import re
from pathlib import Path


INPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_outputs_week13_expanded"
)

OUTPUT_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "ordering_targets/smallest_last_ordering_targets_week14_expanded.csv"
)

TARGET_ORDERING = "SMALLEST_LAST"


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


def list_target_output_files() -> list[Path]:
    """Return Week 14 ColPack output files for the target ordering only."""
    return sorted(INPUT_DIR.glob("*_smallest_last.txt"))


def infer_graph_id_from_filename(file_path: Path) -> str:
    """Infer graph_id from a Week 14 ColPack output filename."""
    stem = file_path.stem

    for graph_id in sorted(GRAPH_IDS, key=len, reverse=True):
        if stem.startswith(f"{graph_id}_"):
            return graph_id

    raise ValueError(f"Could not infer graph_id from filename: {file_path.name}")


def parse_ordered_vertices(text: str) -> list[int]:
    """Extract ordered vertices from ColPack output text."""
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
        raise ValueError("Ordering positions are not consecutive starting at 0.")

    if len(set(ordered_vertices)) != len(ordered_vertices):
        raise ValueError("Ordering contains duplicate vertices.")

    return ordered_vertices


def build_target_rows(file_path: Path) -> list[dict[str, object]]:
    """Build one target row per node from a SMALLEST_LAST ColPack output file."""
    text = file_path.read_text(encoding="utf-8")

    graph_id = infer_graph_id_from_filename(file_path)
    ordered_vertices = parse_ordered_vertices(text)
    num_vertices = len(ordered_vertices)

    rows = []

    for order_position, node_id in enumerate(ordered_vertices):
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
    """Save ordering-target rows to CSV."""
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
    output_files = list_target_output_files()

    if not output_files:
        raise FileNotFoundError(
            f"No SMALLEST_LAST ColPack output files found in {INPUT_DIR}."
        )

    all_rows = []

    for file_path in output_files:
        graph_rows = build_target_rows(file_path)
        all_rows.extend(graph_rows)

        print(f"Parsed {len(graph_rows)} targets from {file_path.name}")

    save_rows_to_csv(all_rows, OUTPUT_CSV)

    print()
    print(f"Saved {len(all_rows)} node-level ordering target rows to:")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()