"""
Parse ColPack output files for the Week 16 larger-graph extension.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


INPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_outputs_week16_larger_graphs"
)

OUTPUT_CSV = Path(
    "results/tables/initial_graph_coloring_benchmarks/"
    "colpack_week16_larger_graph_benchmark.csv"
)

ORDERINGS = [
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "NATURAL",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]

GRAPH_IDS = [
    "bcsstk10",
    "bcsstk14",
    "bcsstk15",
]


def output_file_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def parse_colpack_output(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")

    colors_match = re.search(r"\[Total Colors =\s*(\d+)\]", text)
    graph_match = re.search(
        r"\[Vertex Count =\s*(\d+);\s*Edge Count =\s*(\d+)\]",
        text,
    )

    if colors_match is None:
        raise ValueError(f"Could not parse total colors from {path}")

    if graph_match is None:
        raise ValueError(f"Could not parse vertex/edge count from {path}")

    return {
        "num_colors": int(colors_match.group(1)),
        "num_vertices": int(graph_match.group(1)),
        "num_edges": int(graph_match.group(2)),
    }


def main() -> None:
    rows = []

    for graph_id in GRAPH_IDS:
        for ordering in ORDERINGS:
            path = INPUT_DIR / output_file_name(graph_id, ordering)

            if not path.exists():
                raise FileNotFoundError(f"Missing ColPack output: {path}")

            parsed = parse_colpack_output(path)

            rows.append(
                {
                    "graph_id": graph_id,
                    "source_type": "sparse_matrix",
                    "source_name": f"{graph_id}.mtx",
                    "num_vertices": parsed["num_vertices"],
                    "num_edges": parsed["num_edges"],
                    "coloring_distance": 1,
                    "method_family": "colpack",
                    "method_name": "greedy_coloring",
                    "ordering_name": ordering,
                    "num_colors": parsed["num_colors"],
                    "output_file": str(path),
                }
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "graph_id",
            "source_type",
            "source_name",
            "num_vertices",
            "num_edges",
            "coloring_distance",
            "method_family",
            "method_name",
            "ordering_name",
            "num_colors",
            "output_file",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Parsed larger-graph ColPack outputs")
    print(f"Rows: {len(rows)}")
    print(f"Saved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
    