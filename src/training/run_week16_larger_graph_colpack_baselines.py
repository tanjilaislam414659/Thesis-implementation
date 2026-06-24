"""
Run ColPack distance-1 coloring baselines for the Week 16 larger-graph extension.

This script evaluates three larger structural matrices using the same five
ColPack orderings used in the main Week 15/16 experiments.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


COLPACK_EXE = Path(
    r"C:\Users\riyat\Documents\ColPack\build\cmake\_build\test_colpack.exe"
)

OUTPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_outputs_week16_larger_graphs"
)

GRAPH_INPUTS = {
    "bcsstk10": Path("data/raw/matrices/bcsstk10.mtx"),
    "bcsstk14": Path("data/raw/matrices/bcsstk14.mtx"),
    "bcsstk15": Path("data/raw/matrices/bcsstk15.mtx"),
}

ORDERINGS = [
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "NATURAL",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]


def output_file_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def main() -> None:
    if not COLPACK_EXE.exists():
        raise FileNotFoundError(f"ColPack executable not found: {COLPACK_EXE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Week 16 larger-graph ColPack baseline generation")
    print("-----------------------------------------------")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for graph_id, graph_path in GRAPH_INPUTS.items():
        if not graph_path.exists():
            raise FileNotFoundError(f"Input graph not found: {graph_path}")

        for ordering in ORDERINGS:
            output_path = OUTPUT_DIR / output_file_name(graph_id, ordering)

            command = [
                str(COLPACK_EXE),
                str(graph_path),
                ordering,
            ]

            print(f"Running {graph_id} with {ordering}...")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )

            output_path.write_text(result.stdout, encoding="utf-8")

            print(f"  saved: {output_path}")

    print()
    print("Finished larger-graph ColPack baseline generation.")


if __name__ == "__main__":
    main()