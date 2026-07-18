from __future__ import annotations

import csv
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLPACK_EXE = Path(
    r"C:\Users\riyat\Documents\ColPack\build\cmake\_build\test_colpack.exe"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_additional_heterogeneous_candidate_graph_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_additional_heterogeneous_generalization"
)

ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def output_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def load_metadata() -> list[dict[str, str]]:
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    if not COLPACK_EXE.exists():
        raise FileNotFoundError(
            f"ColPack executable not found: {COLPACK_EXE}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Week 18 additional metadata not found: {METADATA_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = load_metadata()

    total_expected = len(metadata_rows) * len(ORDERINGS)

    completed = 0
    skipped = 0
    failed = 0

    print("Week 18 additional heterogeneous ColPack evaluation")
    print("---------------------------------------------------")
    print(f"Candidate graphs: {len(metadata_rows)}")
    print(f"Orderings per graph: {len(ORDERINGS)}")
    print(f"Expected runs: {total_expected}")
    print()

    for graph_index, row in enumerate(metadata_rows, start=1):
        graph_id = row["graph_id"]
        matrix_path = PROJECT_ROOT / row["matrix_path"]

        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Matrix file not found for {graph_id}: {matrix_path}"
            )

        print(
            f"[Graph {graph_index}/{len(metadata_rows)}] "
            f"{graph_id}"
        )

        for ordering in ORDERINGS:
            output_path = (
                OUTPUT_DIR
                / output_name(
                    graph_id=graph_id,
                    ordering=ordering,
                )
            )

            # Allow interrupted runs to continue safely.
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"  Skipping existing: {ordering}")
                skipped += 1
                continue

            print(f"  Running: {ordering}")

            try:
                with output_path.open(
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    subprocess.run(
                        [
                            str(COLPACK_EXE),
                            str(matrix_path),
                            ordering,
                        ],
                        stdout=output_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )

                completed += 1

            except subprocess.CalledProcessError as error:
                failed += 1

                print(
                    f"  FAILED: {graph_id}, "
                    f"{ordering}, "
                    f"return code={error.returncode}"
                )

    print()
    print("Week 18 additional ColPack runs finished.")
    print(f"Newly completed runs: {completed}")
    print(f"Skipped existing runs: {skipped}")
    print(f"Failed runs: {failed}")
    print(f"Expected total outputs: {total_expected}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()