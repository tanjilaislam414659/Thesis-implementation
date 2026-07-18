from __future__ import annotations

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(
    r"C:\Users\riyat\Documents\Thesis Implementations"
)

COLPACK_EXE = Path(
    r"C:\Users\riyat\Documents\ColPack\build\cmake\_build\test_colpack.exe"
)

MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week18_controlled_mixed_joins"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week18_controlled_mixed_joins"
)

ORDERINGS = [
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "NATURAL",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]

EXPECTED_GRAPH_COUNT = 105
EXPECTED_OUTPUT_COUNT = EXPECTED_GRAPH_COUNT * len(ORDERINGS)


def output_name(
    graph_id: str,
    ordering: str,
) -> str:
    return (
        f"{graph_id}_"
        f"{ordering.lower()}.txt"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not COLPACK_EXE.exists():
        raise FileNotFoundError(
            f"ColPack executable not found: "
            f"{COLPACK_EXE}"
        )

    if not MATRIX_DIR.exists():
        raise FileNotFoundError(
            f"Matrix directory not found: "
            f"{MATRIX_DIR}"
        )

    matrix_paths = sorted(
        MATRIX_DIR.glob(
            "week18_controlled_mixed_*.mtx"
        )
    )

    if len(matrix_paths) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRAPH_COUNT} matrices, "
            f"found {len(matrix_paths)}."
        )

    completed_runs = 0
    skipped_runs = 0
    failed_runs: list[
        tuple[str, str, str]
    ] = []

    print(
        "Running Week 18 controlled mixed-join "
        "ColPack experiments."
    )
    print(
        "------------------------------------------"
    )
    print(
        f"Graphs: {len(matrix_paths)}"
    )
    print(
        f"Orderings per graph: {len(ORDERINGS)}"
    )
    print(
        f"Expected outputs: {EXPECTED_OUTPUT_COUNT}"
    )
    print()

    for graph_index, matrix_path in enumerate(
        matrix_paths,
        start=1,
    ):
        graph_id = matrix_path.stem

        print(
            f"[{graph_index}/{len(matrix_paths)}] "
            f"{graph_id}"
        )

        for ordering in ORDERINGS:
            output_path = (
                OUTPUT_DIR
                / output_name(
                    graph_id,
                    ordering,
                )
            )

            # Preserve already completed runs.
            if (
                output_path.exists()
                and output_path.stat().st_size > 0
            ):
                print(
                    f"  Skipping {ordering}: "
                    f"output already exists."
                )

                skipped_runs += 1
                continue

            print(
                f"  Running {ordering}"
            )

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

                if output_path.stat().st_size == 0:
                    raise RuntimeError(
                        "ColPack produced an empty "
                        "output file."
                    )

                completed_runs += 1

            except Exception as error:
                failed_runs.append(
                    (
                        graph_id,
                        ordering,
                        str(error),
                    )
                )

                print(
                    f"  FAILED {ordering}: "
                    f"{error}"
                )

    output_paths = sorted(
        OUTPUT_DIR.glob("*.txt")
    )

    nonempty_output_paths = [
        path
        for path in output_paths
        if path.stat().st_size > 0
    ]

    print()
    print(
        "Week 18 ColPack run summary"
    )
    print(
        "---------------------------"
    )
    print(
        f"Newly completed runs: "
        f"{completed_runs}"
    )
    print(
        f"Skipped existing runs: "
        f"{skipped_runs}"
    )
    print(
        f"Nonempty output files: "
        f"{len(nonempty_output_paths)}"
    )
    print(
        f"Expected output files: "
        f"{EXPECTED_OUTPUT_COUNT}"
    )
    print(
        f"Failed runs: "
        f"{len(failed_runs)}"
    )

    if failed_runs:
        print()
        print(
            "Failed graph-ordering combinations:"
        )

        for (
            graph_id,
            ordering,
            error_message,
        ) in failed_runs:
            print(
                f"  {graph_id} | "
                f"{ordering} | "
                f"{error_message}"
            )

        raise RuntimeError(
            f"{len(failed_runs)} ColPack runs failed."
        )

    if (
        len(nonempty_output_paths)
        != EXPECTED_OUTPUT_COUNT
    ):
        raise ValueError(
            f"Expected {EXPECTED_OUTPUT_COUNT} "
            f"nonempty outputs, found "
            f"{len(nonempty_output_paths)}."
        )

    print()
    print(
        "All controlled mixed-join ColPack "
        "runs completed successfully."
    )
    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()