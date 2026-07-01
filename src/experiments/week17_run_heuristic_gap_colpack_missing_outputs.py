from pathlib import Path
import subprocess


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")
COLPACK_EXE = Path(
    r"C:\Users\riyat\Documents\ColPack\build\cmake\_build\test_colpack.exe"
)

MATRIX_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "matrices"
    / "week17_heuristic_gap_extension"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_heuristic_gap_extension"
)

ORDERINGS = [
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "NATURAL",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]


def output_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COLPACK_EXE.exists():
        raise FileNotFoundError(f"ColPack runner not found: {COLPACK_EXE}")

    matrix_files = sorted(MATRIX_DIR.glob("*.mtx"))

    if not matrix_files:
        raise FileNotFoundError(f"No .mtx files found in {MATRIX_DIR}")

    run_count = 0
    skip_count = 0

    for matrix_path in matrix_files:
        graph_id = matrix_path.stem

        for ordering in ORDERINGS:
            out_path = OUTPUT_DIR / output_name(graph_id, ordering)

            if out_path.exists():
                skip_count += 1
                continue

            print(f"Running {graph_id} with {ordering} -> {out_path.name}")

            with open(out_path, "w", encoding="utf-8") as out_file:
                subprocess.run(
                    [str(COLPACK_EXE), str(matrix_path), ordering],
                    stdout=out_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

            run_count += 1

    print()
    print("Done.")
    print(f"New ColPack runs completed: {run_count}")
    print(f"Existing outputs skipped: {skip_count}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()