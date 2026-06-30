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
    / "week17_bickle_exact_family"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_bickle_exact_family"
)

GRAPH_FILES = [
    "week17_cycle_square_c8.mtx",
    "week17_cycle_square_c11.mtx",
    "week17_cycle_square_c14.mtx",
    "week17_cycle_square_c17.mtx",
    "week17_cycle_square_c20.mtx",
    "week17_cycle_square_c23.mtx",
    "week17_cycle_square_c26.mtx",
    "week17_cycle_square_c29.mtx",
    "week17_cycle_square_c32.mtx",
]

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

    for graph_file in GRAPH_FILES:
        matrix_path = MATRIX_DIR / graph_file
        graph_id = matrix_path.stem

        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        for ordering in ORDERINGS:
            out_path = OUTPUT_DIR / output_name(graph_id, ordering)

            print(f"Running {graph_id} with {ordering} -> {out_path.name}")

            with open(out_path, "w", encoding="utf-8") as out_file:
                subprocess.run(
                    [str(COLPACK_EXE), str(matrix_path), ordering],
                    stdout=out_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    print()
    print("Done. ColPack outputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()