from pathlib import Path
import subprocess
import csv


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")
COLPACK_EXE = Path(
    r"C:\Users\riyat\Documents\ColPack\build\cmake\_build\test_colpack.exe"
)

MATRIX_DIR = PROJECT_ROOT / "data" / "raw" / "matrices"

METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week17_structured_matrix_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "colpack_outputs_week17_structured_matrices"
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


def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COLPACK_EXE.exists():
        raise FileNotFoundError(f"ColPack runner not found: {COLPACK_EXE}")

    rows = load_metadata()

    for row in rows:
        graph_id = row["graph_id"]
        matrix_path = MATRIX_DIR / row["matrix_file"]

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

    print("\nDone. ColPack outputs saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()