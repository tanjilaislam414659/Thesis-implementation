from pathlib import Path
import subprocess


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")
COLPACK_EXE = Path(r"C:\Users\riyat\Documents\ColPack\build\cmake\_build\test_colpack.exe")

MATRIX_DIR = PROJECT_ROOT / "data" / "raw" / "matrices"
OUTPUT_DIR = Path(
    "data/processed/initial_graph_coloring_dataset/"
    "colpack_outputs_week15_extra_orderings"
)

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

ORDERINGS = [
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
]


def output_name(graph_id: str, ordering: str) -> str:
    return f"{graph_id}_{ordering.lower()}.txt"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COLPACK_EXE.exists():
        raise FileNotFoundError(f"ColPack runner not found: {COLPACK_EXE}")

    for graph_id in GRAPH_IDS:
        matrix_path = MATRIX_DIR / f"{graph_id}.mtx"

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