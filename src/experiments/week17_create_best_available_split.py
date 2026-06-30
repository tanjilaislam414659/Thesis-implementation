from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\riyat\Documents\Thesis Implementations")

WEEK15_SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "expanded_graph_split_week15.csv"
)

OUTPUT_SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week17_best_available_of_5_split.csv"
)


NEW_WEEK17_SPLITS = [
    # Bickle hard-case family
    {
        "graph_id": "week17_bickle_g10",
        "split": "train",
        "group": "bickle_hard_case",
        "reason": "small controlled hard case for learning beyond SMALLEST_LAST",
    },
    {
        "graph_id": "week17_cycle_square_c8",
        "split": "train",
        "group": "bickle_hard_case",
        "reason": "small cycle-square hard case",
    },
    {
        "graph_id": "week17_cycle_square_c11",
        "split": "validation",
        "group": "bickle_hard_case",
        "reason": "held-out validation case from same controlled family",
    },
    {
        "graph_id": "week17_cycle_square_c14",
        "split": "train",
        "group": "bickle_hard_case",
        "reason": "larger training case from controlled family",
    },
    {
        "graph_id": "week17_cycle_square_c17",
        "split": "test",
        "group": "bickle_hard_case",
        "reason": "held-out larger hard case",
    },
    {
        "graph_id": "week17_cycle_square_c20",
        "split": "test",
        "group": "bickle_hard_case",
        "reason": "held-out larger hard case",
    },

    # Structured sparse matrices
    {
        "graph_id": "week17_nos1",
        "split": "train",
        "group": "structured_matrix",
        "reason": "real banded matrix",
    },
    {
        "graph_id": "week17_bwm200",
        "split": "train",
        "group": "structured_matrix",
        "reason": "real narrow-banded matrix",
    },
    {
        "graph_id": "week17_arrowhead_100",
        "split": "train",
        "group": "structured_matrix",
        "reason": "constructed controlled arrowhead pattern",
    },
    {
        "graph_id": "week17_lshp_265",
        "split": "validation",
        "group": "structured_matrix",
        "reason": "finite-element/block-like validation matrix",
    },
    {
        "graph_id": "week17_gr_30_30",
        "split": "test",
        "group": "structured_matrix",
        "reason": "real PDE/grid-like matrix where SMALLEST_LAST is worse than best",
    },
    {
        "graph_id": "week17_bcsstk08",
        "split": "test",
        "group": "structured_matrix",
        "reason": "larger block-structured structural matrix",
    },
]


def main() -> None:
    week15_split = pd.read_csv(WEEK15_SPLIT_CSV)

    if not {"graph_id", "split"}.issubset(week15_split.columns):
        raise ValueError(
            f"Week 15 split must contain graph_id and split columns. "
            f"Got: {list(week15_split.columns)}"
        )

    base_rows = week15_split[["graph_id", "split"]].copy()
    base_rows["group"] = "original_sparse_matrix"
    base_rows["reason"] = "preserved from Week 15 split"

    new_rows = pd.DataFrame(NEW_WEEK17_SPLITS)

    combined = pd.concat([base_rows, new_rows], ignore_index=True)

    if combined["graph_id"].duplicated().any():
        duplicates = combined[combined["graph_id"].duplicated()]["graph_id"].tolist()
        raise ValueError(f"Duplicate graph IDs in split: {duplicates}")

    OUTPUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_SPLIT_CSV, index=False)

    print("Created Week 17 split file.")
    print(f"Saved to: {OUTPUT_SPLIT_CSV}")
    print()

    print("Split counts:")
    print(combined["split"].value_counts())
    print()

    print("Group/split counts:")
    print(pd.crosstab(combined["group"], combined["split"]))
    print()

    print("Full split:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()