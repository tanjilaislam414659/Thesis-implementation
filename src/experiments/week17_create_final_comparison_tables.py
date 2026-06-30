from __future__ import annotations

from pathlib import Path

import pandas as pd


INITIAL_RESULTS_DIR = Path("results/tables/initial_graph_coloring_benchmarks")
GNN_RESULTS_DIR = Path("results/tables/gnn_node_scorer")

# ColPack benchmark files
ORIGINAL_COLPACK_CSV = INITIAL_RESULTS_DIR / "colpack_week15_five_ordering_benchmark.csv"
BICKLE_COLPACK_CSV = INITIAL_RESULTS_DIR / "week17_bickle_hard_cases_colpack_benchmark.csv"
STRUCTURED_COLPACK_CSV = INITIAL_RESULTS_DIR / "week17_structured_colpack_benchmark.csv"

# Week 17 split and target summary
SPLIT_CSV = (
    Path("data/processed/initial_graph_coloring_dataset/splits")
    / "week17_best_available_of_5_split.csv"
)

TARGET_SUMMARY_CSV = INITIAL_RESULTS_DIR / "week17_best_available_of_5_target_summary.csv"

# GNN evaluation files
COLOR_SELECTED_EVAL_CSV = (
    GNN_RESULTS_DIR / "week17_validation_color_selected_checkpoint_per_graph_evaluation.csv"
)

LOSS_SELECTED_EVAL_CSV = (
    GNN_RESULTS_DIR / "week17_validation_loss_selected_checkpoint_per_graph_evaluation.csv"
)

METHOD_SUMMARY_CSV = (
    GNN_RESULTS_DIR / "week17_checkpoint_selection_method_summary.csv"
)

# Output files
FINAL_TEST_COMPARISON_CSV = (
    GNN_RESULTS_DIR / "week17_final_test_graph_comparison.csv"
)

FINAL_GROUP_SUMMARY_CSV = (
    GNN_RESULTS_DIR / "week17_final_test_group_summary.csv"
)

FINAL_METHOD_SUMMARY_CSV = (
    GNN_RESULTS_DIR / "week17_final_method_level_summary.csv"
)


def load_colpack_results() -> pd.DataFrame:
    paths = [
        ORIGINAL_COLPACK_CSV,
        BICKLE_COLPACK_CSV,
        STRUCTURED_COLPACK_CSV,
    ]

    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing ColPack benchmark file: {path}")

        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    required_columns = {"graph_id", "ordering_name", "num_colors"}
    missing = required_columns - set(combined.columns)

    if missing:
        raise ValueError(f"Combined ColPack results missing columns: {missing}")

    combined["num_colors"] = combined["num_colors"].astype(int)

    return combined


def prepare_colpack_summary(colpack: pd.DataFrame) -> pd.DataFrame:
    best_rows = []

    for graph_id, group in colpack.groupby("graph_id"):
        best_colors = int(group["num_colors"].min())
        worst_colors = int(group["num_colors"].max())

        best_orderings = sorted(
            group[group["num_colors"] == best_colors]["ordering_name"]
            .astype(str)
            .unique()
            .tolist()
        )

        sl_rows = group[group["ordering_name"].astype(str).str.upper() == "SMALLEST_LAST"]

        if len(sl_rows) == 0:
            smallest_last_colors = None
        else:
            smallest_last_colors = int(sl_rows.iloc[0]["num_colors"])

        best_rows.append(
            {
                "graph_id": graph_id,
                "best_colpack_colors": best_colors,
                "worst_colpack_colors": worst_colors,
                "colpack_ordering_gap": worst_colors - best_colors,
                "best_colpack_orderings": "; ".join(best_orderings),
                "smallest_last_colors": smallest_last_colors,
                "smallest_last_gap_from_best": (
                    None
                    if smallest_last_colors is None
                    else smallest_last_colors - best_colors
                ),
            }
        )

    return pd.DataFrame(best_rows)


def load_best_seed_rows(
    path: Path,
    method_name: str,
    best_seed: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing GNN evaluation file: {path}")

    df = pd.read_csv(path)

    rows = df[
        (df["split"] == "test")
        & (df["seed"] == best_seed)
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No test rows found for method={method_name}, seed={best_seed}"
        )

    rows = rows.rename(
        columns={
            "num_colors": f"{method_name}_num_colors",
            "gap_from_target": f"{method_name}_gap_from_target",
            "valid": f"{method_name}_valid",
            "seed": f"{method_name}_seed",
        }
    )

    keep_columns = [
        "graph_id",
        f"{method_name}_seed",
        f"{method_name}_num_colors",
        f"{method_name}_gap_from_target",
        f"{method_name}_valid",
    ]

    return rows[keep_columns]


def get_best_seed_from_method_summary(
    method_summary: pd.DataFrame,
    checkpoint_selection: str,
) -> int:
    rows = method_summary[
        method_summary["checkpoint_selection"] == checkpoint_selection
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No method summary row found for {checkpoint_selection}"
        )

    return int(rows.iloc[0]["best_test_seed"])


def main() -> None:
    split = pd.read_csv(SPLIT_CSV)
    target_summary = pd.read_csv(TARGET_SUMMARY_CSV)
    method_summary = pd.read_csv(METHOD_SUMMARY_CSV)

    colpack = load_colpack_results()
    colpack_summary = prepare_colpack_summary(colpack)

    color_method_name = "color_selection"
    loss_method_name = "loss_selection"

    color_best_seed = get_best_seed_from_method_summary(
        method_summary,
        "validation_total_colors_then_validation_loss",
    )

    loss_best_seed = get_best_seed_from_method_summary(
        method_summary,
        "validation_loss",
    )

    color_rows = load_best_seed_rows(
        COLOR_SELECTED_EVAL_CSV,
        color_method_name,
        color_best_seed,
    )

    loss_rows = load_best_seed_rows(
        LOSS_SELECTED_EVAL_CSV,
        loss_method_name,
        loss_best_seed,
    )

    test_split = split[split["split"] == "test"].copy()

    final = (
        test_split
        .merge(target_summary, on="graph_id", how="left")
        .merge(colpack_summary, on="graph_id", how="left")
        .merge(color_rows, on="graph_id", how="left")
        .merge(loss_rows, on="graph_id", how="left")
    )

    if "selected_num_colors" not in final.columns:
        raise ValueError("selected_num_colors column missing after merge.")

    final = final.rename(
        columns={
            "selected_num_colors": "target_colors",
            "selected_ordering": "teacher_ordering",
        }
    )

    final["color_selection_gap_from_best_colpack"] = (
        final["color_selection_num_colors"] - final["best_colpack_colors"]
    )

    final["loss_selection_gap_from_best_colpack"] = (
        final["loss_selection_num_colors"] - final["best_colpack_colors"]
    )

    final["color_selection_minus_loss_selection"] = (
        final["color_selection_num_colors"]
        - final["loss_selection_num_colors"]
    )

    final = final[
        [
            "graph_id",
            "group",
            "target_colors",
            "teacher_ordering",
            "best_colpack_colors",
            "best_colpack_orderings",
            "smallest_last_colors",
            "smallest_last_gap_from_best",
            "color_selection_seed",
            "color_selection_num_colors",
            "color_selection_gap_from_target",
            "color_selection_gap_from_best_colpack",
            "color_selection_valid",
            "loss_selection_seed",
            "loss_selection_num_colors",
            "loss_selection_gap_from_target",
            "loss_selection_gap_from_best_colpack",
            "loss_selection_valid",
            "color_selection_minus_loss_selection",
        ]
    ].sort_values("graph_id")

    group_summary = (
        final.groupby("group")
        .agg(
            num_graphs=("graph_id", "count"),
            target_colors=("target_colors", "sum"),
            best_colpack_colors=("best_colpack_colors", "sum"),
            smallest_last_colors=("smallest_last_colors", "sum"),
            color_selection_colors=("color_selection_num_colors", "sum"),
            color_selection_gap_from_target=("color_selection_gap_from_target", "sum"),
            loss_selection_colors=("loss_selection_num_colors", "sum"),
            loss_selection_gap_from_target=("loss_selection_gap_from_target", "sum"),
        )
        .reset_index()
    )

    group_summary["color_selection_minus_loss_selection"] = (
        group_summary["color_selection_colors"]
        - group_summary["loss_selection_colors"]
    )

    final_method_summary = pd.DataFrame(
        [
            {
                "method": "best_colpack_available",
                "total_colors": int(final["best_colpack_colors"].sum()),
                "gap_from_target": int(
                    final["best_colpack_colors"].sum()
                    - final["target_colors"].sum()
                ),
            },
            {
                "method": "smallest_last",
                "total_colors": int(final["smallest_last_colors"].sum()),
                "gap_from_target": int(
                    final["smallest_last_colors"].sum()
                    - final["target_colors"].sum()
                ),
            },
            {
                "method": "gnn_validation_color_selection",
                "total_colors": int(final["color_selection_num_colors"].sum()),
                "gap_from_target": int(final["color_selection_gap_from_target"].sum()),
            },
            {
                "method": "gnn_validation_loss_selection",
                "total_colors": int(final["loss_selection_num_colors"].sum()),
                "gap_from_target": int(final["loss_selection_gap_from_target"].sum()),
            },
        ]
    )

    FINAL_TEST_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(FINAL_TEST_COMPARISON_CSV, index=False)
    group_summary.to_csv(FINAL_GROUP_SUMMARY_CSV, index=False)
    final_method_summary.to_csv(FINAL_METHOD_SUMMARY_CSV, index=False)

    print("Week 17 final comparison tables")
    print("-------------------------------")
    print()
    print(f"Color-selection best seed: {color_best_seed}")
    print(f"Loss-selection best seed: {loss_best_seed}")
    print()

    print("Final method-level summary:")
    print(final_method_summary.to_string(index=False))
    print()

    print("Final test graph comparison:")
    print(final.to_string(index=False))
    print()

    print("Final group summary:")
    print(group_summary.to_string(index=False))
    print()

    print(f"Saved final test comparison to: {FINAL_TEST_COMPARISON_CSV}")
    print(f"Saved final group summary to: {FINAL_GROUP_SUMMARY_CSV}")
    print(f"Saved final method summary to: {FINAL_METHOD_SUMMARY_CSV}")


if __name__ == "__main__":
    main()