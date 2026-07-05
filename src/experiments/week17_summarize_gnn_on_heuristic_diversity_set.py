from pathlib import Path
import pandas as pd


CLEAN_CASES_CSV = Path(
    "results/tables/gnn_node_scorer/week17_clean_heuristic_diversity_cases.csv"
)

TARGETS_CSV = Path(
    "data/processed/initial_graph_coloring_dataset/ordering_targets/"
    "week17_best_available_of_5_ordering_targets.csv"
)

GNN_EVAL_FILES = {
    "GNN_COLOR_SELECTED": Path(
        "results/tables/gnn_node_scorer/"
        "week17_validation_color_selected_checkpoint_per_graph_evaluation.csv"
    ),
    "GNN_LOSS_SELECTED": Path(
        "results/tables/gnn_node_scorer/"
        "week17_validation_loss_selected_checkpoint_per_graph_evaluation.csv"
    ),
}

OUTPUT_DIR = Path("results/tables/gnn_node_scorer")

COVERED_CASES_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_covered_cases.csv"
METHOD_SUMMARY_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_gnn_vs_oracle_summary.csv"
GNN_SEED_SUMMARY_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_gnn_seed_summary.csv"
GNN_PER_GRAPH_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_gnn_per_graph_results.csv"


ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def build_fixed_heuristic_summary(cases: pd.DataFrame) -> pd.DataFrame:
    oracle_total = int(cases["best_colors"].sum())
    num_graphs = len(cases)

    rows = []

    for ordering in ORDERINGS:
        total_colors = int(cases[ordering].sum())
        gap = total_colors - oracle_total
        matched = int((cases[ordering] == cases["best_colors"]).sum())

        rows.append(
            {
                "checkpoint_selection": "FIXED_COLPACK_ORDERING",
                "method": ordering,
                "num_graphs": num_graphs,
                "total_colors_mean": total_colors,
                "total_colors_min": total_colors,
                "total_colors_max": total_colors,
                "gap_to_best_of_5_oracle_mean": gap,
                "gap_to_best_of_5_oracle_min": gap,
                "gap_to_best_of_5_oracle_max": gap,
                "graphs_matching_best_of_5_mean": matched,
                "graphs_matching_best_of_5_min": matched,
                "graphs_matching_best_of_5_max": matched,
                "oracle_total_colors": oracle_total,
            }
        )

    rows.append(
        {
            "checkpoint_selection": "BEST_OF_5_ORACLE",
            "method": "BEST_OF_5_ORACLE",
            "num_graphs": num_graphs,
            "total_colors_mean": oracle_total,
            "total_colors_min": oracle_total,
            "total_colors_max": oracle_total,
            "gap_to_best_of_5_oracle_mean": 0,
            "gap_to_best_of_5_oracle_min": 0,
            "gap_to_best_of_5_oracle_max": 0,
            "graphs_matching_best_of_5_mean": num_graphs,
            "graphs_matching_best_of_5_min": num_graphs,
            "graphs_matching_best_of_5_max": num_graphs,
            "oracle_total_colors": oracle_total,
        }
    )

    return pd.DataFrame(rows)


def summarize_gnn_file(label: str, path: Path, cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gnn = pd.read_csv(path)

    graph_ids = set(cases["graph_id"])

    gnn_subset = gnn[gnn["graph_id"].isin(graph_ids)].copy()

    if gnn_subset.empty:
        raise ValueError(f"No GNN results found for covered graphs in {path}")

    merged = gnn_subset.merge(
        cases[["graph_id", "best_colors"]],
        on="graph_id",
        how="left",
        suffixes=("", "_oracle"),
    )

    if merged["best_colors"].isna().any():
        missing = merged[merged["best_colors"].isna()]["graph_id"].unique()
        raise ValueError(f"Missing oracle colors after merge: {missing}")

    merged["matches_best_of_5"] = merged["num_colors"] == merged["best_colors"]
    merged["gap_to_best_of_5"] = merged["num_colors"] - merged["best_colors"]
    merged["gnn_checkpoint_label"] = label

    oracle_total = int(cases["best_colors"].sum())
    num_graphs = len(cases)

    seed_summary = (
        merged.groupby("seed")
        .agg(
            num_graphs=("graph_id", "count"),
            total_colors=("num_colors", "sum"),
            oracle_total_colors=("best_colors", "sum"),
            graphs_matching_best_of_5=("matches_best_of_5", "sum"),
            total_gap_to_best_of_5=("gap_to_best_of_5", "sum"),
            mean_gap_per_graph=("gap_to_best_of_5", "mean"),
        )
        .reset_index()
    )

    seed_summary["gnn_checkpoint_label"] = label

    method_summary = pd.DataFrame(
        [
            {
                "checkpoint_selection": label,
                "method": "GNN_MEAN_OVER_SEEDS",
                "num_graphs": num_graphs,
                "total_colors_mean": seed_summary["total_colors"].mean(),
                "total_colors_min": seed_summary["total_colors"].min(),
                "total_colors_max": seed_summary["total_colors"].max(),
                "gap_to_best_of_5_oracle_mean": seed_summary[
                    "total_gap_to_best_of_5"
                ].mean(),
                "gap_to_best_of_5_oracle_min": seed_summary[
                    "total_gap_to_best_of_5"
                ].min(),
                "gap_to_best_of_5_oracle_max": seed_summary[
                    "total_gap_to_best_of_5"
                ].max(),
                "graphs_matching_best_of_5_mean": seed_summary[
                    "graphs_matching_best_of_5"
                ].mean(),
                "graphs_matching_best_of_5_min": seed_summary[
                    "graphs_matching_best_of_5"
                ].min(),
                "graphs_matching_best_of_5_max": seed_summary[
                    "graphs_matching_best_of_5"
                ].max(),
                "oracle_total_colors": oracle_total,
            }
        ]
    )

    return merged, seed_summary, method_summary


def main() -> None:
    cases = pd.read_csv(CLEAN_CASES_CSV)
    targets = pd.read_csv(TARGETS_CSV)

    covered_graphs = sorted(set(cases["graph_id"]) & set(targets["graph_id"]))

    covered_cases = (
        cases[cases["graph_id"].isin(covered_graphs)]
        .copy()
        .sort_values("graph_id")
        .reset_index(drop=True)
    )

    if covered_cases.empty:
        raise ValueError("No covered heuristic-diversity cases found.")

    fixed_summary = build_fixed_heuristic_summary(covered_cases)

    all_gnn_per_graph = []
    all_gnn_seed_summary = []
    all_gnn_method_summary = []

    for label, path in GNN_EVAL_FILES.items():
        per_graph, seed_summary, method_summary = summarize_gnn_file(
            label, path, covered_cases
        )
        all_gnn_per_graph.append(per_graph)
        all_gnn_seed_summary.append(seed_summary)
        all_gnn_method_summary.append(method_summary)

    gnn_per_graph_df = pd.concat(all_gnn_per_graph, ignore_index=True)
    gnn_seed_summary_df = pd.concat(all_gnn_seed_summary, ignore_index=True)

    method_summary_df = pd.concat(
        [fixed_summary] + all_gnn_method_summary,
        ignore_index=True,
    )

    method_summary_df = method_summary_df.sort_values(
        [
            "gap_to_best_of_5_oracle_mean",
            "total_colors_mean",
            "method",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    covered_cases.to_csv(COVERED_CASES_OUTPUT, index=False)
    method_summary_df.to_csv(METHOD_SUMMARY_OUTPUT, index=False)
    gnn_seed_summary_df.to_csv(GNN_SEED_SUMMARY_OUTPUT, index=False)
    gnn_per_graph_df.to_csv(GNN_PER_GRAPH_OUTPUT, index=False)

    print("GNN on heuristic-diversity covered set")
    print("-------------------------------------")
    print()
    print(f"Original clean heuristic-diversity cases: {len(cases)}")
    print(f"Covered by Week 17 best-of-5 targets: {len(covered_cases)}")
    print(f"Covered graph IDs: {covered_graphs}")
    print()
    print("Method summary:")
    print(method_summary_df.to_string(index=False))
    print()
    print("GNN seed summary:")
    print(gnn_seed_summary_df.to_string(index=False))
    print()
    print(f"Saved covered cases to: {COVERED_CASES_OUTPUT}")
    print(f"Saved method summary to: {METHOD_SUMMARY_OUTPUT}")
    print(f"Saved GNN seed summary to: {GNN_SEED_SUMMARY_OUTPUT}")
    print(f"Saved GNN per-graph results to: {GNN_PER_GRAPH_OUTPUT}")


if __name__ == "__main__":
    main()