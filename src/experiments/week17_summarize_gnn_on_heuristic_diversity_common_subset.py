from pathlib import Path
import pandas as pd


CASES_CSV = Path(
    "results/tables/gnn_node_scorer/week17_heuristic_diversity_covered_cases.csv"
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

COMMON_CASES_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_common_gnn_cases.csv"
METHOD_SUMMARY_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_common_gnn_vs_oracle_summary.csv"
GNN_SEED_SUMMARY_OUTPUT = OUTPUT_DIR / "week17_heuristic_diversity_common_gnn_seed_summary.csv"


ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def fixed_heuristic_summary(cases: pd.DataFrame) -> pd.DataFrame:
    oracle_total = int(cases["best_colors"].sum())
    num_graphs = len(cases)

    rows = []

    for ordering in ORDERINGS:
        total_colors = int(cases[ordering].sum())
        gap = total_colors - oracle_total
        matched = int((cases[ordering] == cases["best_colors"]).sum())

        rows.append(
            {
                "method": ordering,
                "selection": "FIXED_COLPACK_ORDERING",
                "num_graphs": num_graphs,
                "total_colors_mean": total_colors,
                "total_colors_min": total_colors,
                "total_colors_max": total_colors,
                "gap_to_oracle_mean": gap,
                "gap_to_oracle_min": gap,
                "gap_to_oracle_max": gap,
                "graphs_matching_oracle_mean": matched,
                "graphs_matching_oracle_min": matched,
                "graphs_matching_oracle_max": matched,
                "oracle_total_colors": oracle_total,
            }
        )

    rows.append(
        {
            "method": "BEST_OF_5_ORACLE",
            "selection": "BEST_OF_5_ORACLE",
            "num_graphs": num_graphs,
            "total_colors_mean": oracle_total,
            "total_colors_min": oracle_total,
            "total_colors_max": oracle_total,
            "gap_to_oracle_mean": 0,
            "gap_to_oracle_min": 0,
            "gap_to_oracle_max": 0,
            "graphs_matching_oracle_mean": num_graphs,
            "graphs_matching_oracle_min": num_graphs,
            "graphs_matching_oracle_max": num_graphs,
            "oracle_total_colors": oracle_total,
        }
    )

    return pd.DataFrame(rows)


def gnn_summary(label: str, path: Path, cases: pd.DataFrame) -> pd.DataFrame:
    gnn = pd.read_csv(path)
    graph_ids = set(cases["graph_id"])

    gnn = gnn[gnn["graph_id"].isin(graph_ids)].copy()

    merged = gnn.merge(
        cases[["graph_id", "best_colors"]],
        on="graph_id",
        how="left",
    )

    merged["gap_to_oracle"] = merged["num_colors"] - merged["best_colors"]
    merged["matches_oracle"] = merged["num_colors"] == merged["best_colors"]

    seed_summary = (
        merged.groupby("seed")
        .agg(
            num_graphs=("graph_id", "count"),
            total_colors=("num_colors", "sum"),
            oracle_total_colors=("best_colors", "sum"),
            gap_to_oracle=("gap_to_oracle", "sum"),
            graphs_matching_oracle=("matches_oracle", "sum"),
        )
        .reset_index()
    )

    # Safety check: every seed must be evaluated on the same common graph set.
    if seed_summary["num_graphs"].nunique() != 1:
        raise ValueError(
            f"{label}: not all seeds have the same number of graphs.\n"
            f"{seed_summary}"
        )

    return pd.DataFrame(
        [
            {
                "method": "GNN_MEAN_OVER_SEEDS",
                "selection": label,
                "num_graphs": int(seed_summary["num_graphs"].iloc[0]),
                "total_colors_mean": seed_summary["total_colors"].mean(),
                "total_colors_min": seed_summary["total_colors"].min(),
                "total_colors_max": seed_summary["total_colors"].max(),
                "gap_to_oracle_mean": seed_summary["gap_to_oracle"].mean(),
                "gap_to_oracle_min": seed_summary["gap_to_oracle"].min(),
                "gap_to_oracle_max": seed_summary["gap_to_oracle"].max(),
                "graphs_matching_oracle_mean": seed_summary[
                    "graphs_matching_oracle"
                ].mean(),
                "graphs_matching_oracle_min": seed_summary[
                    "graphs_matching_oracle"
                ].min(),
                "graphs_matching_oracle_max": seed_summary[
                    "graphs_matching_oracle"
                ].max(),
                "oracle_total_colors": int(seed_summary["oracle_total_colors"].iloc[0]),
            }
        ]
    ), seed_summary


def main() -> None:
    cases = pd.read_csv(CASES_CSV)

    gnn_graph_sets = []

    for label, path in GNN_EVAL_FILES.items():
        gnn = pd.read_csv(path)
        gnn_graph_sets.append(set(gnn["graph_id"]))

    common_graphs = set(cases["graph_id"])

    for graph_set in gnn_graph_sets:
        common_graphs = common_graphs & graph_set

    common_graphs = sorted(common_graphs)

    common_cases = (
        cases[cases["graph_id"].isin(common_graphs)]
        .copy()
        .sort_values("graph_id")
        .reset_index(drop=True)
    )

    missing_from_gnn = sorted(set(cases["graph_id"]) - set(common_graphs))

    fixed_summary = fixed_heuristic_summary(common_cases)

    gnn_method_summaries = []
    gnn_seed_summaries = []

    for label, path in GNN_EVAL_FILES.items():
        method_summary, seed_summary = gnn_summary(label, path, common_cases)
        gnn_method_summaries.append(method_summary)
        seed_summary["selection"] = label
        gnn_seed_summaries.append(seed_summary)

    method_summary = pd.concat(
        [fixed_summary] + gnn_method_summaries,
        ignore_index=True,
    ).sort_values(
        ["gap_to_oracle_mean", "total_colors_mean", "method"],
        ascending=[True, True, True],
    )

    gnn_seed_summary = pd.concat(gnn_seed_summaries, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common_cases.to_csv(COMMON_CASES_OUTPUT, index=False)
    method_summary.to_csv(METHOD_SUMMARY_OUTPUT, index=False)
    gnn_seed_summary.to_csv(GNN_SEED_SUMMARY_OUTPUT, index=False)

    print("Corrected GNN vs best-of-5 summary on common subset")
    print("---------------------------------------------------")
    print()
    print(f"Heuristic-diversity covered cases before GNN filter: {len(cases)}")
    print(f"Common cases with GNN evaluation: {len(common_cases)}")
    print(f"Common graph IDs: {common_graphs}")
    print()
    print("Missing from existing GNN evaluation:")
    print(missing_from_gnn)
    print()
    print("Corrected method summary:")
    print(method_summary.to_string(index=False))
    print()
    print("Corrected GNN seed summary:")
    print(gnn_seed_summary.to_string(index=False))
    print()
    print(f"Saved common cases to: {COMMON_CASES_OUTPUT}")
    print(f"Saved corrected method summary to: {METHOD_SUMMARY_OUTPUT}")
    print(f"Saved corrected GNN seed summary to: {GNN_SEED_SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()