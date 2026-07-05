from pathlib import Path
import pandas as pd


COMMON_CASES_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_heuristic_diversity_common_gnn_cases.csv"
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

PER_GRAPH_OUTPUT = OUTPUT_DIR / "week17_best_of_k_gnn_heuristic_diversity_per_graph.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "week17_best_of_k_gnn_heuristic_diversity_summary.csv"


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

    rows.append(
        {
            "method": "BEST_OF_5_ORACLE",
            "candidate_pool": "COLPACK_ORACLE",
            "num_graphs": num_graphs,
            "total_colors": oracle_total,
            "gap_to_oracle": 0,
            "graphs_matching_oracle": num_graphs,
            "oracle_total_colors": oracle_total,
        }
    )

    for ordering in ORDERINGS:
        total_colors = int(cases[ordering].sum())
        gap = total_colors - oracle_total
        matched = int((cases[ordering] == cases["best_colors"]).sum())

        rows.append(
            {
                "method": ordering,
                "candidate_pool": "FIXED_COLPACK_ORDERING",
                "num_graphs": num_graphs,
                "total_colors": total_colors,
                "gap_to_oracle": gap,
                "graphs_matching_oracle": matched,
                "oracle_total_colors": oracle_total,
            }
        )

    return pd.DataFrame(rows)


def load_gnn_candidates(cases: pd.DataFrame) -> pd.DataFrame:
    graph_ids = set(cases["graph_id"])
    all_rows = []

    for label, path in GNN_EVAL_FILES.items():
        df = pd.read_csv(path)
        df = df[df["graph_id"].isin(graph_ids)].copy()

        if df.empty:
            raise ValueError(f"No GNN rows found in {path}")

        if "valid" in df.columns:
            df = df[df["valid"] == True].copy()

        df["candidate_pool"] = label
        df["candidate_id"] = label + "_seed_" + df["seed"].astype(str)

        all_rows.append(df)

    candidates = pd.concat(all_rows, ignore_index=True)

    candidates = candidates.merge(
        cases[["graph_id", "best_colors"]],
        on="graph_id",
        how="left",
    )

    candidates["gap_to_oracle"] = candidates["num_colors"] - candidates["best_colors"]
    candidates["matches_oracle"] = candidates["num_colors"] == candidates["best_colors"]

    return candidates


def summarize_single_seed_gnn(
    candidates: pd.DataFrame,
    cases: pd.DataFrame,
    candidate_pool: str,
) -> pd.DataFrame:
    subset = candidates[candidates["candidate_pool"] == candidate_pool].copy()

    seed_summary = (
        subset.groupby("seed")
        .agg(
            total_colors=("num_colors", "sum"),
            oracle_total_colors=("best_colors", "sum"),
            gap_to_oracle=("gap_to_oracle", "sum"),
            graphs_matching_oracle=("matches_oracle", "sum"),
            num_graphs=("graph_id", "count"),
        )
        .reset_index()
    )

    return pd.DataFrame(
        [
            {
                "method": f"{candidate_pool}_MEAN_SINGLE_SEED",
                "candidate_pool": candidate_pool,
                "num_graphs": int(seed_summary["num_graphs"].iloc[0]),
                "total_colors": seed_summary["total_colors"].mean(),
                "gap_to_oracle": seed_summary["gap_to_oracle"].mean(),
                "graphs_matching_oracle": seed_summary[
                    "graphs_matching_oracle"
                ].mean(),
                "oracle_total_colors": int(seed_summary["oracle_total_colors"].iloc[0]),
            }
        ]
    )


def summarize_best_of_k(
    candidates: pd.DataFrame,
    cases: pd.DataFrame,
    candidate_pool_label: str,
    allowed_pools: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = candidates[candidates["candidate_pool"].isin(allowed_pools)].copy()

    # Sort so that the first row per graph is the best valid GNN coloring.
    subset = subset.sort_values(
        by=["graph_id", "num_colors", "gap_to_oracle", "candidate_id"],
        ascending=[True, True, True, True],
    )

    best_per_graph = (
        subset.groupby("graph_id", as_index=False)
        .first()
        .sort_values("graph_id")
        .reset_index(drop=True)
    )

    expected_graphs = set(cases["graph_id"])
    found_graphs = set(best_per_graph["graph_id"])

    if expected_graphs != found_graphs:
        missing = sorted(expected_graphs - found_graphs)
        raise ValueError(
            f"{candidate_pool_label}: missing best-of-K results for graphs: {missing}"
        )

    total_colors = int(best_per_graph["num_colors"].sum())
    oracle_total = int(best_per_graph["best_colors"].sum())
    gap = total_colors - oracle_total
    matched = int((best_per_graph["num_colors"] == best_per_graph["best_colors"]).sum())

    summary = pd.DataFrame(
        [
            {
                "method": candidate_pool_label,
                "candidate_pool": ", ".join(allowed_pools),
                "num_graphs": len(best_per_graph),
                "total_colors": total_colors,
                "gap_to_oracle": gap,
                "graphs_matching_oracle": matched,
                "oracle_total_colors": oracle_total,
            }
        ]
    )

    best_per_graph["best_of_k_method"] = candidate_pool_label

    return summary, best_per_graph


def main() -> None:
    cases = pd.read_csv(COMMON_CASES_CSV)

    candidates = load_gnn_candidates(cases)

    summary_parts = []

    summary_parts.append(fixed_heuristic_summary(cases))

    summary_parts.append(
        summarize_single_seed_gnn(
            candidates,
            cases,
            "GNN_COLOR_SELECTED",
        )
    )

    summary_parts.append(
        summarize_single_seed_gnn(
            candidates,
            cases,
            "GNN_LOSS_SELECTED",
        )
    )

    best_color_summary, best_color_per_graph = summarize_best_of_k(
        candidates,
        cases,
        "GNN_BEST_OF_5_COLOR_SELECTED_SEEDS",
        ["GNN_COLOR_SELECTED"],
    )

    best_loss_summary, best_loss_per_graph = summarize_best_of_k(
        candidates,
        cases,
        "GNN_BEST_OF_5_LOSS_SELECTED_SEEDS",
        ["GNN_LOSS_SELECTED"],
    )

    best_combined_summary, best_combined_per_graph = summarize_best_of_k(
        candidates,
        cases,
        "GNN_BEST_OF_10_COLOR_AND_LOSS_SELECTED",
        ["GNN_COLOR_SELECTED", "GNN_LOSS_SELECTED"],
    )

    summary_parts.extend(
        [
            best_color_summary,
            best_loss_summary,
            best_combined_summary,
        ]
    )

    summary = pd.concat(summary_parts, ignore_index=True)

    summary = summary.sort_values(
        by=["gap_to_oracle", "total_colors", "method"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    per_graph = pd.concat(
        [
            best_color_per_graph,
            best_loss_per_graph,
            best_combined_per_graph,
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SUMMARY_OUTPUT, index=False)
    per_graph.to_csv(PER_GRAPH_OUTPUT, index=False)

    print("Best-of-K GNN inference on heuristic-diversity common subset")
    print("------------------------------------------------------------")
    print()
    print(f"Graphs evaluated: {len(cases)}")
    print(f"Graph IDs: {sorted(cases['graph_id'].tolist())}")
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print("Best-of-K per graph:")
    columns_to_show = [
        "best_of_k_method",
        "graph_id",
        "candidate_id",
        "num_colors",
        "best_colors",
        "gap_to_oracle",
        "matches_oracle",
    ]
    print(per_graph[columns_to_show].to_string(index=False))
    print()
    print(f"Saved summary to: {SUMMARY_OUTPUT}")
    print(f"Saved per-graph results to: {PER_GRAPH_OUTPUT}")


if __name__ == "__main__":
    main()