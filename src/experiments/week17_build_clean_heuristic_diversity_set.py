from pathlib import Path
import pandas as pd


INPUT_CSV = Path(
    "results/tables/gnn_node_scorer/"
    "week17_colpack_heuristic_diversity_per_graph.csv"
)

OUTPUT_DIR = Path("results/tables/gnn_node_scorer")

CLEAN_OUTPUT_CSV = OUTPUT_DIR / "week17_clean_heuristic_diversity_cases.csv"
WINNER_COUNTS_OUTPUT_CSV = OUTPUT_DIR / "week17_clean_heuristic_diversity_winner_counts.csv"


SOURCE_PRIORITY = {
    "week17_structured_colpack_benchmark.csv": 80,
    "week17_heuristic_gap_colpack_benchmark.csv": 70,
    "colpack_week16_larger_graph_benchmark.csv": 60,
    "colpack_week15_five_ordering_benchmark.csv": 50,
    "colpack_week15_extra_orderings_benchmark.csv": 40,
    "colpack_week14_expanded_benchmark.csv": 30,
}


def source_priority(source_csv: str) -> int:
    return SOURCE_PRIORITY.get(source_csv, 0)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "source_csv",
        "graph_id",
        "num_orderings_tested",
        "best_colors",
        "worst_colors",
        "spread_best_to_worst",
        "winning_orderings",
        "worst_orderings",
        "has_nontrivial_spread",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df["source_priority"] = df["source_csv"].map(source_priority)

    # Keep only graphs where heuristic choice matters.
    candidates = df[df["spread_best_to_worst"] >= 2].copy()

    if candidates.empty:
        raise ValueError("No heuristic-diversity candidates found with spread >= 2.")

    # Remove duplicate graph IDs.
    # Prefer:
    # 1. More orderings tested
    # 2. Higher source priority
    # 3. Larger spread
    # 4. Lower best color count as final tie-breaker
    candidates = (
        candidates.sort_values(
            by=[
                "graph_id",
                "num_orderings_tested",
                "source_priority",
                "spread_best_to_worst",
                "best_colors",
            ],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates(subset=["graph_id"], keep="first")
        .reset_index(drop=True)
    )

    winner_rows = []

    for row in candidates.itertuples(index=False):
        winners = [w.strip() for w in row.winning_orderings.split(",")]

        for winner in winners:
            winner_rows.append(
                {
                    "ordering_name": winner,
                    "graph_id": row.graph_id,
                    "source_csv": row.source_csv,
                    "best_colors": row.best_colors,
                    "worst_colors": row.worst_colors,
                    "spread_best_to_worst": row.spread_best_to_worst,
                }
            )

    winner_df = pd.DataFrame(winner_rows)

    winner_counts = (
        winner_df.groupby("ordering_name")
        .agg(
            winner_or_tied_count=("graph_id", "count"),
            mean_spread_when_winning=("spread_best_to_worst", "mean"),
            max_spread_when_winning=("spread_best_to_worst", "max"),
        )
        .reset_index()
        .sort_values(
            ["winner_or_tied_count", "max_spread_when_winning"],
            ascending=False,
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(CLEAN_OUTPUT_CSV, index=False)
    winner_counts.to_csv(WINNER_COUNTS_OUTPUT_CSV, index=False)

    print("Clean heuristic-diversity candidate set")
    print("--------------------------------------")
    print()
    print(f"Input graph cases: {len(df)}")
    print(f"Candidates with spread >= 2 before deduplication: {len(df[df['spread_best_to_worst'] >= 2])}")
    print(f"Clean deduplicated candidates: {len(candidates)}")
    print()

    print("Winner counts in clean candidate set:")
    print(winner_counts.to_string(index=False))
    print()

    print("Clean candidates:")
    columns_to_show = [
        "source_csv",
        "graph_id",
        "best_colors",
        "worst_colors",
        "spread_best_to_worst",
        "winning_orderings",
        "worst_orderings",
    ]
    print(candidates[columns_to_show].to_string(index=False))
    print()

    print(f"Saved clean candidate set to: {CLEAN_OUTPUT_CSV}")
    print(f"Saved winner counts to: {WINNER_COUNTS_OUTPUT_CSV}")


if __name__ == "__main__":
    main()