from pathlib import Path
import pandas as pd


BENCHMARK_DIR = Path("results/tables/initial_graph_coloring_benchmarks")

OUTPUT_DIR = Path("results/tables/gnn_node_scorer")
PER_GRAPH_OUTPUT = OUTPUT_DIR / "week17_colpack_heuristic_diversity_per_graph.csv"
WINNER_COUNTS_OUTPUT = OUTPUT_DIR / "week17_colpack_heuristic_diversity_winner_counts.csv"
INTERESTING_OUTPUT = OUTPUT_DIR / "week17_colpack_heuristic_diversity_interesting_cases.csv"

ORDERINGS = {
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
}


def normalize_ordering(value: str) -> str:
    return str(value).strip().upper()


def load_colpack_rows() -> pd.DataFrame:
    rows = []

    csv_paths = sorted(BENCHMARK_DIR.glob("*.csv"))

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as error:
            print(f"Skipping unreadable CSV {path}: {error}")
            continue

        required = {"graph_id", "ordering_name", "num_colors"}
        if not required.issubset(df.columns):
            continue

        temp = df.copy()
        temp["source_csv"] = path.name
        temp["ordering_name"] = temp["ordering_name"].map(normalize_ordering)

        temp = temp[temp["ordering_name"].isin(ORDERINGS)]

        if temp.empty:
            continue

        rows.append(
            temp[
                [
                    "source_csv",
                    "graph_id",
                    "ordering_name",
                    "num_colors",
                ]
            ]
        )

    if not rows:
        raise ValueError(
            "No ColPack benchmark CSVs found with columns "
            "graph_id, ordering_name, num_colors."
        )

    return pd.concat(rows, ignore_index=True)


def summarize_per_graph(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    grouped = df.groupby(["source_csv", "graph_id"], sort=True)

    for (source_csv, graph_id), group in grouped:
        # If a CSV accidentally contains repeated rows for the same ordering,
        # keep the best color count for that ordering.
        by_ordering = (
            group.groupby("ordering_name")["num_colors"]
            .min()
            .reset_index()
        )

        if by_ordering["ordering_name"].nunique() < 2:
            continue

        best_colors = int(by_ordering["num_colors"].min())
        worst_colors = int(by_ordering["num_colors"].max())
        spread = worst_colors - best_colors

        winners = sorted(
            by_ordering[by_ordering["num_colors"] == best_colors][
                "ordering_name"
            ].tolist()
        )

        worst_orderings = sorted(
            by_ordering[by_ordering["num_colors"] == worst_colors][
                "ordering_name"
            ].tolist()
        )

        row = {
            "source_csv": source_csv,
            "graph_id": graph_id,
            "num_orderings_tested": int(by_ordering["ordering_name"].nunique()),
            "best_colors": best_colors,
            "worst_colors": worst_colors,
            "spread_best_to_worst": spread,
            "winner_count": len(winners),
            "winning_orderings": ", ".join(winners),
            "worst_orderings": ", ".join(worst_orderings),
            "has_unique_winner": len(winners) == 1,
            "has_nontrivial_spread": spread >= 2,
        }

        for ordering in sorted(ORDERINGS):
            match = by_ordering[by_ordering["ordering_name"] == ordering]
            if match.empty:
                row[ordering] = None
            else:
                row[ordering] = int(match.iloc[0]["num_colors"])

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_winner_counts(per_graph_df: pd.DataFrame) -> pd.DataFrame:
    winner_rows = []

    for row in per_graph_df.itertuples(index=False):
        winners = [w.strip() for w in row.winning_orderings.split(",")]

        for winner in winners:
            winner_rows.append(
                {
                    "ordering_name": winner,
                    "graph_id": row.graph_id,
                    "source_csv": row.source_csv,
                    "spread_best_to_worst": row.spread_best_to_worst,
                    "has_unique_winner": row.has_unique_winner,
                    "has_nontrivial_spread": row.has_nontrivial_spread,
                }
            )

    winner_df = pd.DataFrame(winner_rows)

    counts = (
        winner_df.groupby("ordering_name")
        .agg(
            winner_or_tied_count=("graph_id", "count"),
            unique_winner_count=("has_unique_winner", "sum"),
            nontrivial_spread_winner_count=("has_nontrivial_spread", "sum"),
        )
        .reset_index()
        .sort_values(
            ["winner_or_tied_count", "unique_winner_count"],
            ascending=False,
        )
    )

    return counts


def main() -> None:
    df = load_colpack_rows()
    per_graph_df = summarize_per_graph(df)

    if per_graph_df.empty:
        raise ValueError("No per-graph summaries could be built.")

    winner_counts = build_winner_counts(per_graph_df)

    interesting_df = (
        per_graph_df[
            per_graph_df["has_nontrivial_spread"]
        ]
        .sort_values(
            ["spread_best_to_worst", "graph_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_graph_df.to_csv(PER_GRAPH_OUTPUT, index=False)
    winner_counts.to_csv(WINNER_COUNTS_OUTPUT, index=False)
    interesting_df.to_csv(INTERESTING_OUTPUT, index=False)

    print("ColPack heuristic diversity scan")
    print("--------------------------------")
    print()
    print(f"Benchmark CSV rows scanned: {len(df)}")
    print(f"Graph cases summarized: {len(per_graph_df)}")
    print()

    print("Winner counts:")
    print(winner_counts.to_string(index=False))
    print()

    print("Spread summary:")
    print(per_graph_df["spread_best_to_worst"].describe().to_string())
    print()

    print("Number of cases with spread >= 2:")
    print(int(per_graph_df["has_nontrivial_spread"].sum()))
    print()

    print("Top interesting cases, spread >= 2:")
    if interesting_df.empty:
        print("No cases found with spread >= 2.")
    else:
        columns_to_show = [
            "source_csv",
            "graph_id",
            "best_colors",
            "worst_colors",
            "spread_best_to_worst",
            "winning_orderings",
            "worst_orderings",
        ]

        print(interesting_df[columns_to_show].head(30).to_string(index=False))

    print()
    print(f"Saved per-graph summary to: {PER_GRAPH_OUTPUT}")
    print(f"Saved winner counts to: {WINNER_COUNTS_OUTPUT}")
    print(f"Saved interesting cases to: {INTERESTING_OUTPUT}")


if __name__ == "__main__":
    main()