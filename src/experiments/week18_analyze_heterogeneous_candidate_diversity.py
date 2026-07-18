from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_colpack_summary.csv"
)

GAP2_POOL_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_gap2_candidate_pool.csv"
)

UNIQUE_WINNER_POOL_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_heterogeneous_gap2_unique_winner_pool.csv"
)


ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Week 18 ColPack summary not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "graph_id",
        "family",
        "ordering_gap",
        "ordering_gap_at_least_2",
        "num_best_orderings",
        "unique_best_ordering",
        "best_colpack5_orderings",
        "best_colpack5_colors",
        "worst_colpack5_colors",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    gap2_df = df[df["ordering_gap"] >= 2].copy()

    unique_gap2_df = gap2_df[
        gap2_df["num_best_orderings"] == 1
    ].copy()

    gap2_df = gap2_df.sort_values(
        by=[
            "ordering_gap",
            "family",
            "graph_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    unique_gap2_df = unique_gap2_df.sort_values(
        by=[
            "unique_best_ordering",
            "ordering_gap",
            "family",
            "graph_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    GAP2_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)

    gap2_df.to_csv(
        GAP2_POOL_PATH,
        index=False,
    )

    unique_gap2_df.to_csv(
        UNIQUE_WINNER_POOL_PATH,
        index=False,
    )

    print("Week 18 heterogeneous candidate-diversity analysis")
    print("--------------------------------------------------")
    print(f"All candidates: {len(df)}")
    print(f"Candidates with ordering gap >= 2: {len(gap2_df)}")
    print(
        "Gap >= 2 candidates with a unique winner: "
        f"{len(unique_gap2_df)}"
    )
    print(
        "Gap >= 2 candidates with tied winners: "
        f"{len(gap2_df) - len(unique_gap2_df)}"
    )
    print()

    print("Gap >= 2 candidates by family:")

    family_counts = (
        gap2_df
        .groupby("family")
        .size()
        .sort_values(ascending=False)
    )

    for family, count in family_counts.items():
        print(f"  {family}: {count}")

    print()
    print("Unique-winner, gap >= 2 candidates by heuristic:")

    unique_counts = Counter(
        unique_gap2_df["unique_best_ordering"]
        .dropna()
        .astype(str)
    )

    for ordering in ORDERINGS:
        print(f"  {ordering}: {unique_counts.get(ordering, 0)}")

    print()
    print("Unique-winner, gap >= 2 candidates by family and heuristic:")

    if unique_gap2_df.empty:
        print("  No candidates found.")
    else:
        pivot = pd.crosstab(
            unique_gap2_df["family"],
            unique_gap2_df["unique_best_ordering"],
        )

        pivot = pivot.reindex(
            columns=ORDERINGS,
            fill_value=0,
        )

        print(pivot.to_string())

    print()
    print("Ordering-gap distribution:")

    gap_counts = (
        gap2_df["ordering_gap"]
        .value_counts()
        .sort_index()
    )

    for gap, count in gap_counts.items():
        print(f"  gap={int(gap)}: {int(count)}")

    print()
    print("Maximum gap by family:")

    maximum_gap_by_family = (
        gap2_df
        .groupby("family")["ordering_gap"]
        .max()
        .sort_values(ascending=False)
    )

    for family, maximum_gap in maximum_gap_by_family.items():
        print(f"  {family}: {int(maximum_gap)}")

    print()
    print("Most common tied-winner combinations for gap >= 2:")

    tied_gap2_df = gap2_df[
        gap2_df["num_best_orderings"] > 1
    ]

    tied_counts = Counter(
        tied_gap2_df["best_colpack5_orderings"]
        .dropna()
        .astype(str)
    )

    if tied_counts:
        for winners, count in tied_counts.most_common(10):
            print(f"  {winners}: {count}")
    else:
        print("  No tied-winner cases.")

    print()
    print(f"Saved gap >= 2 pool to: {GAP2_POOL_PATH}")
    print(
        "Saved unique-winner gap >= 2 pool to: "
        f"{UNIQUE_WINNER_POOL_PATH}"
    )


if __name__ == "__main__":
    main()