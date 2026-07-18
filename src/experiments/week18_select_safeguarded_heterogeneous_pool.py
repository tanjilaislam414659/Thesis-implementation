from __future__ import annotations

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

SELECTED_POOL_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_safeguarded_heterogeneous_pool.csv"
)

SELECTION_AUDIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_safeguarded_pool_selection_audit.csv"
)


def choose_crown_representative(
    crown_group: pd.DataFrame,
) -> str:
    """
    Select one crown graph for a given partition size.

    Priority:
    1. Larger heuristic gap.
    2. Fewer tied best orderings.
    3. Deterministic labeling preference.
    4. Graph ID as a final stable tie-break.
    """
    labeling_priority = {
        "random": 0,
        "alternating": 1,
        "blocked": 2,
    }

    ranked = crown_group.copy()

    ranked["labeling_priority"] = (
        ranked["labeling"]
        .map(labeling_priority)
        .fillna(99)
        .astype(int)
    )

    ranked = ranked.sort_values(
        by=[
            "ordering_gap",
            "num_best_orderings",
            "labeling_priority",
            "graph_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    )

    return str(ranked.iloc[0]["graph_id"])


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Week 18 ColPack summary not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "graph_id",
        "family",
        "labeling",
        "parameter_1_name",
        "parameter_1_value",
        "ordering_gap",
        "num_best_orderings",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["ordering_gap"] = pd.to_numeric(
        df["ordering_gap"],
        errors="raise",
    )

    df["num_best_orderings"] = pd.to_numeric(
        df["num_best_orderings"],
        errors="raise",
    )

    df["is_selected"] = False
    df["selection_reason"] = ""
    df["deduplication_group"] = ""

    # First exclude all candidates without a meaningful heuristic gap.
    no_gap_mask = df["ordering_gap"] < 2

    df.loc[
        no_gap_mask,
        "selection_reason",
    ] = "excluded_ordering_gap_below_2"

    # Keep every useful Erdos-Renyi graph.
    er_mask = (
        (df["family"] == "erdos_renyi")
        & (df["ordering_gap"] >= 2)
    )

    df.loc[er_mask, "is_selected"] = True

    df.loc[
        er_mask,
        "selection_reason",
    ] = "selected_distinct_erdos_renyi_instance"

    df.loc[
        er_mask,
        "deduplication_group",
    ] = df.loc[er_mask, "graph_id"]

    # Crown variants with the same partition size represent the same
    # unlabeled graph structure. Keep only one representative per size.
    crown_pool = df[
        (df["family"] == "crown")
        & (df["ordering_gap"] >= 2)
    ].copy()

    if not crown_pool.empty:
        crown_pool["partition_size"] = pd.to_numeric(
            crown_pool["parameter_1_value"],
            errors="raise",
        ).astype(int)

        for partition_size, group in crown_pool.groupby(
            "partition_size",
            sort=True,
        ):
            selected_graph_id = choose_crown_representative(group)

            group_mask = df["graph_id"].isin(group["graph_id"])

            df.loc[
                group_mask,
                "deduplication_group",
            ] = f"crown_partition_size_{partition_size}"

            selected_mask = df["graph_id"] == selected_graph_id

            df.loc[selected_mask, "is_selected"] = True

            df.loc[
                selected_mask,
                "selection_reason",
            ] = (
                "selected_crown_representative_"
                "largest_gap_then_fewest_ties"
            )

            excluded_group_mask = (
                group_mask
                & ~selected_mask
            )

            df.loc[
                excluded_group_mask,
                "selection_reason",
            ] = (
                "excluded_duplicate_crown_structure_"
                f"representative_is_{selected_graph_id}"
            )

    # Any future family with gap >= 2 must be considered explicitly
    # rather than entering the dataset silently.
    unhandled_mask = (
        (df["ordering_gap"] >= 2)
        & (df["selection_reason"] == "")
    )

    df.loc[
        unhandled_mask,
        "selection_reason",
    ] = "excluded_family_not_yet_approved"

    selected_df = df[df["is_selected"]].copy()

    selected_df = selected_df.sort_values(
        by=[
            "family",
            "parameter_1_value",
            "graph_id",
        ],
    )

    audit_df = df.sort_values(
        by=[
            "is_selected",
            "family",
            "parameter_1_value",
            "graph_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    )

    SELECTED_POOL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_df.to_csv(
        SELECTED_POOL_PATH,
        index=False,
    )

    audit_df.to_csv(
        SELECTION_AUDIT_PATH,
        index=False,
    )

    selected_crowns = selected_df[
        selected_df["family"] == "crown"
    ]

    selected_er = selected_df[
        selected_df["family"] == "erdos_renyi"
    ]

    duplicate_crown_sizes = (
        selected_crowns["parameter_1_value"]
        .duplicated()
        .sum()
    )

    if duplicate_crown_sizes != 0:
        raise ValueError(
            "Safeguard failed: more than one crown graph was "
            "selected for a partition size."
        )

    print("Week 18 safeguarded heterogeneous pool selection")
    print("------------------------------------------------")
    print(f"Original candidates: {len(df)}")
    print(f"Selected graphs: {len(selected_df)}")
    print(f"Excluded graphs: {len(df) - len(selected_df)}")
    print()

    print("Selected graphs by family:")
    print(f"  erdos_renyi: {len(selected_er)}")
    print(f"  crown: {len(selected_crowns)}")
    print()

    print("Selected crown representatives:")

    for row in selected_crowns.itertuples(index=False):
        print(
            f"  partition_size={int(float(row.parameter_1_value))}: "
            f"{row.graph_id}, "
            f"gap={int(row.ordering_gap)}, "
            f"best={row.best_colpack5_orderings}"
        )

    print()
    print(
        "Duplicate selected crown partition sizes: "
        f"{duplicate_crown_sizes}"
    )
    print()
    print(f"Saved selected pool to: {SELECTED_POOL_PATH}")
    print(f"Saved selection audit to: {SELECTION_AUDIT_PATH}")


if __name__ == "__main__":
    main()