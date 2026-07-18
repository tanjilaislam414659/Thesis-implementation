from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_heterogeneous_balanced_split.csv"
)

OUTPUT_ASSIGNMENT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_training_size_ablation_assignment.csv"
)

OUTPUT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_training_size_ablation_summary.csv"
)

OUTPUT_GROUP_AUDIT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_training_size_ablation_group_audit.csv"
)


TEACHERS = [
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]

ENTRY_TIERS = [12, 24, 36, 48]

# Each incremental tier adds 12 graphs:
# three graphs for each teacher.
TIER_TEACHER_QUOTA = {
    tier: (3, 3, 3, 3)
    for tier in ENTRY_TIERS
}

CUMULATIVE_TEACHER_COUNT = {
    12: 3,
    24: 6,
    36: 9,
    48: 12,
}

EXPECTED_CUMULATIVE_FAMILY_COUNTS = {
    12: {
        "barabasi_albert": 1,
        "crown": 1,
        "erdos_renyi": 8,
        "stochastic_block_model": 2,
    },
    24: {
        "barabasi_albert": 1,
        "crown": 2,
        "erdos_renyi": 17,
        "stochastic_block_model": 4,
    },
    36: {
        "barabasi_albert": 1,
        "crown": 3,
        "erdos_renyi": 28,
        "stochastic_block_model": 4,
    },
    48: {
        "barabasi_albert": 1,
        "crown": 5,
        "erdos_renyi": 37,
        "stochastic_block_model": 5,
    },
}

# Fixed non-ER progression.
#
# Crown graphs enter in increasing size:
# 20 vertices -> 24 -> 28 -> 32 and 36.
FIXED_GROUP_TIERS = {
    "barabasi_albert:week18_ba_n120_m6": 12,

    "crown:week18_crown_m10_alternating": 12,
    "crown:week18_crown_m12_alternating": 24,
    "crown:week18_crown_m14_alternating": 36,
    "crown:week18_crown_m16_alternating": 48,
    "crown:week18_crown_m18_alternating": 48,

    (
        "stochastic_block_model:"
        "week18_sbm_three_blocks_120_strong"
    ): 12,

    (
        "stochastic_block_model:"
        "week18_sbm_three_blocks_90_weak"
    ): 24,

    (
        "stochastic_block_model:"
        "week18_sbm_two_blocks_80_weak"
    ): 48,
}


def subtract_vectors(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        first - second
        for first, second in zip(left, right)
    )


def vector_fits(
    values: tuple[int, ...],
    capacity: tuple[int, ...],
) -> bool:
    return all(
        value <= limit
        for value, limit in zip(values, capacity)
    )


def vector_is_zero(
    values: tuple[int, ...],
) -> bool:
    return all(value == 0 for value in values)


def load_training_split() -> pd.DataFrame:
    if not INPUT_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Week 18 split file not found: "
            f"{INPUT_SPLIT_CSV}"
        )

    split_df = pd.read_csv(INPUT_SPLIT_CSV)

    required_columns = {
        "graph_id",
        "split",
        "split_group_id",
        "family",
        "num_vertices",
        "num_edges",
        "selected_teacher_ordering",
        "matrix_path",
    }

    missing_columns = (
        required_columns
        - set(split_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Split CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    train_df = (
        split_df[
            split_df["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train_df) != 48:
        raise ValueError(
            f"Expected 48 training graphs, "
            f"found {len(train_df)}."
        )

    if train_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in training split."
        )

    teacher_counts = (
        train_df["selected_teacher_ordering"]
        .value_counts()
    )

    for teacher in TEACHERS:
        actual = int(
            teacher_counts.get(teacher, 0)
        )

        if actual != 12:
            raise ValueError(
                f"{teacher}: expected 12 graphs, "
                f"found {actual}."
            )

    return train_df


def build_groups(
    train_df: pd.DataFrame,
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []

    for group_id, group_df in train_df.groupby(
        "split_group_id",
        sort=True,
    ):
        families = (
            group_df["family"]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(families) != 1:
            raise ValueError(
                f"Group {group_id} contains multiple "
                f"families: {families}"
            )

        family = str(families[0])

        teacher_counts = tuple(
            int(
                (
                    group_df["selected_teacher_ordering"]
                    == teacher
                ).sum()
            )
            for teacher in TEACHERS
        )

        groups.append(
            {
                "group_id": str(group_id),
                "family": family,
                "teacher_counts": teacher_counts,
                "group_size": int(len(group_df)),
                "graph_ids": (
                    group_df["graph_id"]
                    .astype(str)
                    .tolist()
                ),
                "minimum_vertices": int(
                    group_df["num_vertices"].min()
                ),
                "maximum_vertices": int(
                    group_df["num_vertices"].max()
                ),
            }
        )

    actual_group_ids = {
        str(group["group_id"])
        for group in groups
    }

    missing_fixed_groups = (
        set(FIXED_GROUP_TIERS)
        - actual_group_ids
    )

    if missing_fixed_groups:
        raise ValueError(
            "Fixed groups are missing from the training split: "
            f"{sorted(missing_fixed_groups)}"
        )

    for group in groups:
        group_id = str(group["group_id"])
        family = str(group["family"])

        if (
            family != "erdos_renyi"
            and group_id not in FIXED_GROUP_TIERS
        ):
            raise ValueError(
                f"Non-ER group lacks a fixed tier: {group_id}"
            )

    return groups


def find_er_assignment(
    groups: list[dict[str, object]],
    seed: int,
    node_limit: int = 2_000_000,
) -> dict[str, int] | None:
    """
    Apply all fixed non-ER assignments first, then assign complete
    Erdős–Rényi replicate groups so that every incremental tier has
    exactly three graphs from each teacher.
    """
    rng = random.Random(seed)

    assignment: dict[str, int] = {}

    remaining_capacity = {
        tier: TIER_TEACHER_QUOTA[tier]
        for tier in ENTRY_TIERS
    }

    # Apply fixed BA, crown and SBM groups.
    for group in groups:
        group_id = str(group["group_id"])

        if group_id not in FIXED_GROUP_TIERS:
            continue

        tier = FIXED_GROUP_TIERS[group_id]
        counts = tuple(group["teacher_counts"])

        if not vector_fits(
            counts,
            remaining_capacity[tier],
        ):
            return None

        remaining_capacity[tier] = subtract_vectors(
            remaining_capacity[tier],
            counts,
        )

        assignment[group_id] = tier

    er_groups = [
        group
        for group in groups
        if str(group["family"]) == "erdos_renyi"
    ]

    rng.shuffle(er_groups)

    er_groups.sort(
        key=lambda group: (
            int(group["group_size"]),
            max(tuple(group["teacher_counts"])),
            sum(
                value > 0
                for value in tuple(
                    group["teacher_counts"]
                )
            ),
            int(group["maximum_vertices"]),
        ),
        reverse=True,
    )

    memo: set[tuple[object, ...]] = set()
    visited_nodes = 0

    def recurse(
        index: int,
        capacity_12: tuple[int, ...],
        capacity_24: tuple[int, ...],
        capacity_36: tuple[int, ...],
        capacity_48: tuple[int, ...],
    ) -> bool:
        nonlocal visited_nodes

        visited_nodes += 1

        if visited_nodes > node_limit:
            return False

        if index == len(er_groups):
            return (
                vector_is_zero(capacity_12)
                and vector_is_zero(capacity_24)
                and vector_is_zero(capacity_36)
                and vector_is_zero(capacity_48)
            )

        state = (
            index,
            capacity_12,
            capacity_24,
            capacity_36,
            capacity_48,
        )

        if state in memo:
            return False

        group = er_groups[index]
        group_id = str(group["group_id"])
        counts = tuple(group["teacher_counts"])

        capacities = {
            12: capacity_12,
            24: capacity_24,
            36: capacity_36,
            48: capacity_48,
        }

        eligible_tiers = [
            tier
            for tier in ENTRY_TIERS
            if vector_fits(
                counts,
                capacities[tier],
            )
        ]

        if not eligible_tiers:
            memo.add(state)
            return False

        rng.shuffle(eligible_tiers)

        # Prefer tighter remaining capacities.
        eligible_tiers.sort(
            key=lambda tier: (
                sum(
                    subtract_vectors(
                        capacities[tier],
                        counts,
                    )
                ),
                tier,
            )
        )

        for tier in eligible_tiers:
            next_capacities = dict(capacities)

            next_capacities[tier] = subtract_vectors(
                capacities[tier],
                counts,
            )

            assignment[group_id] = tier

            solved = recurse(
                index=index + 1,
                capacity_12=next_capacities[12],
                capacity_24=next_capacities[24],
                capacity_36=next_capacities[36],
                capacity_48=next_capacities[48],
            )

            if solved:
                return True

            assignment.pop(group_id, None)

        memo.add(state)
        return False

    solved = recurse(
        index=0,
        capacity_12=remaining_capacity[12],
        capacity_24=remaining_capacity[24],
        capacity_36=remaining_capacity[36],
        capacity_48=remaining_capacity[48],
    )

    return assignment if solved else None


def expand_assignment(
    train_df: pd.DataFrame,
    group_assignment: dict[str, int],
) -> pd.DataFrame:
    output = train_df.copy()

    output["minimum_training_size"] = (
        output["split_group_id"]
        .map(group_assignment)
    )

    if output["minimum_training_size"].isna().any():
        missing = output.loc[
            output["minimum_training_size"].isna(),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Missing ablation assignments for: {missing}"
        )

    output["minimum_training_size"] = (
        output["minimum_training_size"]
        .astype(int)
    )

    output["ablation_entry_tier"] = (
        output["minimum_training_size"]
        .apply(
            lambda value: f"added_at_{value}"
        )
    )

    for training_size in ENTRY_TIERS:
        output[
            f"included_in_train_{training_size}"
        ] = (
            output["minimum_training_size"]
            <= training_size
        )

    return output


def score_assignment(
    assignment_df: pd.DataFrame,
) -> float:
    """
    Prefer broad ER size coverage inside every incremental tier.
    """
    score = 0.0

    er_df = assignment_df[
        assignment_df["family"] == "erdos_renyi"
    ]

    overall_mean_vertices = float(
        er_df["num_vertices"].mean()
    )

    for tier in ENTRY_TIERS:
        tier_er = er_df[
            er_df["minimum_training_size"] == tier
        ]

        if tier_er.empty:
            return float("-inf")

        unique_sizes = int(
            tier_er["num_vertices"].nunique()
        )

        tier_mean_vertices = float(
            tier_er["num_vertices"].mean()
        )

        score += 100.0 * unique_sizes

        score -= abs(
            tier_mean_vertices
            - overall_mean_vertices
        )

        # Reward both small and large ER examples in each tier.
        if (
            tier_er["num_vertices"] <= 60
        ).any():
            score += 50.0

        if (
            tier_er["num_vertices"] >= 100
        ).any():
            score += 50.0

    return score


def validate_assignment(
    assignment_df: pd.DataFrame,
) -> None:
    if len(assignment_df) != 48:
        raise ValueError(
            f"Expected 48 assigned graphs, "
            f"found {len(assignment_df)}."
        )

    if assignment_df["graph_id"].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found."
        )

    group_tier_counts = (
        assignment_df
        .groupby("split_group_id")[
            "minimum_training_size"
        ]
        .nunique()
    )

    leaked_groups = group_tier_counts[
        group_tier_counts > 1
    ]

    if not leaked_groups.empty:
        raise ValueError(
            "Replicate groups were divided across tiers: "
            f"{leaked_groups.index.tolist()}"
        )

    for group_id, expected_tier in (
        FIXED_GROUP_TIERS.items()
    ):
        rows = assignment_df[
            assignment_df["split_group_id"]
            == group_id
        ]

        if rows.empty:
            raise ValueError(
                f"Fixed group missing: {group_id}"
            )

        actual_tiers = set(
            rows["minimum_training_size"]
            .astype(int)
        )

        if actual_tiers != {expected_tier}:
            raise ValueError(
                f"{group_id}: expected tier "
                f"{expected_tier}, found "
                f"{sorted(actual_tiers)}."
            )

    for training_size in ENTRY_TIERS:
        subset = assignment_df[
            assignment_df["minimum_training_size"]
            <= training_size
        ]

        if len(subset) != training_size:
            raise ValueError(
                f"Training size {training_size}: expected "
                f"{training_size} graphs, found {len(subset)}."
            )

        expected_teacher_count = (
            CUMULATIVE_TEACHER_COUNT[
                training_size
            ]
        )

        teacher_counts = (
            subset["selected_teacher_ordering"]
            .value_counts()
        )

        for teacher in TEACHERS:
            actual = int(
                teacher_counts.get(teacher, 0)
            )

            if actual != expected_teacher_count:
                raise ValueError(
                    f"Training size {training_size}, "
                    f"{teacher}: expected "
                    f"{expected_teacher_count}, "
                    f"found {actual}."
                )

        family_counts = (
            subset["family"]
            .value_counts()
            .to_dict()
        )

        expected_family_counts = (
            EXPECTED_CUMULATIVE_FAMILY_COUNTS[
                training_size
            ]
        )

        for family, expected_count in (
            expected_family_counts.items()
        ):
            actual_count = int(
                family_counts.get(family, 0)
            )

            if actual_count != expected_count:
                raise ValueError(
                    f"Training size {training_size}, "
                    f"{family}: expected {expected_count}, "
                    f"found {actual_count}."
                )

    crown_df = (
        assignment_df[
            assignment_df["family"] == "crown"
        ]
        .sort_values("num_vertices")
    )

    expected_crown_progression = {
        20: 12,
        24: 24,
        28: 36,
        32: 48,
        36: 48,
    }

    actual_crown_progression = {
        int(row.num_vertices):
        int(row.minimum_training_size)
        for row in crown_df.itertuples(index=False)
    }

    if (
        actual_crown_progression
        != expected_crown_progression
    ):
        raise ValueError(
            "Unexpected crown progression.\n"
            f"Expected: {expected_crown_progression}\n"
            f"Found: {actual_crown_progression}"
        )


def build_summary(
    assignment_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for training_size in ENTRY_TIERS:
        subset = assignment_df[
            assignment_df["minimum_training_size"]
            <= training_size
        ]

        teacher_counts = (
            subset["selected_teacher_ordering"]
            .value_counts()
        )

        family_counts = (
            subset["family"]
            .value_counts()
        )

        crown_rows = subset[
            subset["family"] == "crown"
        ]

        rows.append(
            {
                "training_size": training_size,
                "num_graphs": len(subset),
                "num_replicate_groups": (
                    subset["split_group_id"]
                    .nunique()
                ),
                "minimum_vertices": int(
                    subset["num_vertices"].min()
                ),
                "maximum_vertices": int(
                    subset["num_vertices"].max()
                ),
                "LARGEST_FIRST": int(
                    teacher_counts.get(
                        "LARGEST_FIRST", 0
                    )
                ),
                "DYNAMIC_LARGEST_FIRST": int(
                    teacher_counts.get(
                        "DYNAMIC_LARGEST_FIRST", 0
                    )
                ),
                "INCIDENCE_DEGREE": int(
                    teacher_counts.get(
                        "INCIDENCE_DEGREE", 0
                    )
                ),
                "SMALLEST_LAST": int(
                    teacher_counts.get(
                        "SMALLEST_LAST", 0
                    )
                ),
                "barabasi_albert": int(
                    family_counts.get(
                        "barabasi_albert", 0
                    )
                ),
                "crown": int(
                    family_counts.get(
                        "crown", 0
                    )
                ),
                "erdos_renyi": int(
                    family_counts.get(
                        "erdos_renyi", 0
                    )
                ),
                "stochastic_block_model": int(
                    family_counts.get(
                        "stochastic_block_model", 0
                    )
                ),
                "maximum_training_crown_vertices": (
                    int(
                        crown_rows[
                            "num_vertices"
                        ].max()
                    )
                    if not crown_rows.empty
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_group_audit(
    assignment_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group_id, group_df in assignment_df.groupby(
        "split_group_id",
        sort=True,
    ):
        teacher_counter = Counter(
            group_df["selected_teacher_ordering"]
        )

        rows.append(
            {
                "split_group_id": group_id,
                "minimum_training_size": int(
                    group_df.iloc[0][
                        "minimum_training_size"
                    ]
                ),
                "family": str(
                    group_df.iloc[0]["family"]
                ),
                "num_graphs": len(group_df),
                "graph_ids": "; ".join(
                    sorted(
                        group_df["graph_id"]
                        .astype(str)
                        .tolist()
                    )
                ),
                "minimum_vertices": int(
                    group_df["num_vertices"].min()
                ),
                "maximum_vertices": int(
                    group_df["num_vertices"].max()
                ),
                "LARGEST_FIRST": (
                    teacher_counter[
                        "LARGEST_FIRST"
                    ]
                ),
                "DYNAMIC_LARGEST_FIRST": (
                    teacher_counter[
                        "DYNAMIC_LARGEST_FIRST"
                    ]
                ),
                "INCIDENCE_DEGREE": (
                    teacher_counter[
                        "INCIDENCE_DEGREE"
                    ]
                ),
                "SMALLEST_LAST": (
                    teacher_counter[
                        "SMALLEST_LAST"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    train_df = load_training_split()
    groups = build_groups(train_df)

    best_assignment_df: pd.DataFrame | None = None
    best_score: float | None = None
    valid_solution_count = 0

    for attempt in range(500):
        group_assignment = find_er_assignment(
            groups=groups,
            seed=18_000 + attempt,
        )

        if group_assignment is None:
            continue

        candidate_df = expand_assignment(
            train_df=train_df,
            group_assignment=group_assignment,
        )

        validate_assignment(candidate_df)

        candidate_score = score_assignment(
            candidate_df
        )

        valid_solution_count += 1

        if (
            best_assignment_df is None
            or best_score is None
            or candidate_score > best_score
        ):
            best_assignment_df = candidate_df
            best_score = candidate_score

    print(
        "Valid controlled ablation solutions found: "
        f"{valid_solution_count}"
    )

    if (
        best_assignment_df is None
        or best_score is None
    ):
        raise RuntimeError(
            "Could not assign the Erdős–Rényi replicate "
            "groups while preserving exact teacher quotas."
        )

    validate_assignment(best_assignment_df)

    best_assignment_df = (
        best_assignment_df
        .sort_values(
            [
                "minimum_training_size",
                "family",
                "selected_teacher_ordering",
                "graph_id",
            ]
        )
        .reset_index(drop=True)
    )

    summary_df = build_summary(
        best_assignment_df
    )

    group_audit_df = build_group_audit(
        best_assignment_df
    )

    OUTPUT_ASSIGNMENT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_assignment_df.to_csv(
        OUTPUT_ASSIGNMENT_CSV,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    group_audit_df.to_csv(
        OUTPUT_GROUP_AUDIT_CSV,
        index=False,
    )

    print()
    print(
        "Week 18 controlled training-size ablation subsets"
    )
    print(
        "-------------------------------------------------"
    )
    print(
        f"Selected solution score: {best_score:.2f}"
    )
    print()

    print("Cumulative subset summary:")
    print(
        summary_df.to_string(index=False)
    )

    print()
    print("Crown entry progression:")

    crown_display = (
        best_assignment_df[
            best_assignment_df["family"] == "crown"
        ][
            [
                "graph_id",
                "num_vertices",
                "selected_teacher_ordering",
                "minimum_training_size",
            ]
        ]
        .sort_values("num_vertices")
    )

    print(
        crown_display.to_string(index=False)
    )

    print()
    print("Incremental family additions:")

    incremental_family = pd.crosstab(
        best_assignment_df[
            "minimum_training_size"
        ],
        best_assignment_df["family"],
    ).reindex(
        index=ENTRY_TIERS,
        fill_value=0,
    )

    print(
        incremental_family.to_string()
    )

    print()
    print(
        "All subsets are nested."
    )
    print(
        "All replicate groups remain intact."
    )
    print(
        "Every cumulative subset has exact teacher balance."
    )
    print(
        "Crown graphs enter progressively from smaller "
        "to larger sizes."
    )
    print()
    print(
        f"Saved assignment to: "
        f"{OUTPUT_ASSIGNMENT_CSV}"
    )
    print(
        f"Saved summary to: "
        f"{OUTPUT_SUMMARY_CSV}"
    )
    print(
        f"Saved group audit to: "
        f"{OUTPUT_GROUP_AUDIT_CSV}"
    )


if __name__ == "__main__":
    main()