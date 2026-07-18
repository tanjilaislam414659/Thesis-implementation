from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_balanced_best_of_5_teacher_assignments.csv"
)

SPLIT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_heterogeneous_balanced_split.csv"
)

SPLIT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_balanced_split_summary.csv"
)

GROUP_AUDIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_heterogeneous_split_group_audit.csv"
)


TEACHERS = [
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]

SPLITS = [
    "train",
    "validation",
    "test",
]

TEACHER_QUOTAS = {
    "train": (12, 12, 12, 12),
    "validation": (2, 2, 2, 2),
    "test": (3, 3, 3, 3),
}

EXPECTED_SPLIT_SIZES = {
    "train": 48,
    "validation": 8,
    "test": 12,
}


# Fixed size-aware crown allocation:
#
# Training:   smaller crowns, 20–36 vertices
# Validation: medium crowns, 40–44 vertices
# Test:       larger crowns, 48–60 vertices
FIXED_CROWN_SPLITS = {
    "week18_crown_m10_alternating": "train",
    "week18_crown_m12_alternating": "train",
    "week18_crown_m14_alternating": "train",
    "week18_crown_m16_alternating": "train",
    "week18_crown_m18_alternating": "train",

    "week18_crown_m20_alternating": "validation",
    "week18_crown_m22_alternating": "validation",

    "week18_crown_m24_alternating": "test",
    "week18_crown_m26_alternating": "test",
    "week18_crown_m28_alternating": "test",
    "week18_crown_m30_alternating": "test",
}


def remove_seed_suffix(graph_id: str) -> str:
    """
    Remove a final replicate suffix such as _s0, _s1 or _s2.
    """
    return re.sub(r"_s\d+$", "", graph_id)


def build_group_id(
    graph_id: str,
    family: str,
) -> str:
    """
    Build strict replicate groups.

    Replicates from the same generator configuration must remain
    in the same dataset split.
    """
    replicate_families = {
        "erdos_renyi",
        "barabasi_albert",
        "watts_strogatz",
        "stochastic_block_model",
    }

    if family in replicate_families:
        return f"{family}:{remove_seed_suffix(graph_id)}"

    # Retained crown graphs are already structurally distinct by size.
    return f"{family}:{graph_id}"


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


def build_groups(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    Combine graph replicates into indivisible split groups.
    """
    working = df.copy()

    working["split_group_id"] = working.apply(
        lambda row: build_group_id(
            graph_id=str(row["graph_id"]),
            family=str(row["family"]),
        ),
        axis=1,
    )

    groups: list[dict[str, object]] = []

    for group_id, group_df in working.groupby(
        "split_group_id",
        sort=True,
    ):
        teacher_counts = tuple(
            int(
                (
                    group_df["selected_teacher_ordering"]
                    == teacher
                ).sum()
            )
            for teacher in TEACHERS
        )

        fixed_splits = {
            FIXED_CROWN_SPLITS[graph_id]
            for graph_id in group_df["graph_id"].astype(str)
            if graph_id in FIXED_CROWN_SPLITS
        }

        if len(fixed_splits) > 1:
            raise ValueError(
                f"Group {group_id} contains conflicting fixed "
                f"split requirements: {fixed_splits}"
            )

        fixed_split: str | None = None

        if fixed_splits:
            fixed_split = next(iter(fixed_splits))

        # The single useful Barabási–Albert graph remains in training.
        if (
            "barabasi_albert"
            in set(group_df["family"].astype(str))
        ):
            if (
                fixed_split is not None
                and fixed_split != "train"
            ):
                raise ValueError(
                    f"Barabási–Albert group {group_id} has a "
                    f"conflicting fixed assignment."
                )

            fixed_split = "train"

        groups.append(
            {
                "group_id": str(group_id),
                "graph_ids": (
                    group_df["graph_id"]
                    .astype(str)
                    .tolist()
                ),
                "families": frozenset(
                    group_df["family"]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                "teacher_counts": teacher_counts,
                "group_size": len(group_df),
                "maximum_vertices": int(
                    group_df["num_vertices"].max()
                ),
                "fixed_split": fixed_split,
            }
        )

    return groups


def family_requirements_are_met(
    train_families: frozenset[str],
    validation_families: frozenset[str],
    test_families: frozenset[str],
) -> bool:
    """
    Check final graph-family safeguards.
    """
    # Training must genuinely be heterogeneous.
    if len(train_families) < 4:
        return False

    # Validation and test must each contain at least three families.
    if len(validation_families) < 3:
        return False

    if len(test_families) < 3:
        return False

    # Erdős–Rényi must appear in every split.
    if "erdos_renyi" not in train_families:
        return False

    if "erdos_renyi" not in validation_families:
        return False

    if "erdos_renyi" not in test_families:
        return False

    # Fixed crown allocation should guarantee this, but verify it.
    if "crown" not in train_families:
        return False

    if "crown" not in validation_families:
        return False

    if "crown" not in test_families:
        return False

    return True


def family_processing_priority(
    families: frozenset[str],
) -> int:
    """
    Process rarer graph families earlier during backtracking.
    """
    if "watts_strogatz" in families:
        return 5

    if "stochastic_block_model" in families:
        return 4

    if "barabasi_albert" in families:
        return 3

    if "crown" in families:
        return 2

    if "erdos_renyi" in families:
        return 1

    return 0


def find_group_assignment(
    groups: list[dict[str, object]],
    seed: int,
    node_limit: int = 5_000_000,
) -> dict[str, str] | None:
    """
    Search for an exact split satisfying:

    - fixed size-aware crown allocation;
    - exact teacher quotas;
    - strict replicate grouping;
    - Barabási–Albert in training;
    - family diversity in all splits.
    """
    rng = random.Random(seed)

    assignment: dict[str, str] = {}

    remaining_capacity = {
        split: TEACHER_QUOTAS[split]
        for split in SPLITS
    }

    split_families = {
        "train": frozenset(),
        "validation": frozenset(),
        "test": frozenset(),
    }

    # Apply all fixed assignments first.
    for group in groups:
        fixed_split = group["fixed_split"]

        if fixed_split is None:
            continue

        fixed_split = str(fixed_split)
        group_id = str(group["group_id"])
        counts = tuple(group["teacher_counts"])
        families = frozenset(group["families"])

        if not vector_fits(
            counts,
            remaining_capacity[fixed_split],
        ):
            return None

        remaining_capacity[fixed_split] = subtract_vectors(
            remaining_capacity[fixed_split],
            counts,
        )

        split_families[fixed_split] = (
            split_families[fixed_split]
            | families
        )

        assignment[group_id] = fixed_split

    remaining_groups = [
        group
        for group in groups
        if group["fixed_split"] is None
    ]

    # Randomize equivalent groups between attempts.
    rng.shuffle(remaining_groups)

    # Process rare, large and constrained groups first.
    remaining_groups.sort(
        key=lambda group: (
            family_processing_priority(
                frozenset(group["families"])
            ),
            int(group["group_size"]),
            max(tuple(group["teacher_counts"])),
            sum(
                count > 0
                for count in tuple(group["teacher_counts"])
            ),
            int(group["maximum_vertices"]),
        ),
        reverse=True,
    )

    memo: set[tuple[object, ...]] = set()
    visited_nodes = 0

    def recurse(
        index: int,
        train_remaining: tuple[int, ...],
        validation_remaining: tuple[int, ...],
        test_remaining: tuple[int, ...],
        train_families: frozenset[str],
        validation_families: frozenset[str],
        test_families: frozenset[str],
    ) -> bool:
        nonlocal visited_nodes

        visited_nodes += 1

        if visited_nodes > node_limit:
            return False

        if index == len(remaining_groups):
            quotas_complete = (
                vector_is_zero(train_remaining)
                and vector_is_zero(validation_remaining)
                and vector_is_zero(test_remaining)
            )

            return (
                quotas_complete
                and family_requirements_are_met(
                    train_families=train_families,
                    validation_families=validation_families,
                    test_families=test_families,
                )
            )

        state = (
            index,
            train_remaining,
            validation_remaining,
            test_remaining,
            tuple(sorted(train_families)),
            tuple(sorted(validation_families)),
            tuple(sorted(test_families)),
        )

        if state in memo:
            return False

        group = remaining_groups[index]
        group_id = str(group["group_id"])
        counts = tuple(group["teacher_counts"])
        families = frozenset(group["families"])

        capacities = {
            "train": train_remaining,
            "validation": validation_remaining,
            "test": test_remaining,
        }

        eligible_splits = [
            split
            for split in SPLITS
            if vector_fits(
                counts,
                capacities[split],
            )
        ]

        if not eligible_splits:
            memo.add(state)
            return False

        rng.shuffle(eligible_splits)

        def placement_priority(
            split: str,
        ) -> tuple[float, int, str]:
            """
            Prefer placements that add missing graph families.
            """
            current_families = {
                "train": train_families,
                "validation": validation_families,
                "test": test_families,
            }[split]

            bonus = (
                300.0
                * len(
                    families - current_families
                )
            )

            if (
                "erdos_renyi"
                not in current_families
                and "erdos_renyi" in families
            ):
                bonus += 1000.0

            if (
                split in {"validation", "test"}
                and len(current_families) < 3
            ):
                bonus += (
                    500.0
                    * len(
                        families - current_families
                    )
                )

            if (
                split == "train"
                and len(current_families) < 4
            ):
                bonus += (
                    300.0
                    * len(
                        families - current_families
                    )
                )

            return (
                -bonus,
                sum(capacities[split]),
                split,
            )

        eligible_splits.sort(
            key=placement_priority
        )

        for split in eligible_splits:
            next_capacities = dict(capacities)

            next_capacities[split] = subtract_vectors(
                capacities[split],
                counts,
            )

            next_train_families = train_families
            next_validation_families = validation_families
            next_test_families = test_families

            if split == "train":
                next_train_families = (
                    train_families | families
                )

            elif split == "validation":
                next_validation_families = (
                    validation_families | families
                )

            elif split == "test":
                next_test_families = (
                    test_families | families
                )

            assignment[group_id] = split

            solved = recurse(
                index=index + 1,
                train_remaining=next_capacities["train"],
                validation_remaining=next_capacities[
                    "validation"
                ],
                test_remaining=next_capacities["test"],
                train_families=next_train_families,
                validation_families=(
                    next_validation_families
                ),
                test_families=next_test_families,
            )

            if solved:
                return True

            assignment.pop(group_id, None)

        memo.add(state)
        return False

    solved = recurse(
        index=0,
        train_remaining=remaining_capacity["train"],
        validation_remaining=remaining_capacity[
            "validation"
        ],
        test_remaining=remaining_capacity["test"],
        train_families=split_families["train"],
        validation_families=split_families["validation"],
        test_families=split_families["test"],
    )

    return assignment if solved else None


def expand_assignment(
    df: pd.DataFrame,
    group_assignment: dict[str, str],
) -> pd.DataFrame:
    """
    Convert group assignments back to graph-level rows.
    """
    output = df.copy()

    output["split_group_id"] = output.apply(
        lambda row: build_group_id(
            graph_id=str(row["graph_id"]),
            family=str(row["family"]),
        ),
        axis=1,
    )

    output["split"] = output["split_group_id"].map(
        group_assignment
    )

    output["split_grouping_mode"] = (
        "strict_replicate_groups_with_fixed_crown_sizes"
    )

    if output["split"].isna().any():
        missing_graphs = output.loc[
            output["split"].isna(),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Missing split assignments for: "
            f"{missing_graphs}"
        )

    return output


def validate_split(
    split_df: pd.DataFrame,
) -> None:
    """
    Strictly validate the final split.
    """
    if len(split_df) != 68:
        raise ValueError(
            f"Expected 68 graphs, found {len(split_df)}."
        )

    if split_df["graph_id"].duplicated().any():
        duplicates = split_df.loc[
            split_df["graph_id"].duplicated(
                keep=False
            ),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph IDs found: {duplicates}"
        )

    split_counts = (
        split_df["split"]
        .value_counts()
    )

    for split, expected_count in (
        EXPECTED_SPLIT_SIZES.items()
    ):
        actual_count = int(
            split_counts.get(split, 0)
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} graphs, "
                f"found {actual_count}."
            )

    teacher_table = pd.crosstab(
        split_df["split"],
        split_df["selected_teacher_ordering"],
    ).reindex(
        index=SPLITS,
        columns=TEACHERS,
        fill_value=0,
    )

    for split in SPLITS:
        expected_quota = TEACHER_QUOTAS[split]

        for teacher_index, teacher in enumerate(
            TEACHERS
        ):
            actual_count = int(
                teacher_table.loc[split, teacher]
            )

            expected_count = expected_quota[
                teacher_index
            ]

            if actual_count != expected_count:
                raise ValueError(
                    f"{split}, {teacher}: expected "
                    f"{expected_count}, found "
                    f"{actual_count}."
                )

    # Verify that replicate groups do not cross splits.
    group_split_counts = (
        split_df
        .groupby("split_group_id")["split"]
        .nunique()
    )

    leaked_groups = group_split_counts[
        group_split_counts > 1
    ]

    if not leaked_groups.empty:
        raise ValueError(
            "Replicate-group leakage detected: "
            f"{leaked_groups.index.tolist()}"
        )

    # Verify every fixed crown allocation.
    for graph_id, expected_split in (
        FIXED_CROWN_SPLITS.items()
    ):
        graph_rows = split_df[
            split_df["graph_id"] == graph_id
        ]

        if len(graph_rows) != 1:
            raise ValueError(
                f"Expected exactly one row for fixed crown "
                f"graph {graph_id}."
            )

        actual_split = str(
            graph_rows.iloc[0]["split"]
        )

        if actual_split != expected_split:
            raise ValueError(
                f"{graph_id}: expected split "
                f"{expected_split}, found {actual_split}."
            )

    ba_rows = split_df[
        split_df["family"] == "barabasi_albert"
    ]

    if not ba_rows.empty and not (
        ba_rows["split"] == "train"
    ).all():
        raise ValueError(
            "Barabási–Albert graph must remain in training."
        )

    train_families = frozenset(
        split_df.loc[
            split_df["split"] == "train",
            "family",
        ].astype(str)
    )

    validation_families = frozenset(
        split_df.loc[
            split_df["split"] == "validation",
            "family",
        ].astype(str)
    )

    test_families = frozenset(
        split_df.loc[
            split_df["split"] == "test",
            "family",
        ].astype(str)
    )

    if not family_requirements_are_met(
        train_families=train_families,
        validation_families=validation_families,
        test_families=test_families,
    ):
        raise ValueError(
            "Final split does not satisfy the required "
            "graph-family safeguards."
        )


def score_split(
    split_df: pd.DataFrame,
) -> float:
    """
    Prefer broader family coverage and larger test graphs.
    """
    train_df = split_df[
        split_df["split"] == "train"
    ]

    validation_df = split_df[
        split_df["split"] == "validation"
    ]

    test_df = split_df[
        split_df["split"] == "test"
    ]

    score = (
        300.0
        * train_df["family"].nunique()
        + 500.0
        * validation_df["family"].nunique()
        + 1000.0
        * test_df["family"].nunique()
    )

    score += float(
        test_df["num_vertices"].mean()
    )

    score += 0.25 * float(
        validation_df["num_vertices"].mean()
    )

    # Slightly prefer SBM and WS representation outside training.
    score += (
        100.0
        * int(
            (
                validation_df["family"]
                == "stochastic_block_model"
            ).any()
        )
    )

    score += (
        100.0
        * int(
            (
                test_df["family"]
                == "stochastic_block_model"
            ).any()
        )
    )

    score += (
        75.0
        * int(
            (
                validation_df["family"]
                == "watts_strogatz"
            ).any()
        )
    )

    score += (
        75.0
        * int(
            (
                test_df["family"]
                == "watts_strogatz"
            ).any()
        )
    )

    return score


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Teacher assignment file not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "graph_id",
        "family",
        "num_vertices",
        "num_edges",
        "density",
        "matrix_path",
        "selected_teacher_ordering",
        "best_colpack5_colors",
        "worst_colpack5_colors",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
        "teacher_selection_reason",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    missing_fixed_crowns = [
        graph_id
        for graph_id in FIXED_CROWN_SPLITS
        if graph_id not in set(
            df["graph_id"].astype(str)
        )
    ]

    if missing_fixed_crowns:
        raise ValueError(
            f"Fixed crown graphs are missing from the input: "
            f"{missing_fixed_crowns}"
        )

    teacher_totals = (
        df["selected_teacher_ordering"]
        .value_counts()
    )

    for teacher in TEACHERS:
        actual_count = int(
            teacher_totals.get(teacher, 0)
        )

        if actual_count != 17:
            raise ValueError(
                f"{teacher}: expected 17 graphs, "
                f"found {actual_count}."
            )

    groups = build_groups(df)

    best_split: pd.DataFrame | None = None
    best_score: float | None = None
    safe_solution_count = 0

    for attempt in range(300):
        assignment = find_group_assignment(
            groups=groups,
            seed=18_000 + attempt,
        )

        if assignment is None:
            continue

        candidate_split = expand_assignment(
            df=df,
            group_assignment=assignment,
        )

        validate_split(candidate_split)

        candidate_score = score_split(
            candidate_split
        )

        safe_solution_count += 1

        if (
            best_split is None
            or best_score is None
            or candidate_score > best_score
        ):
            best_split = candidate_split
            best_score = candidate_score

    print(
        "strict_replicate_groups_with_fixed_crowns: "
        f"valid solutions={safe_solution_count}"
    )

    if (
        best_split is None
        or best_score is None
    ):
        raise RuntimeError(
            "No exact teacher-balanced split was found with "
            "the fixed 5/2/4 crown allocation and strict "
            "replicate grouping."
        )

    validate_split(best_split)

    best_split = best_split.sort_values(
        by=[
            "split",
            "selected_teacher_ordering",
            "family",
            "graph_id",
        ]
    ).reset_index(drop=True)

    output_columns = [
        "graph_id",
        "split",
        "split_group_id",
        "split_grouping_mode",
        "family",
        "num_vertices",
        "num_edges",
        "density",
        "matrix_path",
        "selected_teacher_ordering",
        "best_colpack5_colors",
        "worst_colpack5_colors",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
        "teacher_selection_reason",
    ]

    SPLIT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPLIT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_split[
        output_columns
    ].to_csv(
        SPLIT_OUTPUT_PATH,
        index=False,
    )

    summary_df = (
        best_split
        .groupby(
            [
                "split",
                "selected_teacher_ordering",
            ],
            as_index=False,
        )
        .agg(
            num_graphs=("graph_id", "count"),
            minimum_vertices=("num_vertices", "min"),
            maximum_vertices=("num_vertices", "max"),
            average_vertices=("num_vertices", "mean"),
            minimum_gap=("ordering_gap", "min"),
            maximum_gap=("ordering_gap", "max"),
        )
    )

    summary_df.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    group_audit_rows: list[
        dict[str, object]
    ] = []

    for group_id, group_df in best_split.groupby(
        "split_group_id",
        sort=True,
    ):
        teacher_counter = Counter(
            group_df["selected_teacher_ordering"]
        )

        group_audit_rows.append(
            {
                "split_group_id": group_id,
                "split": group_df["split"].iloc[0],
                "split_grouping_mode": (
                    "strict_replicate_groups_"
                    "with_fixed_crown_sizes"
                ),
                "num_graphs": len(group_df),
                "graph_ids": "; ".join(
                    sorted(
                        group_df["graph_id"]
                        .astype(str)
                        .tolist()
                    )
                ),
                "families": "; ".join(
                    sorted(
                        group_df["family"]
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                ),
                "LARGEST_FIRST": teacher_counter[
                    "LARGEST_FIRST"
                ],
                "DYNAMIC_LARGEST_FIRST": teacher_counter[
                    "DYNAMIC_LARGEST_FIRST"
                ],
                "INCIDENCE_DEGREE": teacher_counter[
                    "INCIDENCE_DEGREE"
                ],
                "SMALLEST_LAST": teacher_counter[
                    "SMALLEST_LAST"
                ],
            }
        )

    pd.DataFrame(
        group_audit_rows
    ).to_csv(
        GROUP_AUDIT_PATH,
        index=False,
    )

    print()
    print(
        "Week 18 fixed-crown heterogeneous balanced split"
    )
    print(
        "------------------------------------------------"
    )
    print(
        "Selected grouping mode: "
        "strict_replicate_groups_with_fixed_crown_sizes"
    )
    print(f"Split score: {best_score:.2f}")
    print()

    print("Graph counts:")

    for split in SPLITS:
        count = int(
            (
                best_split["split"]
                == split
            ).sum()
        )

        print(f"  {split}: {count}")

    print()
    print("Teacher distribution by split:")

    teacher_table = pd.crosstab(
        best_split["split"],
        best_split["selected_teacher_ordering"],
    ).reindex(
        index=SPLITS,
        columns=TEACHERS,
        fill_value=0,
    )

    print(teacher_table.to_string())
    print()

    print("Family distribution by split:")

    family_table = pd.crosstab(
        best_split["split"],
        best_split["family"],
    ).reindex(
        index=SPLITS,
        fill_value=0,
    )

    print(family_table.to_string())
    print()

    print("Crown distribution by split:")

    crown_table = (
        best_split[
            best_split["family"] == "crown"
        ]
        .groupby("split")
        .agg(
            num_crowns=("graph_id", "count"),
            minimum_vertices=("num_vertices", "min"),
            maximum_vertices=("num_vertices", "max"),
        )
        .reindex(SPLITS)
    )

    print(crown_table.to_string())
    print()

    print("Vertex ranges by split:")

    for split in SPLITS:
        split_rows = best_split[
            best_split["split"] == split
        ]

        print(
            f"  {split}: "
            f"{int(split_rows['num_vertices'].min())}–"
            f"{int(split_rows['num_vertices'].max())}"
        )

    print()
    print(
        "All fixed crown assignments were verified."
    )
    print(
        "Crown sizes increase from training to validation "
        "to testing."
    )
    print(
        "No replicate group appears in more than one split."
    )
    print(
        "Exact teacher balance is preserved."
    )
    print()
    print(f"Saved split to: {SPLIT_OUTPUT_PATH}")
    print(
        f"Saved split summary to: "
        f"{SPLIT_SUMMARY_PATH}"
    )
    print(
        f"Saved group audit to: "
        f"{GROUP_AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()