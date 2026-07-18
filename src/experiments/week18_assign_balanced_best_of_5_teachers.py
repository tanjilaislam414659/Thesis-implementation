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
    / "week18_combined_safeguarded_heterogeneous_pool.csv"
)

ASSIGNMENT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_balanced_best_of_5_teacher_assignments.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "initial_graph_coloring_benchmarks"
    / "week18_balanced_teacher_assignment_summary.csv"
)


ALL_ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]

STRUCTURAL_ORDERINGS = [
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
]

# Used only when eligible orderings currently have equal counts.
BALANCE_TIE_PRIORITY = [
    "INCIDENCE_DEGREE",
    "SMALLEST_LAST",
    "LARGEST_FIRST",
    "DYNAMIC_LARGEST_FIRST",
]


def parse_best_orderings(value: object) -> list[str]:
    if pd.isna(value):
        raise ValueError("Missing best_colpack5_orderings value.")

    orderings = [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]

    if not orderings:
        raise ValueError(
            f"Could not parse best orderings from: {value}"
        )

    unknown = [
        ordering
        for ordering in orderings
        if ordering not in ALL_ORDERINGS
    ]

    if unknown:
        raise ValueError(
            f"Unknown ordering names found: {unknown}"
        )

    return orderings


def choose_balanced_teacher(
    eligible_orderings: list[str],
    current_counts: Counter[str],
) -> str:
    """
    Choose the least-used eligible best ordering.

    The fixed priority is used only when several eligible orderings
    currently have the same number of assignments.
    """
    priority_index = {
        ordering: index
        for index, ordering in enumerate(BALANCE_TIE_PRIORITY)
    }

    return min(
        eligible_orderings,
        key=lambda ordering: (
            current_counts[ordering],
            priority_index.get(ordering, 99),
            ordering,
        ),
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Combined Week 18 pool not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "graph_id",
        "family",
        "ordering_gap",
        "best_colpack5_orderings",
        "num_best_orderings",
        "best_colpack5_colors",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["graph_id"].duplicated().any():
        duplicates = df.loc[
            df["graph_id"].duplicated(keep=False),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Duplicate graph IDs found: {duplicates}"
        )

    df["num_best_orderings"] = pd.to_numeric(
        df["num_best_orderings"],
        errors="raise",
    ).astype(int)

    df["ordering_gap"] = pd.to_numeric(
        df["ordering_gap"],
        errors="raise",
    ).astype(int)

    current_counts: Counter[str] = Counter()
    unique_assignment_counts: Counter[str] = Counter()
    tied_assignment_counts: Counter[str] = Counter()

    assignment_records: dict[str, dict[str, object]] = {}

    # Preserve every unique ColPack winner.
    unique_df = df[
        df["num_best_orderings"] == 1
    ].copy()

    unique_df = unique_df.sort_values(
        by=[
            "family",
            "graph_id",
        ]
    )

    for row in unique_df.itertuples(index=False):
        best_orderings = parse_best_orderings(
            row.best_colpack5_orderings
        )

        if len(best_orderings) != 1:
            raise ValueError(
                f"{row.graph_id}: expected one best ordering, "
                f"but found {best_orderings}"
            )

        selected_teacher = best_orderings[0]
        count_before = current_counts[selected_teacher]

        current_counts[selected_teacher] += 1
        unique_assignment_counts[selected_teacher] += 1

        assignment_records[row.graph_id] = {
            "selected_teacher_ordering": selected_teacher,
            "teacher_selection_reason": "unique_best_ordering",
            "teacher_candidate_orderings": "; ".join(
                best_orderings
            ),
            "teacher_candidate_count": len(best_orderings),
            "teacher_count_before_assignment": count_before,
            "natural_excluded_from_tied_selection": False,
            "selected_teacher_is_colpack_best": True,
        }

    # Process the most constrained tied cases first.
    tied_df = df[
        df["num_best_orderings"] > 1
    ].copy()

    tied_df["parsed_best_orderings"] = (
        tied_df["best_colpack5_orderings"]
        .apply(parse_best_orderings)
    )

    tied_df["structural_candidate_count"] = (
        tied_df["parsed_best_orderings"]
        .apply(
            lambda orderings: sum(
                ordering in STRUCTURAL_ORDERINGS
                for ordering in orderings
            )
        )
    )

    tied_df = tied_df.sort_values(
        by=[
            "structural_candidate_count",
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

    for row in tied_df.itertuples(index=False):
        best_orderings = list(row.parsed_best_orderings)

        structural_candidates = [
            ordering
            for ordering in best_orderings
            if ordering in STRUCTURAL_ORDERINGS
        ]

        natural_excluded = (
            "NATURAL" in best_orderings
            and bool(structural_candidates)
        )

        # NATURAL depends strongly on the input vertex labels.
        # Use it only when no structural best ordering is available.
        eligible_orderings = (
            structural_candidates
            if structural_candidates
            else best_orderings
        )

        selected_teacher = choose_balanced_teacher(
            eligible_orderings=eligible_orderings,
            current_counts=current_counts,
        )

        if selected_teacher not in best_orderings:
            raise ValueError(
                f"{row.graph_id}: selected teacher "
                f"{selected_teacher} is not in the verified best set "
                f"{best_orderings}"
            )

        count_before = current_counts[selected_teacher]

        current_counts[selected_teacher] += 1
        tied_assignment_counts[selected_teacher] += 1

        assignment_records[row.graph_id] = {
            "selected_teacher_ordering": selected_teacher,
            "teacher_selection_reason": (
                "balanced_among_tied_best_orderings"
            ),
            "teacher_candidate_orderings": "; ".join(
                best_orderings
            ),
            "teacher_candidate_count": len(best_orderings),
            "teacher_count_before_assignment": count_before,
            "natural_excluded_from_tied_selection": natural_excluded,
            "selected_teacher_is_colpack_best": True,
        }

    if len(assignment_records) != len(df):
        raise ValueError(
            f"Expected {len(df)} teacher assignments, "
            f"but created {len(assignment_records)}."
        )

    assignment_df = pd.DataFrame.from_dict(
        assignment_records,
        orient="index",
    )

    assignment_df.index.name = "graph_id"
    assignment_df = assignment_df.reset_index()

    output_df = df.merge(
        assignment_df,
        on="graph_id",
        how="left",
        validate="one_to_one",
    )

    if output_df["selected_teacher_ordering"].isna().any():
        missing_graphs = output_df.loc[
            output_df["selected_teacher_ordering"].isna(),
            "graph_id",
        ].tolist()

        raise ValueError(
            f"Missing assignments for: {missing_graphs}"
        )

    # Final safeguard: every teacher must belong to the graph's
    # verified ColPack-best set.
    invalid_graphs: list[str] = []

    for row in output_df.itertuples(index=False):
        best_orderings = parse_best_orderings(
            row.best_colpack5_orderings
        )

        if row.selected_teacher_ordering not in best_orderings:
            invalid_graphs.append(row.graph_id)

    if invalid_graphs:
        raise ValueError(
            "Teachers outside their verified best sets: "
            f"{invalid_graphs}"
        )

    output_df = output_df.sort_values(
        by=[
            "selected_teacher_ordering",
            "family",
            "graph_id",
        ]
    )

    ASSIGNMENT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        ASSIGNMENT_OUTPUT_PATH,
        index=False,
    )

    summary_rows = []

    for ordering in ALL_ORDERINGS:
        ordering_rows = output_df[
            output_df["selected_teacher_ordering"] == ordering
        ]

        summary_rows.append(
            {
                "ordering_name": ordering,
                "total_teacher_assignments": len(ordering_rows),
                "unique_winner_assignments": (
                    unique_assignment_counts[ordering]
                ),
                "tied_winner_assignments": (
                    tied_assignment_counts[ordering]
                ),
                "barabasi_albert_assignments": int(
                    (
                        ordering_rows["family"]
                        == "barabasi_albert"
                    ).sum()
                ),
                "crown_assignments": int(
                    (
                        ordering_rows["family"]
                        == "crown"
                    ).sum()
                ),
                "erdos_renyi_assignments": int(
                    (
                        ordering_rows["family"]
                        == "erdos_renyi"
                    ).sum()
                ),
                "stochastic_block_model_assignments": int(
                    (
                        ordering_rows["family"]
                        == "stochastic_block_model"
                    ).sum()
                ),
                "watts_strogatz_assignments": int(
                    (
                        ordering_rows["family"]
                        == "watts_strogatz"
                    ).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print("Week 18 balanced best-of-five teacher assignment")
    print("------------------------------------------------")
    print(f"Graphs assigned: {len(output_df)}")
    print(
        f"Unique-winner graphs: "
        f"{int((output_df['num_best_orderings'] == 1).sum())}"
    )
    print(
        f"Tied-winner graphs: "
        f"{int((output_df['num_best_orderings'] > 1).sum())}"
    )
    print()

    print("Final teacher distribution:")

    for row in summary_df.itertuples(index=False):
        print(
            f"  {row.ordering_name}: "
            f"total={row.total_teacher_assignments}, "
            f"unique={row.unique_winner_assignments}, "
            f"tied={row.tied_winner_assignments}, "
            f"BA={row.barabasi_albert_assignments}, "
            f"crown={row.crown_assignments}, "
            f"ER={row.erdos_renyi_assignments}, "
            f"SBM={row.stochastic_block_model_assignments}, "
            f"WS={row.watts_strogatz_assignments}"
        )

    print()
    print(
        "All selected teachers are verified members of their "
        "graph's ColPack-best ordering set."
    )
    print()
    print(f"Saved assignments to: {ASSIGNMENT_OUTPUT_PATH}")
    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()