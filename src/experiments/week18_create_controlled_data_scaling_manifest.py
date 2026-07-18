from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_PYG_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week17_heuristic_gap_symmetry_breaking"
)

MIXED_PYG_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "pyg_data_week18_controlled_mixed_exact"
)

MIXED_PYG_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_mixed_exact_pyg_summary.csv"
)

SELECTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_mixed_selection.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "initial_graph_coloring_dataset"
    / "splits"
    / "week18_controlled_data_scaling_manifest.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gnn_node_scorer"
    / "week18_controlled_data_scaling_manifest_summary.csv"
)


EXPECTED_ORIGINAL_COUNTS = {
    "train": 20,
    "validation": 5,
    "test": 5,
}

EXPECTED_MIXED_COUNT = 105

GAP_LEVELS = [
    2,
    3,
    4,
    5,
]

CONDITIONS = [
    {
        "condition": "train_20_baseline",
        "added_mixed_count": 0,
        "total_train_graphs": 20,
    },
    {
        "condition": "train_32_plus12",
        "added_mixed_count": 12,
        "total_train_graphs": 32,
    },
    {
        "condition": "train_44_plus24",
        "added_mixed_count": 24,
        "total_train_graphs": 44,
    },
    {
        "condition": "train_125_plus105",
        "added_mixed_count": 105,
        "total_train_graphs": 125,
    },
]


def load_torch_data(
    path: Path,
):
    """
    Load a saved PyG object while supporting PyTorch versions with
    different torch.load signatures.
    """
    try:
        return torch.load(
            path,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path)


def scalar_to_int(
    value: object,
) -> int:
    """
    Convert Python, NumPy, or scalar tensor values to int.
    """
    if isinstance(value, torch.Tensor):
        return int(
            value.detach().cpu().item()
        )

    return int(value)


def evenly_spaced_positions(
    num_items: int,
    num_selected: int,
) -> list[int]:
    """
    Select deterministic positions distributed across an ordered list.
    """
    if num_selected < 0:
        raise ValueError(
            "num_selected cannot be negative."
        )

    if num_selected > num_items:
        raise ValueError(
            f"Cannot select {num_selected} items from "
            f"{num_items} available items."
        )

    if num_selected == 0:
        return []

    if num_selected == 1:
        return [
            num_items // 2
        ]

    positions = [
        int(
            round(
                index
                * (num_items - 1)
                / (num_selected - 1)
            )
        )
        for index in range(
            num_selected
        )
    ]

    if len(set(positions)) != num_selected:
        raise ValueError(
            "Evenly spaced selection produced "
            "duplicate positions."
        )

    return positions


def load_original_dataset() -> pd.DataFrame:
    """
    Load the original Week 17 controlled PyG graphs and preserve their
    train, validation, and test labels.
    """
    if not ORIGINAL_PYG_DIR.exists():
        raise FileNotFoundError(
            f"Original PyG directory not found: "
            f"{ORIGINAL_PYG_DIR}"
        )

    paths = sorted(
        ORIGINAL_PYG_DIR.glob("*.pt")
    )

    if not paths:
        raise ValueError(
            f"No PyG graphs found in: "
            f"{ORIGINAL_PYG_DIR}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for path in paths:
        data = load_torch_data(
            path
        )

        graph_id = str(
            data.graph_id
        )

        split = str(
            data.split
        )

        base_cycle_size = scalar_to_int(
            data.base_cycle_size
        )

        gap_level = scalar_to_int(
            data.gap_level
        )

        num_nodes = scalar_to_int(
            data.num_nodes
        )

        target_colors = scalar_to_int(
            data.selected_num_colors
        )

        best_colpack5_colors = scalar_to_int(
            data.best_colpack5_colors
        )

        rows.append(
            {
                "graph_id": graph_id,
                "split": split,
                "source_dataset": (
                    "week17_original_controlled"
                ),
                "source_pt_path": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "is_added_mixed_graph": False,
                "graph_family": str(
                    data.graph_family
                ),
                "base_cycle_size": (
                    base_cycle_size
                ),
                "component_cycle_sizes": (
                    str(base_cycle_size)
                ),
                "num_unique_component_sizes": 1,
                "gap_level": (
                    gap_level
                ),
                "num_nodes": (
                    num_nodes
                ),
                "target_colors": (
                    target_colors
                ),
                "best_colpack5_colors": (
                    best_colpack5_colors
                ),
            }
        )

    original_df = pd.DataFrame(
        rows
    )

    if original_df[
        "graph_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in the "
            "original controlled dataset."
        )

    actual_counts = (
        original_df[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    for split, expected_count in (
        EXPECTED_ORIGINAL_COUNTS.items()
    ):
        actual_count = int(
            actual_counts.get(
                split,
                0,
            )
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} original "
                f"{split} graphs, found {actual_count}."
            )

    validation_df = original_df[
        original_df["split"]
        == "validation"
    ]

    test_df = original_df[
        original_df["split"]
        == "test"
    ]

    if not (
        validation_df[
            "base_cycle_size"
        ]
        == 38
    ).all():
        raise ValueError(
            "The frozen validation set is not "
            "exclusively based on cycle size 38."
        )

    if not (
        test_df[
            "base_cycle_size"
        ]
        == 41
    ).all():
        raise ValueError(
            "The frozen test set is not exclusively "
            "based on cycle size 41."
        )

    return original_df


def load_mixed_dataset_summary() -> pd.DataFrame:
    """
    Load and validate metadata for the 105 new controlled graphs.
    """
    if not MIXED_PYG_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Mixed PyG summary not found: "
            f"{MIXED_PYG_SUMMARY_PATH}"
        )

    mixed_df = pd.read_csv(
        MIXED_PYG_SUMMARY_PATH
    )

    required_columns = {
        "graph_id",
        "split",
        "graph_family",
        "num_components_joined",
        "component_cycle_sizes",
        "num_unique_component_sizes",
        "gap_level",
        "num_nodes",
        "target_colors",
        "best_colpack5_colors",
        "path",
    }

    missing_columns = (
        required_columns
        - set(mixed_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Mixed PyG summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(mixed_df) != EXPECTED_MIXED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MIXED_COUNT} mixed graphs, "
            f"found {len(mixed_df)}."
        )

    if mixed_df[
        "graph_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate graph IDs found in mixed summary."
        )

    if not (
        mixed_df["split"]
        == "train"
    ).all():
        raise ValueError(
            "All mixed graphs must initially be "
            "marked as training graphs."
        )

    if set(
        mixed_df[
            "gap_level"
        ].astype(int).unique()
    ) != set(GAP_LEVELS):
        raise ValueError(
            "Mixed dataset does not contain exactly "
            "gap levels 2, 3, 4, and 5."
        )

    for row in mixed_df.itertuples(
        index=False
    ):
        source_path = (
            PROJECT_ROOT
            / str(row.path)
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Mixed PyG file not found: "
                f"{source_path}"
            )

        component_sizes = [
            int(part)
            for part in str(
                row.component_cycle_sizes
            ).split(";")
        ]

        if 38 in component_sizes:
            raise ValueError(
                f"{row.graph_id} leaks frozen "
                "validation size 38."
            )

        if 41 in component_sizes:
            raise ValueError(
                f"{row.graph_id} leaks frozen "
                "test size 41."
            )

        if int(
            row.target_colors
        ) != (
            4
            * int(
                row.num_components_joined
            )
        ):
            raise ValueError(
                f"{row.graph_id}: inconsistent "
                "target color count."
            )

        if int(
            row.gap_level
        ) != int(
            row.num_components_joined
        ):
            raise ValueError(
                f"{row.graph_id}: gap level does "
                "not equal component count."
            )

    return mixed_df


def select_nested_mixed_graphs(
    mixed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select nested additions:

    - 12 graphs: 3 per gap level;
    - 24 graphs: 6 per gap level;
    - 105 graphs: all candidates.

    Within each gap level, graphs are ordered by size and structural
    diversity. Six are spread across that ordered range, and three
    are selected as a nested subset of those six.
    """
    selection_parts: list[
        pd.DataFrame
    ] = []

    for gap_level in GAP_LEVELS:
        gap_df = (
            mixed_df[
                mixed_df["gap_level"]
                == gap_level
            ]
            .sort_values(
                [
                    "num_nodes",
                    "num_unique_component_sizes",
                    "component_cycle_sizes",
                    "graph_id",
                ]
            )
            .reset_index(drop=True)
            .copy()
        )

        if len(gap_df) < 6:
            raise ValueError(
                f"Gap level {gap_level} has only "
                f"{len(gap_df)} graphs; at least 6 "
                "are required."
            )

        plus24_positions = (
            evenly_spaced_positions(
                num_items=len(gap_df),
                num_selected=6,
            )
        )

        plus24_graph_ids = (
            gap_df.iloc[
                plus24_positions
            ]["graph_id"]
            .tolist()
        )

        plus24_df = (
            gap_df[
                gap_df["graph_id"].isin(
                    plus24_graph_ids
                )
            ]
            .sort_values(
                [
                    "num_nodes",
                    "num_unique_component_sizes",
                    "component_cycle_sizes",
                    "graph_id",
                ]
            )
            .reset_index(drop=True)
        )

        plus12_positions = (
            evenly_spaced_positions(
                num_items=len(
                    plus24_df
                ),
                num_selected=3,
            )
        )

        plus12_graph_ids = set(
            plus24_df.iloc[
                plus12_positions
            ]["graph_id"]
            .tolist()
        )

        plus24_graph_id_set = set(
            plus24_graph_ids
        )

        gap_df[
            "selected_in_plus12"
        ] = gap_df[
            "graph_id"
        ].isin(
            plus12_graph_ids
        )

        gap_df[
            "selected_in_plus24"
        ] = gap_df[
            "graph_id"
        ].isin(
            plus24_graph_id_set
        )

        gap_df[
            "selected_in_plus105"
        ] = True

        selection_parts.append(
            gap_df
        )

    selection_df = pd.concat(
        selection_parts,
        ignore_index=True,
    )

    selected_12 = selection_df[
        selection_df[
            "selected_in_plus12"
        ]
    ]

    selected_24 = selection_df[
        selection_df[
            "selected_in_plus24"
        ]
    ]

    if len(selected_12) != 12:
        raise ValueError(
            f"Expected 12 selected graphs, "
            f"found {len(selected_12)}."
        )

    if len(selected_24) != 24:
        raise ValueError(
            f"Expected 24 selected graphs, "
            f"found {len(selected_24)}."
        )

    if not set(
        selected_12[
            "graph_id"
        ]
    ).issubset(
        set(
            selected_24[
                "graph_id"
            ]
        )
    ):
        raise ValueError(
            "The 12-graph selection is not nested "
            "inside the 24-graph selection."
        )

    for gap_level in GAP_LEVELS:
        plus12_count = int(
            (
                selected_12[
                    "gap_level"
                ]
                == gap_level
            ).sum()
        )

        plus24_count = int(
            (
                selected_24[
                    "gap_level"
                ]
                == gap_level
            ).sum()
        )

        if plus12_count != 3:
            raise ValueError(
                f"Gap {gap_level}: expected 3 graphs "
                f"in plus12, found {plus12_count}."
            )

        if plus24_count != 6:
            raise ValueError(
                f"Gap {gap_level}: expected 6 graphs "
                f"in plus24, found {plus24_count}."
            )

    return selection_df


def original_rows_for_manifest(
    original_df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    rows = original_df.copy()

    rows.insert(
        0,
        "condition",
        condition,
    )

    return rows


def mixed_rows_for_manifest(
    selected_df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "condition": condition,
            "graph_id": (
                selected_df[
                    "graph_id"
                ].astype(str)
            ),
            "split": "train",
            "source_dataset": (
                "week18_controlled_mixed_exact"
            ),
            "source_pt_path": (
                selected_df[
                    "path"
                ].astype(str)
            ),
            "is_added_mixed_graph": True,
            "graph_family": (
                selected_df[
                    "graph_family"
                ].astype(str)
            ),
            "base_cycle_size": pd.NA,
            "component_cycle_sizes": (
                selected_df[
                    "component_cycle_sizes"
                ].astype(str)
            ),
            "num_unique_component_sizes": (
                selected_df[
                    "num_unique_component_sizes"
                ].astype(int)
            ),
            "gap_level": (
                selected_df[
                    "gap_level"
                ].astype(int)
            ),
            "num_nodes": (
                selected_df[
                    "num_nodes"
                ].astype(int)
            ),
            "target_colors": (
                selected_df[
                    "target_colors"
                ].astype(int)
            ),
            "best_colpack5_colors": (
                selected_df[
                    "best_colpack5_colors"
                ].astype(int)
            ),
        }
    )

    return rows


def build_manifest(
    original_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    manifest_parts: list[
        pd.DataFrame
    ] = []

    for condition_info in CONDITIONS:
        condition = str(
            condition_info[
                "condition"
            ]
        )

        added_count = int(
            condition_info[
                "added_mixed_count"
            ]
        )

        original_rows = (
            original_rows_for_manifest(
                original_df=original_df,
                condition=condition,
            )
        )

        manifest_parts.append(
            original_rows
        )

        if added_count == 0:
            continue

        if added_count == 12:
            selected_mixed = selection_df[
                selection_df[
                    "selected_in_plus12"
                ]
            ]

        elif added_count == 24:
            selected_mixed = selection_df[
                selection_df[
                    "selected_in_plus24"
                ]
            ]

        elif added_count == 105:
            selected_mixed = selection_df[
                selection_df[
                    "selected_in_plus105"
                ]
            ]

        else:
            raise ValueError(
                f"Unsupported added count: "
                f"{added_count}"
            )

        if len(selected_mixed) != added_count:
            raise ValueError(
                f"{condition}: expected "
                f"{added_count} added mixed graphs, "
                f"found {len(selected_mixed)}."
            )

        mixed_rows = (
            mixed_rows_for_manifest(
                selected_df=selected_mixed,
                condition=condition,
            )
        )

        manifest_parts.append(
            mixed_rows
        )

    manifest_df = pd.concat(
        manifest_parts,
        ignore_index=True,
    )

    duplicate_mask = (
        manifest_df[
            [
                "condition",
                "graph_id",
            ]
        ].duplicated()
    )

    if duplicate_mask.any():
        duplicates = manifest_df[
            duplicate_mask
        ][
            [
                "condition",
                "graph_id",
            ]
        ]

        raise ValueError(
            "Duplicate graph entries found within "
            "a condition:\n"
            + duplicates.to_string(
                index=False
            )
        )

    for condition_info in CONDITIONS:
        condition = str(
            condition_info[
                "condition"
            ]
        )

        expected_train_count = int(
            condition_info[
                "total_train_graphs"
            ]
        )

        condition_df = manifest_df[
            manifest_df[
                "condition"
            ]
            == condition
        ]

        actual_split_counts = (
            condition_df[
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        actual_train_count = int(
            actual_split_counts.get(
                "train",
                0,
            )
        )

        actual_validation_count = int(
            actual_split_counts.get(
                "validation",
                0,
            )
        )

        actual_test_count = int(
            actual_split_counts.get(
                "test",
                0,
            )
        )

        if (
            actual_train_count
            != expected_train_count
        ):
            raise ValueError(
                f"{condition}: expected "
                f"{expected_train_count} train graphs, "
                f"found {actual_train_count}."
            )

        if actual_validation_count != 5:
            raise ValueError(
                f"{condition}: expected 5 validation "
                f"graphs, found "
                f"{actual_validation_count}."
            )

        if actual_test_count != 5:
            raise ValueError(
                f"{condition}: expected 5 test graphs, "
                f"found {actual_test_count}."
            )

    return manifest_df


def build_summary(
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[
        dict[str, object]
    ] = []

    for condition_info in CONDITIONS:
        condition = str(
            condition_info[
                "condition"
            ]
        )

        condition_df = manifest_df[
            manifest_df[
                "condition"
            ]
            == condition
        ]

        train_df = condition_df[
            condition_df[
                "split"
            ]
            == "train"
        ]

        validation_df = condition_df[
            condition_df[
                "split"
            ]
            == "validation"
        ]

        test_df = condition_df[
            condition_df[
                "split"
            ]
            == "test"
        ]

        mixed_train_df = train_df[
            train_df[
                "is_added_mixed_graph"
            ]
        ]

        gap_counts = (
            mixed_train_df[
                "gap_level"
            ]
            .value_counts()
            .to_dict()
        )

        summary_rows.append(
            {
                "condition": condition,
                "num_train_graphs": (
                    len(train_df)
                ),
                "num_original_train_graphs": int(
                    (
                        ~train_df[
                            "is_added_mixed_graph"
                        ]
                    ).sum()
                ),
                "num_added_mixed_graphs": (
                    len(
                        mixed_train_df
                    )
                ),
                "num_validation_graphs": (
                    len(validation_df)
                ),
                "num_test_graphs": (
                    len(test_df)
                ),
                "total_train_nodes": int(
                    train_df[
                        "num_nodes"
                    ].sum()
                ),
                "added_gap2_graphs": int(
                    gap_counts.get(
                        2,
                        0,
                    )
                ),
                "added_gap3_graphs": int(
                    gap_counts.get(
                        3,
                        0,
                    )
                ),
                "added_gap4_graphs": int(
                    gap_counts.get(
                        4,
                        0,
                    )
                ),
                "added_gap5_graphs": int(
                    gap_counts.get(
                        5,
                        0,
                    )
                ),
                "validation_graph_ids": (
                    "; ".join(
                        sorted(
                            validation_df[
                                "graph_id"
                            ].astype(str)
                        )
                    )
                ),
                "test_graph_ids": (
                    "; ".join(
                        sorted(
                            test_df[
                                "graph_id"
                            ].astype(str)
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def main() -> None:
    original_df = (
        load_original_dataset()
    )

    mixed_df = (
        load_mixed_dataset_summary()
    )

    selection_df = (
        select_nested_mixed_graphs(
            mixed_df
        )
    )

    manifest_df = (
        build_manifest(
            original_df=original_df,
            selection_df=selection_df,
        )
    )

    summary_df = (
        build_summary(
            manifest_df
        )
    )

    SELECTION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_df = (
        selection_df
        .sort_values(
            [
                "gap_level",
                "num_nodes",
                "graph_id",
            ]
        )
        .reset_index(drop=True)
    )

    manifest_df = (
        manifest_df
        .sort_values(
            [
                "condition",
                "split",
                "is_added_mixed_graph",
                "gap_level",
                "num_nodes",
                "graph_id",
            ]
        )
        .reset_index(drop=True)
    )

    selection_df.to_csv(
        SELECTION_OUTPUT_PATH,
        index=False,
    )

    manifest_df.to_csv(
        MANIFEST_OUTPUT_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print(
        "Week 18 controlled data-scaling "
        "manifest created successfully."
    )
    print(
        "--------------------------------------------"
    )
    print()

    print(
        summary_df[
            [
                "condition",
                "num_train_graphs",
                "num_original_train_graphs",
                "num_added_mixed_graphs",
                "num_validation_graphs",
                "num_test_graphs",
                "total_train_nodes",
                "added_gap2_graphs",
                "added_gap3_graphs",
                "added_gap4_graphs",
                "added_gap5_graphs",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Nested-selection checks:"
    )
    print(
        "  plus12 is contained in plus24: yes"
    )
    print(
        "  plus24 is contained in plus105: yes"
    )
    print(
        "  validation remains cycle size 38: yes"
    )
    print(
        "  test remains cycle size 41: yes"
    )
    print(
        "  no mixed graph uses size 38 or 41: yes"
    )
    print()

    print(
        f"Saved mixed selection to: "
        f"{SELECTION_OUTPUT_PATH}"
    )
    print(
        f"Saved experiment manifest to: "
        f"{MANIFEST_OUTPUT_PATH}"
    )
    print(
        f"Saved manifest summary to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()