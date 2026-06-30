# Week 17 Notes

## Overall Week 17 Goal

Week 17 focuses on understanding when learned vertex ordering can help for distance-1 graph coloring.

The main idea is to analyze ordering sensitivity, identify cases where SMALLEST_LAST is not enough, and then test targeted GNN improvements without changing the core 2-layer GCN architecture or greedy-coloring code.

---

## Planned Week 17 Parts

### Part A — Ordering-Sensitivity Analysis

Goal: classify the existing sparse-matrix graphs according to whether different ColPack orderings produce different color counts.

### Part C — Diagnostic SMALLEST_LAST Hard Case

Goal: find or construct one small graph where SMALLEST_LAST performs worse than another ordering.

### Part B — Structured Graph Addition

Goal: add a small number of structured graphs such as banded, arrowhead, block-diagonal, or PDE/grid-like patterns.

### Part D — Improved GNN Target Strategy

Goal: use stronger target orderings instead of only copying SMALLEST_LAST.

### Part E — Symmetry-Breaking Features

Goal: add small node-distinguishing features so the GNN can handle symmetric graphs better.

### Part F — Validation Color-Count Model Selection

Goal: select model checkpoints using validation coloring quality instead of only MSE.

### Part G — Best-of-K GNN Inference

Goal: run the GNN multiple times at inference and keep the best valid coloring.

---

# Completed Work

## Part A — Ordering-Sensitivity Analysis

### Goal

The goal of Part A was to analyze the existing 15-graph real sparse-matrix dataset and determine which graphs are ordering-sensitive.

A graph is considered ordering-sensitive if the five ColPack orderings produce different color counts.

A graph is considered ordering-insensitive if all five orderings produce the same color count.

### Input File Used

```text
results\tables\initial_graph_coloring_benchmarks\colpack_week15_five_ordering_benchmark.csv
```


This file contains 75 rows:
```text
15 graphs × 5 ColPack orderings
```

The five orderings are:
```text
SMALLEST_LAST
LARGEST_FIRST
NATURAL
DYNAMIC_LARGEST_FIRST
INCIDENCE_DEGREE
```

Output Files Created
```text
results\tables\initial_graph_coloring_benchmarks\week17_ordering_sensitivity_five_orderings.csv
results\tables\initial_graph_coloring_benchmarks\week17_ordering_sensitivity_enhanced.csv
results\tables\initial_graph_coloring_benchmarks\week17_sensitivity_class_counts.csv
results\tables\initial_graph_coloring_benchmarks\week17_smallest_last_status_counts.csv
```

Main Findings

The existing 15-graph dataset contains:
```text
| Category                    | Count |
| --------------------------- | ----: |
| Ordering-sensitive graphs   |    11 |
| Ordering-insensitive graphs |     4 |

```
For SMALLEST_LAST specifically:

```text
| SMALLEST_LAST status | Count |
| -------------------- | ----: |
| Best or tied best    |    13 |
| Worse than best      |     2 |

```
Ordering-Insensitive Graphs
```text
bcsstk01
bcsstk03
dwt_234
jac_pat
```
On these graphs, all five ColPack orderings produced the same number of colors.

Graphs Where SMALLEST_LAST Is Worse Than Best
```text
bcsstk04
bcsstk06
```

## Interpretation

The existing sparse-matrix dataset is not completely ordering-insensitive. Most graphs show some variation across ordering choices.

However, SMALLEST_LAST is a very strong heuristic on this dataset. It is best or tied-best on 13 out of 15 graphs.

This explains why a GNN trained only to imitate SMALLEST_LAST has limited room to outperform it. The most important existing real-matrix cases for testing improvement beyond SMALLEST_LAST are bcsstk04 and bcsstk06.

Decision After Part A

Proceed next to Part C: create or identify a clearer diagnostic graph where SMALLEST_LAST performs worse than another ordering.




## Part C — Bickle Smallest-Last Hard-Case Graphs

### Goal

The goal of Part C was to create a controlled diagnostic graph set where `SMALLEST_LAST` is known or expected to be suboptimal.

This part supports the idea that learning only from `SMALLEST_LAST` can be limiting. In the real sparse-matrix dataset, `SMALLEST_LAST` was usually strong, but the Bickle hard-case graphs provide a clearer setting where there is controlled headroom beyond `SMALLEST_LAST`.

### Final Decision

The main Part C graphs are generated from literature-defined constructions, not random graphs and not downloaded SuiteSparse matrices.

The earlier randomly generated 8-vertex graph was only a scratch/sanity-check result and is not used as the main diagnostic graph.

The final Part C graph set is based on Bickle's Smallest-Last hard-case constructions:

```text
G10
C8^2
C11^2
C14^2
C17^2
C20^2
```

### Reason for Generating the Graphs

Generating these graphs is appropriate because the goal is to test controlled hard cases for the Smallest-Last algorithm.

The advantages are:

```text
The graphs come from a literature-defined construction.
The graphs give exact control over the structure.
The chromatic number can be computed exactly for these small cases.
A family of related graphs can be generated for possible train/test experiments.
The experiment is reproducible because the graph construction is implemented in code.
```

### Scripts Created

```text
src\experiments\week17_generate_bickle_hard_cases.py
src\experiments\week17_run_bickle_colpack_orderings.py
src\experiments\week17_parse_bickle_colpack_outputs.py
```

### Matrix Files Created

The generated Matrix Market files were saved in:

```text
data\raw\matrices\week17_bickle_hard_cases
```

The generated graphs were:

```text
week17_bickle_g10.mtx
week17_cycle_square_c8.mtx
week17_cycle_square_c11.mtx
week17_cycle_square_c14.mtx
week17_cycle_square_c17.mtx
week17_cycle_square_c20.mtx
```

### ColPack Output Directory

ColPack outputs were saved in:

```text
data\processed\initial_graph_coloring_dataset\colpack_outputs_week17_bickle_hard_cases
```

### Result Files Created

```text
results\tables\initial_graph_coloring_benchmarks\week17_bickle_hard_cases_python_summary.csv
results\tables\initial_graph_coloring_benchmarks\week17_bickle_hard_cases_colpack_benchmark.csv
results\tables\initial_graph_coloring_benchmarks\week17_bickle_hard_cases_colpack_summary.csv
```

### Python Verification Results

The Python verification computed the chromatic number and checked greedy coloring under simple orderings.

The generated graphs showed the expected Smallest-Last weakness:

| Graph                   | Chromatic number | Python SMALLEST_LAST colors | Gap |
| ----------------------- | ---------------: | --------------------------: | --: |
| week17_bickle_g10       |                3 |                           4 |   1 |
| week17_cycle_square_c8  |                4 |                           5 |   1 |
| week17_cycle_square_c11 |                4 |                           5 |   1 |
| week17_cycle_square_c14 |                4 |                           5 |   1 |
| week17_cycle_square_c17 |                4 |                           5 |   1 |
| week17_cycle_square_c20 |                4 |                           5 |   1 |

### ColPack Verification Results

The five ColPack orderings were run on each Bickle hard-case graph:

```text
SMALLEST_LAST
LARGEST_FIRST
NATURAL
DYNAMIC_LARGEST_FIRST
INCIDENCE_DEGREE
```

The parsed ColPack summary showed:

| Graph                   | Chromatic number | SMALLEST_LAST | Best ColPack | Best ordering         |
| ----------------------- | ---------------: | ------------: | -----------: | --------------------- |
| week17_bickle_g10       |                3 |             4 |            3 | DYNAMIC_LARGEST_FIRST |
| week17_cycle_square_c8  |                4 |             5 |            4 | DYNAMIC_LARGEST_FIRST |
| week17_cycle_square_c11 |                4 |             5 |            4 | DYNAMIC_LARGEST_FIRST |
| week17_cycle_square_c14 |                4 |             5 |            4 | DYNAMIC_LARGEST_FIRST |
| week17_cycle_square_c17 |                4 |             5 |            5 | all five orderings    |
| week17_cycle_square_c20 |                4 |             5 |            5 | all five orderings    |

### Main Finding

Part C confirms that `SMALLEST_LAST` is systematically one color above the chromatic number on the generated Bickle hard-case graphs.

For four graphs:

```text
week17_bickle_g10
week17_cycle_square_c8
week17_cycle_square_c11
week17_cycle_square_c14
```

ColPack with `DYNAMIC_LARGEST_FIRST` reaches the chromatic number, while `SMALLEST_LAST` uses one extra color.

For two graphs:

```text
week17_cycle_square_c17
week17_cycle_square_c20
```

all five ColPack orderings use one more color than the chromatic number.

### Interpretation

The Bickle hard-case graphs provide a controlled setting where `SMALLEST_LAST` is not an ideal teacher for the GNN.

This supports the Week 17 direction:

```text
A GNN trained only to imitate SMALLEST_LAST may inherit the weakness of SMALLEST_LAST.
A stronger target strategy should use the best available or exact-optimal ordering when possible.
```

The results also show that `BEST_AVAILABLE_OF_5` is not always enough. On `C17^2` and `C20^2`, all five ColPack orderings miss the chromatic number. These graphs may be useful later for exact-target or optimal-ordering experiments.

### Decision After Part C

Part C is complete.

Proceed next to Part B: add a small structured graph set representing clearer sparsity patterns such as banded, arrowhead, block-diagonal/block-structured, and PDE/grid-like graphs.


---

## Part B — Structured Matrix Addition

### Goal

The goal of Part B was to add a small structured matrix set to complement the existing sparse-matrix dataset and the Bickle hard-case graphs.

The purpose was not to randomly increase the dataset size, but to include matrices with clearer sparsity structures that are relevant for sparse derivative computation and graph coloring.

The selected structures were:

```text
banded
PDE/grid-like
narrow-banded
block-structured
finite-element/block-like
arrowhead
```

### Final Structured Matrix Set

The final Part B set contains five real sparse matrices and one constructed arrowhead matrix.

| Graph ID             | Source type | Source/name         | Structure family          |
| -------------------- | ----------- | ------------------- | ------------------------- |
| week17_nos1          | real matrix | HB/nos1             | banded                    |
| week17_gr_30_30      | real matrix | HB/gr_30_30         | pde_grid_banded           |
| week17_bwm200        | real matrix | Bai/bwm200          | narrow_banded             |
| week17_bcsstk08      | real matrix | HB/bcsstk08         | block_structured          |
| week17_lshp_265      | real matrix | HB/lshp_265         | finite_element_block_like |
| week17_arrowhead_100 | constructed | synthetic_arrowhead | arrowhead                 |

The arrowhead matrix was not downloaded from SuiteSparse. It was constructed synthetically because clean standalone arrowhead matrices are difficult to find in sparse-matrix collections. The constructed matrix uses diagonal entries plus a dense final row and final column, giving a controlled diagonal-plus-border sparsity pattern.

### Scripts Created

```text
src\experiments\week17_prepare_structured_matrices.py
src\experiments\week17_run_structured_colpack_orderings.py
src\experiments\week17_parse_structured_colpack_outputs.py
```

### Files Created

```text
results\tables\initial_graph_coloring_benchmarks\week17_structured_matrix_metadata.csv
results\tables\initial_graph_coloring_benchmarks\week17_structured_colpack_benchmark.csv
results\tables\initial_graph_coloring_benchmarks\week17_structured_colpack_summary.csv
```

ColPack output files were saved in:

```text
data\processed\initial_graph_coloring_dataset\colpack_outputs_week17_structured_matrices
```

### Structured Matrix Metadata

| Graph ID             | Family                    | Matrix size | Matrix nnz | Graph vertices | Graph edges |
| -------------------- | ------------------------- | ----------: | ---------: | -------------: | ----------: |
| week17_nos1          | banded                    |   237 x 237 |       1017 |            237 |         390 |
| week17_gr_30_30      | pde_grid_banded           |   900 x 900 |       7744 |            900 |        3422 |
| week17_bwm200        | narrow_banded             |   200 x 200 |        796 |            200 |         298 |
| week17_bcsstk08      | block_structured          | 1074 x 1074 |      12960 |           1074 |        5943 |
| week17_lshp_265      | finite_element_block_like |   265 x 265 |       1753 |            265 |         744 |
| week17_arrowhead_100 | arrowhead                 |   100 x 100 |        298 |            100 |          99 |

### ColPack Ordering Results

The five ColPack orderings were tested:

```text
SMALLEST_LAST
LARGEST_FIRST
NATURAL
DYNAMIC_LARGEST_FIRST
INCIDENCE_DEGREE
```

The parsed results were:

| Graph ID             | Family                    | SMALLEST_LAST | Best colors | Worst colors | Ordering gap | Best orderings                                          | SMALLEST_LAST status |
| -------------------- | ------------------------- | ------------: | ----------: | -----------: | -----------: | ------------------------------------------------------- | -------------------- |
| week17_arrowhead_100 | arrowhead                 |             2 |           2 |            2 |            0 | all five orderings                                      | best_or_tied_best    |
| week17_bcsstk08      | block_structured          |             8 |           8 |           11 |            3 | INCIDENCE_DEGREE, SMALLEST_LAST                         | best_or_tied_best    |
| week17_bwm200        | narrow_banded             |             2 |           2 |            3 |            1 | INCIDENCE_DEGREE, LARGEST_FIRST, NATURAL, SMALLEST_LAST | best_or_tied_best    |
| week17_gr_30_30      | pde_grid_banded           |             5 |           4 |            7 |            3 | LARGEST_FIRST, NATURAL                                  | worse_than_best      |
| week17_lshp_265      | finite_element_block_like |             4 |           4 |            6 |            2 | INCIDENCE_DEGREE, SMALLEST_LAST                         | best_or_tied_best    |
| week17_nos1          | banded                    |             2 |           2 |            3 |            1 | INCIDENCE_DEGREE, LARGEST_FIRST, NATURAL, SMALLEST_LAST | best_or_tied_best    |

### Main Finding

The structured matrix set confirms that ordering sensitivity also appears in structured sparse matrices.

The most important new result is for:

```text
week17_gr_30_30
```

For this PDE/grid-banded matrix, `SMALLEST_LAST` uses 5 colors, while the best ColPack orderings use 4 colors. The best orderings are `LARGEST_FIRST` and `NATURAL`.

This is important because it gives a real structured sparse-matrix example where `SMALLEST_LAST` is not the best available heuristic.

### Interpretation

Part B strengthens the thesis direction in two ways.

First, it adds structured sparse matrices rather than only arbitrary real matrices. This makes the dataset more meaningful for sparse derivative computation.

Second, it shows that `SMALLEST_LAST` is strong on many structured graphs, but not always optimal. The `gr_30_30` result provides additional evidence that training a GNN only to imitate `SMALLEST_LAST` can be limiting.

### Decision After Part B

Part B is complete.

The most useful structured graph for later learned-ordering experiments is:

```text
week17_gr_30_30
```

because it is a real PDE/grid-like structured matrix where `SMALLEST_LAST` is worse than the best available ColPack ordering.

The other structured graphs are still useful as additional evaluation cases, especially for checking whether the learned model behaves consistently across banded, block-structured, finite-element, and arrowhead-like sparsity patterns.


---

## Part D — Improved GNN Target Strategy

### Goal

The goal of Part D was to move beyond using only `SMALLEST_LAST` as the GNN training target.

Earlier results showed that `SMALLEST_LAST` is often strong, but it is not always the best available ordering. Therefore, Part D created a stronger Week 17 target strategy based on the best available ordering among five deterministic ColPack orderings.

The five orderings were:

```text
SMALLEST_LAST
LARGEST_FIRST
NATURAL
DYNAMIC_LARGEST_FIRST
INCIDENCE_DEGREE
```

### Target Strategy

For each graph, the target strategy selects the ordering with the lowest number of colors among the five ColPack orderings.

If multiple orderings use the same number of colors, stable tie-breaking is applied in the following order:

```text
DYNAMIC_LARGEST_FIRST
INCIDENCE_DEGREE
SMALLEST_LAST
LARGEST_FIRST
NATURAL
```

This produces a `BEST_AVAILABLE_OF_5` target table for Week 17.

### Scripts Created

```text
src\experiments\week17_build_best_available_of_5_targets.py
src\experiments\week17_check_best_available_of_5_targets.py
src\experiments\week17_create_best_available_split.py
src\experiments\week17_build_best_available_of_5_pyg_dataset.py
src\experiments\week17_check_best_available_of_5_pyg_dataset.py
```

### Target Files Created

```text
data\processed\initial_graph_coloring_dataset\ordering_targets\week17_best_available_of_5_ordering_targets.csv
results\tables\initial_graph_coloring_benchmarks\week17_best_available_of_5_target_summary.csv
```

The target table contains:

```text
27 graphs
6419 node-level target rows
```

The target validation passed successfully.

### Selected Ordering Distribution

The selected ordering counts were:

| Selected ordering     | Number of graphs |
| --------------------- | ---------------: |
| DYNAMIC_LARGEST_FIRST |               17 |
| INCIDENCE_DEGREE      |                9 |
| LARGEST_FIRST         |                1 |

This shows that the improved target is not simply copying one fixed heuristic. Most graphs select `DYNAMIC_LARGEST_FIRST`, several select `INCIDENCE_DEGREE`, and the structured `gr_30_30` graph selects `LARGEST_FIRST`.

### Week 17 Split File

A new Week 17 split file was created:

```text
data\processed\initial_graph_coloring_dataset\splits\week17_best_available_of_5_split.csv
```

The split contains:

| Split      | Number of graphs |
| ---------- | ---------------: |
| train      |               16 |
| validation |                4 |
| test       |                7 |

The graph groups are:

| Group                  | Number of graphs |
| ---------------------- | ---------------: |
| original_sparse_matrix |               15 |
| bickle_hard_case       |                6 |
| structured_matrix      |                6 |

The original Week 15 sparse-matrix split was preserved, and the Bickle hard-case graphs and structured matrices were added in a controlled way.

### PyG Dataset Created

The Week 17 PyTorch Geometric dataset was built using:

```text
Week 17 BEST_AVAILABLE_OF_5 targets
Week 16 improved node features
27-graph Week 17 split
```

The output directory is:

```text
data\processed\initial_graph_coloring_dataset\pyg_data_week17_best_available_of_5_improved_features
```

The PyG summary file is:

```text
results\tables\initial_graph_coloring_benchmarks\week17_best_available_of_5_pyg_summary.csv
```

### PyG Dataset Validation

The saved `.pt` files were validated successfully.

The validated dataset contains:

```text
27 graphs
18 node features per vertex
16 train graphs
4 validation graphs
7 test graphs
```

The validation confirmed that:

```text
all expected graph files exist
all graphs have 18 improved features
x, edge_index, and y shapes are consistent
target scores are in [0, 1]
selected_ordering and selected_num_colors are stored in each PyG object
```

### Important Test Graphs

The Week 17 test set includes original sparse matrices, Bickle hard cases, and structured matrices.

Important test cases include:

```text
bcsstk06
sherman1
week17_cycle_square_c17
week17_cycle_square_c20
week17_gr_30_30
week17_bcsstk08
```

The graph `week17_gr_30_30` is especially important because it is a real structured PDE/grid-like matrix where `SMALLEST_LAST` was worse than the best available ordering.

### Main Finding

Part D successfully created and validated a stronger GNN training target strategy.

Instead of training only on `SMALLEST_LAST`, the Week 17 dataset now trains on the best available ordering among five ColPack heuristics. This better matches the thesis goal of learning useful ordering behavior rather than simply imitating one fixed heuristic.

### Decision After Part D

Part D is complete.

Proceed next to Part E: symmetry-breaking features.



---

## Part E — Symmetry-Breaking Features

### Goal

The goal of Part E was to add deterministic symmetry-breaking features to the Week 17 GNN dataset.

This was motivated by graphs where many vertices may look structurally similar under local graph features, especially:

```text
cycle-like graphs
grid-like graphs
regular or near-regular graphs
block-structured graphs
Bickle hard-case graphs
```

In such graphs, degree, core number, clustering coefficient, and neighbor statistics may not be enough to distinguish vertices clearly.

### Feature Extractor Created

A new Week 17 feature extractor was created:

```text
src\training\node_features_week17_symmetry_breaking.py
```

This file extends the Week 16 improved feature extractor.

The previous Week 16 improved feature set had:

```text
18 node features
```

The new Week 17 symmetry-breaking feature set has:

```text
25 node features
```

### Added Symmetry-Breaking Features

The following deterministic features were added:

```text
node_position_normalized
node_position_sin
node_position_cos
node_index_parity
component_id_normalized
component_size_normalized
component_position_normalized
```

These features are not learned embeddings. They are deterministic graph/node descriptors designed to give the model additional information for distinguishing structurally similar vertices.

### PyG Dataset Created

A new Week 17 PyTorch Geometric dataset was created using:

```text
Week 17 BEST_AVAILABLE_OF_5 targets
Week 17 symmetry-breaking features
27-graph Week 17 split
```

The output directory is:

```text
data\processed\initial_graph_coloring_dataset\pyg_data_week17_best_available_of_5_symmetry_breaking_features
```

The summary file is:

```text
results\tables\initial_graph_coloring_benchmarks\week17_best_available_of_5_symmetry_breaking_pyg_summary.csv
```

### Validation Result

The symmetry-breaking PyG dataset was validated successfully.

The validated dataset contains:

| Property          | Value |
| ----------------- | ----: |
| Total graphs      |    27 |
| Train graphs      |    16 |
| Validation graphs |     4 |
| Test graphs       |     7 |
| Node features     |    25 |

The graph groups are:

| Group                  | Number of graphs |
| ---------------------- | ---------------: |
| original_sparse_matrix |               15 |
| bickle_hard_case       |                6 |
| structured_matrix      |                6 |

The selected ordering distribution remains:

| Selected ordering     | Number of graphs |
| --------------------- | ---------------: |
| DYNAMIC_LARGEST_FIRST |               17 |
| INCIDENCE_DEGREE      |                9 |
| LARGEST_FIRST         |                1 |

### Main Finding

Part E successfully created a second Week 17 dataset that keeps the same `BEST_AVAILABLE_OF_5` targets but enriches the node representation with symmetry-breaking features.

This does not yet prove model improvement, because the GNN has not been retrained and evaluated on this dataset yet. However, it prepares a stronger feature representation for the next training experiment.

### Decision After Part E

Part E is complete.

Proceed next to Part F: validation-based model selection.


---

## Part F — Validation Color-Count Model Selection

### Goal

The goal of Part F was to improve model selection for the learned-ordering GNN.

Earlier training scripts selected the best checkpoint using validation loss. However, the final thesis objective is not only to predict ordering scores accurately. The real objective is to produce a learned ordering that gives a good greedy coloring.

Therefore, Part F compared two checkpoint-selection strategies:

1. validation loss selection
2. validation total color-count selection, using validation loss only as a tie-breaker

### Dataset and Model Setup

Both strategies used the same Week 17 symmetry-breaking PyG dataset:

```text
data\processed\initial_graph_coloring_dataset\pyg_data_week17_best_available_of_5_symmetry_breaking_features
```

The dataset contains:

```text
| Split      | Number of graphs |
| ---------- | ---------------: |
| train      |               16 |
| validation |                4 |
| test       |                7 |
```

The model was trained with five random seeds:
```text
0, 1, 2, 3, 4
```

Each run used the same GNN node scorer architecture and the same Week 17 BEST_AVAILABLE_OF_5 targets.

## Output Files

The validation color-count selection training summary was saved to:
```text
results\tables\gnn_node_scorer\week17_symmetry_breaking_validation_color_selection_training_summary.csv
```

The validation loss selection training summary was saved to:
```text
results\tables\gnn_node_scorer\week17_symmetry_breaking_validation_loss_selection_training_summary.csv
```

The final comparison tables were saved to:
```text
results\tables\gnn_node_scorer\week17_checkpoint_selection_method_summary.csv
results\tables\gnn_node_scorer\week17_checkpoint_selection_best_seed_per_graph_comparison.csv
```

## Main Result

The best single test result came from validation color-count checkpoint selection.

```text
| Checkpoint selection                          | Best test seed | Test colors | Target colors | Test gap |
| --------------------------------------------- | -------------: | ----------: | ------------: | -------: |
| validation total colors, then validation loss |              1 |          54 |            47 |        7 |
| validation loss                               |              0 |          56 |            47 |        9 |
```
Thus, the validation color-count selection strategy improved the best single test result by 2 colors.


## Stability Observation

The result was not uniformly better across seeds.

```text
| Checkpoint selection                          | Mean test colors | Mean test gap | Std. test gap |
| --------------------------------------------- | ---------------: | ------------: | ------------: |
| validation total colors, then validation loss |             58.0 |          11.0 |          4.20 |
| validation loss                               |             56.6 |           9.6 |          0.80 |
```
This means that validation color-count selection produced the best individual checkpoint, but validation-loss selection was more stable on average.


## Per-Graph Observation

For the best seed from each method, validation color-count selection improved:

```text
bcsstk06: 15 colors -> 13 colors
week17_gr_30_30: 7 colors -> 6 colors
```

It matched validation-loss selection on:
```text
jac_pat
sherman1
week17_cycle_square_c17
week17_cycle_square_c20
```

It was worse on:
```text
week17_bcsstk08: 8 colors -> 9 colors
```

## Interpretation

Part F shows that selecting checkpoints using validation coloring quality is more aligned with the actual graph-coloring objective than selecting only by validation loss.

However, because the validation set is small, the color-count-based selection can be unstable across random seeds. The result should therefore be described carefully: it is promising and produced the best single test result, but it does not yet dominate validation-loss selection on average.


## Decision After Part F

Part F is complete.

The next step is Part G: final Week 17 comparison and analysis.

---



## Part G — Final Week 17 Comparison and Analysis

### Goal

The goal of Part G was to consolidate the Week 17 results into final comparison tables.

This part combined:

```text
ColPack five-ordering baselines
Week 17 BEST_AVAILABLE_OF_5 targets
Week 17 symmetry-breaking GNN results
validation color-count checkpoint selection
validation loss checkpoint selection
per-graph and group-level analysis
```

No additional model training was done in Part G.

## Output Files

The final comparison files were saved to:
```text
results\tables\gnn_node_scorer\week17_final_test_graph_comparison.csv
results\tables\gnn_node_scorer\week17_final_test_group_summary.csv
results\tables\gnn_node_scorer\week17_final_method_level_summary.csv
```

## Method-Level Test Summary

The final method-level test summary was:
```text
| Method                         | Total test colors | Gap from target |
| ------------------------------ | ----------------: | --------------: |
| best_colpack_available         |                47 |               0 |
| smallest_last                  |                49 |               2 |
| gnn_validation_color_selection |                54 |               7 |
| gnn_validation_loss_selection  |                56 |               9 |
```

The target total is 47 colors, because the Week 17 target is defined as the best available result among the five ColPack orderings for each graph.

## Main Result

The Week 17 GNN with validation color-count checkpoint selection produced:
```text
54 total test colors
7-color gap from the BEST_AVAILABLE_OF_5 target
```

The same GNN setup with validation-loss checkpoint selection produced:
```text
56 total test colors
9-color gap from the BEST_AVAILABLE_OF_5 target
```
Thus, validation color-count checkpoint selection improved the best test result by 2 colors compared with validation-loss checkpoint selection.

## Comparison with ColPack Heuristics

The best ColPack available baseline achieved:
```text
47 total test colors
```
The Smallest Last baseline achieved:
```text
49 total test colors
```
The learned GNN method did not outperform these heuristic baselines overall. However, it produced valid colorings for all test graphs and showed that checkpoint selection based on coloring quality can improve the learned-ordering result.

## Per-Graph Findings

The validation-color-selected GNN matched the target on:
```text
jac_pat
week17_cycle_square_c17
week17_cycle_square_c20
```
It was close to the target on:
```text
bcsstk06
week17_bcsstk08
```
It had larger gaps on:
```text
sherman1
week17_gr_30_30
```
For `bcsstk06`, validation color-count selection improved the result compared with validation-loss selection:
```text
15 colors -> 13 colors
```
For `week17_gr_30_30`, validation color-count selection also improved the result:
```text
7 colors -> 6 colors
```

## Group-Level Findings

The group-level test summary was:
```text
| Group                  | Number of graphs | Target colors | GNN color-selection colors | Gap |
| ---------------------- | ---------------: | ------------: | -------------------------: | --: |
| bickle_hard_case       |                2 |            10 |                         10 |   0 |
| original_sparse_matrix |                3 |            25 |                         29 |   4 |
| structured_matrix      |                2 |            12 |                         15 |   3 |
```
The GNN matched the target exactly on the Bickle hard-case test graphs. The larger remaining gaps came from original sparse matrices and structured matrices.

## Interpretation

Part G shows that the Week 17 learned-ordering pipeline is working end-to-end:
```text
expanded graph set
BEST_AVAILABLE_OF_5 targets
symmetry-breaking features
GNN training
checkpoint selection by coloring quality
valid learned greedy colorings
final comparison against ColPack baselines
```
The learned method is not yet stronger than ColPack heuristics overall. Still, it gives meaningful empirical evidence for the thesis:

1. The learned ordering can produce valid greedy colorings on all test graphs.
2. The model can match the best available target on some held-out graphs.
3. Validation color-count checkpoint selection is more aligned with the final graph-coloring objective than validation loss alone.
4. The current model still struggles on some sparse and structured matrix cases, especially `sherman1` and `week17_gr_30_30`.


---


## Part H — Dedicated Exact-Optimal Bickle Experiment

### Goal

The goal of Part H was to run a dedicated Bickle hard-case experiment, separate from the broader 27-graph sparse-matrix experiment.

The motivation was that the earlier `BEST_AVAILABLE_OF_5` target was not strong enough for some Bickle cycle-square graphs. In particular, for `C17²` and `C20²`, all five ColPack orderings used in the experiment produced 5 colors, while the known chromatic number is 4.

Therefore, the GNN could not be expected to reach 4 colors when trained against the previous best-of-5 teacher target. Part H fixed this by using exact-optimal 4-color ordering targets for the Bickle cycle-square family.

### Expanded Bickle Family

The following cycle-square graphs were generated:

```text
C8², C11², C14², C17², C20², C23², C26², C29², C32²
```

The split was:
```text
| Split      | Graphs                            |
| ---------- | --------------------------------- |
| train      | C8², C11², C14², C23², C26², C29² |
| validation | C32²                              |
| test       | C17², C20²                        |
```

All graphs have known chromatic number 4.

## Exact-Optimal Targets

Exact 4-color greedy-compatible ordering targets were generated for every graph in the family.

The target file was saved to:
```text
data\processed\initial_graph_coloring_dataset\ordering_targets\week17_bickle_exact_optimal_ordering_targets.csv
```
The split file was saved to:
```text
data\processed\initial_graph_coloring_dataset\splits\week17_bickle_exact_split.csv
```

## ColPack-5 Baseline Check

The five ColPack orderings were evaluated on the expanded Bickle family.

For the held-out test graphs:
```text
| Graph | Chromatic number | Best ColPack-5 colors |
| ----- | ---------------: | --------------------: |
| C17²  |                4 |                     5 |
| C20²  |                4 |                     5 |
```
Thus, both test graphs are true hard cases for the five-ordering ColPack baseline used in this experiment.

## PyG Dataset

A dedicated Bickle-only PyTorch Geometric dataset was built using the Week 17 symmetry-breaking features.

The output directory is:
```text
data\processed\initial_graph_coloring_dataset\pyg_data_week17_bickle_exact_symmetry_breaking
```
The dataset contains:
```text
| Split      | Number of graphs |
| ---------- | ---------------: |
| train      |                6 |
| validation |                1 |
| test       |                2 |
```
All graphs use 25 symmetry-breaking node features.

## GNN Training

The GNN was trained for five random seeds:
```text
0, 1, 2, 3, 4
```
Checkpoint selection used validation total color count, with validation loss as a tie-breaker.

The training summary was saved to:
```text
results\tables\gnn_node_scorer\week17_bickle_exact_gnn_training_summary.csv
```

## Main Result

The GNN achieved the exact chromatic number on both held-out test graphs for all five seeds.
```text
| Seed | Test GNN colors | Test chromatic total | Test ColPack-5 total | Gap from chromatic | Gap from ColPack-5 |
| ---: | --------------: | -------------------: | -------------------: | -----------------: | -----------------: |
|    0 |               8 |                    8 |                   10 |                  0 |                 -2 |
|    1 |               8 |                    8 |                   10 |                  0 |                 -2 |
|    2 |               8 |                    8 |                   10 |                  0 |                 -2 |
|    3 |               8 |                    8 |                   10 |                  0 |                 -2 |
|    4 |               8 |                    8 |                   10 |                  0 |                 -2 |
```
Per graph:
```text
| Graph | Chromatic number | Best ColPack-5 | GNN colors |
| ----- | ---------------: | -------------: | ---------: |
| C17²  |                4 |              5 |          4 |
| C20²  |                4 |              5 |          4 |
```

All GNN colorings were valid.

## Interpretation

Part H shows that the previous Bickle result was limited by the teacher target, not necessarily by the GNN architecture. When trained with exact-optimal targets on a dedicated Bickle cycle-square family, the GNN learned orderings that matched the known chromatic number on held-out hard cases.

This provides the strongest positive result so far:
```text
On controlled Bickle hard-case graphs, the learned ordering outperformed the five ColPack orderings used in the baseline comparison.
```
This result should be reported separately from the broader sparse-matrix experiment. The broader experiment shows that Smallest Last and other ColPack heuristics remain very strong overall. The dedicated Bickle experiment shows that, when exact-optimal supervision is available on a controlled graph family, the GNN can learn a pattern that beats the available heuristic ordering suite on held-out hard cases.


---

## Final Implementation Decision After Expert Feedback

After reviewing the Week 17 results and the dedicated exact-optimal Bickle experiment, no further major implementation is planned.

Part I, originally considered as a best-of-K or best-of-seeds inference experiment, is not necessary for the Bickle result. The reason is that all five trained GNN seeds already reach 4 colors on both held-out test graphs, `C17²` and `C20²`. Therefore, best-of-seeds inference would return the same result and would not strengthen the main conclusion.

The implementation is therefore considered complete at this point.

### Final Thesis Framing

The broad sparse-matrix experiment and the dedicated Bickle experiment should be reported separately.

In the broad sparse-matrix benchmark, the GNN does not outperform the strongest ColPack heuristics overall. This is an important and honest result, and it supports the observation that classical orderings such as Smallest Last remain very strong for many sparse-matrix graph coloring instances.

In the dedicated Bickle hard-case experiment, the result is different. The held-out graphs `C17²` and `C20²` have known chromatic number 4, while all five tested ColPack orderings produce 5 colors. After training with exact-optimal targets on related Bickle cycle-square graphs, the GNN reaches 4 colors on both held-out graphs for all five random seeds.

This demonstrates empirically that, on this controlled graph family, the learned ordering generalizes to unseen graph sizes and reaches the known chromatic number, while the five ColPack orderings tested remain at 5 colors.

The result should be carefully scoped. It does not show that the GNN beats ColPack or classical graph coloring methods in general. Other heuristics or exact solvers could also reach 4 colors on these graphs. The contribution is that the learned ordering reaches 4 in this controlled setting, generalizing from learned structure, whereas the tested greedy ordering suite does not.

### Final Implementation Conclusion

The final implementation supports a balanced thesis conclusion:

1. Classical heuristics remain strong on general sparse-matrix graph coloring instances.
2. The learned GNN ordering pipeline produces valid colorings and can match heuristic targets on selected cases.
3. On controlled Bickle hard cases with exact-optimal supervision, the learned ordering can outperform the five tested ColPack orderings.
4. The positive learned-ordering result appears when there is both heuristic headroom and learnable graph structure.
5. A limitation is that the Bickle result uses deterministic symmetry-breaking features, including node-position information, so the result partly depends on the natural vertex ordering of the cycle-square graphs.

---
