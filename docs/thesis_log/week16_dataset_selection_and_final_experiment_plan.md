# Week 16: Dataset Selection and Final GNN Experiment Plan

## Purpose of Week 16

The goal of Week 16 is to consolidate the distance-1 graph coloring study and prepare the final implementation direction before moving more strongly into thesis writing.

After the Week 15 experiments, the main distance-1 pipeline includes:

- ColPack baselines on the expanded graph dataset
- learned node-ordering targets
- GNN training with different target strategies
- learned coloring evaluation on held-out test graphs

The final implementation focus is not to expand the project into many new directions, but to strengthen the existing GNN-based learned-ordering study.

## Dataset Selection Rationale

The dataset was intentionally kept moderate in size. The aim of the thesis is not to perform a large-scale benchmark of graph coloring tools, but to study whether a GNN can learn useful vertex-ordering signals for graph coloring problems derived from sparse matrix structures.

The selected graphs cover different structural cases:

- small sparse graphs
- medium-sized sparse graphs
- larger sparse graphs
- sparse derivative-related patterns
- structural/mechanical matrix patterns
- general sparse benchmark matrices

This gives enough variation to test whether the learned ordering model behaves consistently across different graph structures, while keeping the implementation and analysis manageable within the thesis timeline.

## Included Graph Groups

The current dataset contains 15 graphs:

### Derivative-related or thesis-specific patterns

- `jac_pat`
- `hess_pat`
- `hess_pat_small`

These graphs are directly connected to the sparse derivative motivation of the thesis.

### Structural or mechanical benchmark matrices

- `bcsstk01`
- `bcsstk03`
- `bcsstk04`
- `bcsstk05`
- `bcsstk06`

These matrices provide sparse graph structures from structural engineering-style problems.

### General sparse benchmark graphs

- `ash85`
- `can_24`
- `dwt_234`
- `dwt_361`
- `dwt_419`
- `west0479`
- `sherman1`

These graphs add additional structural variety in graph size, density, and ordering sensitivity.

## Excluded Graphs

Two candidate graphs were excluded during dataset preparation:

- `bcsstm01` was excluded because the converted graph had 48 vertices but 0 edges. Such a graph is not informative for evaluating graph coloring behaviour.
- `bcsstk07` was excluded because it had the same graph structure as `bcsstk06`, so including both would duplicate the same structural case.

## Current Dataset Split

The expanded Week 15 split is:

- 10 training graphs
- 2 validation graphs
- 3 test graphs

The test graphs are:

- `bcsstk06`
- `jac_pat`
- `sherman1`

These test graphs represent different graph structures and sizes, including a larger sparse graph, a derivative-related graph, and a graph where the learned model still has difficulty matching the best ColPack ordering.

## Reason for Not Expanding the Dataset Further

At this point, adding many more matrices may increase implementation complexity without necessarily improving the main thesis contribution. The current focus is to finish the learning-based distance-1 study clearly and leave enough time for analysis and writing.

Therefore, the current dataset is treated as sufficient for the final distance-1 experimental study.

## Final GNN Experiment Direction

The Week 15 results showed that `BEST_AVAILABLE_OF_5` produced the best regression loss among the tested target strategies. However, the improvement in score prediction did not always transfer strongly into better greedy coloring results.

This suggests that exact score regression may not be fully aligned with the final coloring objective. Graph coloring depends mainly on the relative order of vertices, not only on the exact predicted target score.

Therefore, the final Week 16 implementation will test a ranking-oriented GNN training variant using the same `BEST_AVAILABLE_OF_5` target dataset.

The goal is to check whether training the model more directly around ordering relationships improves the final learned coloring quality.

## Expected Final Comparison

The final comparison will include:

- ColPack five-ordering baselines
- `SMALLEST_LAST` target GNN
- `BEST_AVAILABLE_OF_3` target GNN
- `BEST_AVAILABLE_OF_5` target GNN
- `BEST_AVAILABLE_OF_5` ranking-loss GNN

The main evaluation will compare:

- regression or ranking loss
- number of colors produced by learned greedy coloring
- validity of learned colorings
- comparison against ColPack baselines


## Optional Heuristic-Failure Graph Experiment

A possible optional experiment is to include a small constructed or literature-inspired graph where a standard greedy ordering heuristic, such as `SMALLEST_LAST`, performs poorly.

The motivation is to test whether the learned GNN ordering behaves differently from a fixed heuristic on a graph with known ordering sensitivity.

This experiment is not part of the main required pipeline. The main thesis contribution remains the distance-1 learned-ordering study on sparse-matrix-derived graphs. However, if time allows, this optional experiment could provide an additional qualitative analysis of the GNN's behaviour.

The goal would not be to force the GNN to outperform ColPack, but to study whether the learned ordering produces a different coloring pattern on a graph where heuristic choice matters.


## Ranking-Loss GNN Experiment

After the `BEST_AVAILABLE_OF_5` target experiment, a ranking-oriented GNN training variant was tested.

The motivation was that graph coloring depends mainly on the relative ordering of vertices, while the previous model was trained with mean squared error on continuous target scores. Therefore, a pairwise ranking loss was introduced to encourage the model to preserve the relative order between node pairs.

A new training script was created:

```text
src/training/run_week16_best_available_of_5_ranking_gnn_experiments.py
```

The ranking-loss model used the same PyG dataset as the `BEST_AVAILABLE_OF_5 MSE` model:

```text
data/processed/initial_graph_coloring_dataset/pyg_data_week15_best_available_of_5
```

The ranking-loss training summary was saved in:
```text
results/tables/gnn_node_scorer/week16_best_available_of_5_ranking_training_summary.csv
```

The model was trained using five random seeds. The learned coloring evaluation was saved in:
```text
results/tables/gnn_node_scorer/week16_best_available_of_5_ranking_learned_coloring_evaluation.csv
```


All learned colorings were valid.

The comparison with previous target strategies was saved in:
```text
results/tables/gnn_node_scorer/week16_ranking_vs_previous_target_strategy_comparison.csv
```


The final learned-coloring comparison was:
```text
bcsstk06:
  SMALLEST_LAST target:        mean 13.0
  BEST_AVAILABLE_OF_3:         mean 13.2
  BEST_AVAILABLE_OF_5 MSE:     mean 13.2
  BEST_AVAILABLE_OF_5 ranking: mean 13.4

jac_pat:
  SMALLEST_LAST target:        mean 11.0
  BEST_AVAILABLE_OF_3:         mean 11.0
  BEST_AVAILABLE_OF_5 MSE:     mean 11.0
  BEST_AVAILABLE_OF_5 ranking: mean 11.0

sherman1:
  SMALLEST_LAST target:        mean 4.8
  BEST_AVAILABLE_OF_3:         mean 4.8
  BEST_AVAILABLE_OF_5 MSE:     mean 4.4
  BEST_AVAILABLE_OF_5 ranking: mean 4.6
```

The ranking-loss model did not improve the final coloring quality compared with the MSE-based `BEST_AVAILABLE_OF_5` model. The MSE-based `BEST_AVAILABLE_OF_5` model remained the strongest learned model in this experiment, especially on `sherman1`.

This result is still useful because it shows that aligning the loss more directly with ordering is not automatically sufficient. The relation between predicted ordering scores and final greedy coloring quality remains delicate.



## Improved Node Feature Set

To strengthen the learning-based ordering experiment, an improved node feature set was introduced in Week 16. The earlier GNN experiments used five basic structural features: degree, normalized degree, clustering coefficient, core number, and a constant bias feature. While these features captured some local structure, they did not provide enough information about the surrounding coloring pressure of each vertex.

The improved feature set extends the node representation from 5 features to 18 features. The new features include additional information about neighborhood degree statistics, triangle participation, normalized graph-level ranks, core-based neighborhood information, and local edge density.

The complete improved feature set is:

```text
degree
normalized_degree
graph_normalized_degree
clustering_coefficient
core_number
normalized_core_number
triangle_count
normalized_triangle_count
average_neighbor_degree
max_neighbor_degree
min_neighbor_degree
neighbor_degree_std
average_neighbor_core
max_neighbor_core
degree_rank
core_rank
local_edge_density
constant_bias
```

The motivation for this change is that graph coloring quality depends not only on the properties of a vertex itself, but also on the structure of its local neighborhood. For example, a vertex connected to many high-degree or high-core neighbors may be more difficult to color later in a greedy ordering. Similarly, triangle count and local edge density indicate how dense the local neighborhood is, which is directly related to coloring conflicts.

This improved feature set is also motivated by the graph-coloring GNN literature, where graph coloring is treated as a heterophily-oriented task: adjacent vertices should be distinguished rather than made similar. Therefore, providing richer local structural information may help the GNN learn more useful ordering scores for coloring.

The improved feature extractor was implemented separately from the original feature extractor to keep earlier experiments reproducible:

```text
src/training/node_features_week16_improved.py
```

A new PyG dataset was then generated using the improved features and the existing `BEST_AVAILABLE_OF_5` ordering target:

```text
data/processed/initial_graph_coloring_dataset/pyg_data_week16_improved_features_best_available_of_5
```

The resulting node feature matrix has shape `[num_nodes, 18]` for each graph.



## Controlled Larger-Graph Extension

After completing the main 15-graph experiment, I decided to add a small controlled larger-graph extension. The purpose of this step is not to randomly increase the dataset size, but to test whether the learned-ordering approach remains stable when the graph size increases within a structurally related matrix family.

The selected candidates are:

- `bcsstk10`
- `bcsstk14`
- `bcsstk15`

These matrices belong to the same structural stiffness matrix family as several graphs already used in the dataset, including `bcsstk01`, `bcsstk03`, `bcsstk04`, `bcsstk05`, and `bcsstk06`. This makes the extension meaningful because it increases the graph scale while keeping the graph category consistent.

The larger-graph extension will be treated as an additional experiment after the main 15-graph comparison. The intended steps are:

1. Download and inspect the candidate matrices.
2. Convert them into graph representations using the same pipeline.
3. Check graph size, edge count, and possible duplicates.
4. Generate ColPack baselines using the five selected orderings.
5. Build `BEST_AVAILABLE_OF_5` ordering targets.
6. Train one selected learned-ordering model and compare the result with the existing 15-graph setting.

This extension is useful for testing whether the GNN-based ordering model behaves consistently on larger sparse structural matrices.


### Initial Larger-Graph Inspection

The three selected larger structural matrices were downloaded and converted into undirected graph form using the same conversion logic as the earlier dataset. The initial graph statistics are:

| Graph | Matrix rows | Matrix nonzeros | Graph vertices | Graph edges |
|---|---:|---:|---:|---:|
| `bcsstk10` | 1086 | 22070 | 1086 | 10492 |
| `bcsstk14` | 1806 | 63454 | 1806 | 30824 |
| `bcsstk15` | 3948 | 117816 | 3948 | 56934 |

These graphs provide a controlled scale increase compared with the existing `bcsstk06` graph, which has 420 vertices and 3720 graph edges. Since all three new matrices belong to the same structural stiffness family, they are suitable for testing whether the learned-ordering pipeline remains stable on larger sparse structural graphs.


## Larger-Extension 10-Seed Stability Check

After building the larger-extension dataset, I repeated the larger-extension GNN experiment with 10 random seeds. The purpose was to check whether the learned-ordering results were stable or mainly dependent on a small number of random seeds.

The larger-extension dataset contains 18 graphs in total. Two larger structural graphs, `bcsstk10` and `bcsstk14`, were added to the training set, while `bcsstk15` was added as a larger structural test graph. The model used the normalized 15-feature representation and the `BEST_AVAILABLE_OF_5` ordering target.

The 10-seed learned-coloring summary was:

| Graph | Min colors | Max colors | Mean colors | Std. dev. |
|---|---:|---:|---:|---:|
| `bcsstk06` | 13 | 13 | 13.0 | 0.000 |
| `bcsstk15` | 17 | 20 | 17.6 | 0.966 |
| `jac_pat` | 11 | 11 | 11.0 | 0.000 |
| `sherman1` | 5 | 5 | 5.0 | 0.000 |

The results show that the learned ordering was very stable on `bcsstk06`, `jac_pat`, and `sherman1`. On the larger unseen structural graph `bcsstk15`, the model reached the best available ColPack result of 17 colors in several runs, but there was still seed-dependent variation. This suggests that adding larger structurally related graphs improves the structural-family evaluation, but larger graph generalization remains sensitive to training randomness.


## Final Week 16 Learned-Strategy Comparison

At the end of Week 16, I compared all learned-ordering strategies that had been tested so far. The comparison included the original `SMALLEST_LAST` target, the `BEST_AVAILABLE_OF_3` and `BEST_AVAILABLE_OF_5` targets, the ranking-loss variant, the raw and normalized improved-feature variants, the edge-separation variant, and the larger-extension model.

The final comparison showed that no single learned strategy dominated all test graphs. On `bcsstk06`, the `SMALLEST_LAST` model and the larger-extension model both achieved a stable mean of 13 colors. On `jac_pat`, all learned strategies consistently produced 11 colors. On `sherman1`, the strongest result still came from the original `BEST_AVAILABLE_OF_5` model, with a mean of 4.4 colors.

The larger-extension model was especially useful for evaluating structural-family generalization. It achieved stable 13-color results on `bcsstk06` and reached 17 colors on the larger unseen graph `bcsstk15` in several runs. In the 10-seed stability check, `bcsstk15` had a minimum of 17 colors, a mean of 17.6 colors, and a maximum of 20 colors. This suggests that the model can generalize reasonably well to larger structurally related graphs, although the result is still sensitive to random seed on the larger test case.

Overall, the Week 16 experiments support a balanced conclusion: the GNN-based learned ordering can produce valid and competitive colorings, and larger structurally related training data improves stability on structural graphs. However, the experiments also show that generalization across different sparse matrix families remains limited.