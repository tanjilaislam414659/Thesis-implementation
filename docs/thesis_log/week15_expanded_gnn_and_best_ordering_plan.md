# Week 15: Expanded GNN Dataset, Best-Ordering Targets, and Diagnostic Graph Plan

## Goal

The goal of Week 15 is to continue the distance-1 learning-based graph coloring pipeline using the expanded graph dataset from Weeks 13--14.

After the meeting with Prof. Uwe Naumann, the priority is to strengthen the main distance-1 GNN contribution before moving to distance-2 coloring. The expanded dataset of 15 graphs is sufficient for the current stage. Instead of adding many more matrices, the focus should now be on using the existing graphs carefully, preparing stronger learning targets, and testing whether the GNN can do more than simply reproduce one known heuristic.

## Main Feedback from the Meeting

The current thesis direction is on track. The implementation pipeline already provides a good basis for the thesis.

The main points from the meeting were:

- The current number of graphs is acceptable; the focus should be on relevant graph structures, not just more graphs.
- The selected sparse matrix graphs should be described by their structural categories where possible.
- The main thesis can be built around the distance-1 case if the implementation, experiments, and writing are done well.
- Distance-2 and star coloring remain possible extensions, but they are lower priority.
- The remaining implementation work should be completed soon, so that enough time is left for writing.
- Before moving to distance-2, it is useful to test whether the GNN can behave differently from a standard heuristic in cases where that heuristic is not ideal.

## Updated Week 15 Priorities

The updated priorities are:

1. Keep the current 15-graph dataset as the main experimental dataset.
2. Rebuild the PyTorch Geometric dataset for all 15 graphs.
3. Attach node-level ordering targets from the Week 14 `SMALLEST_LAST` target file.
4. Prepare an additional best-available-ordering target based on the best result among tested ColPack orderings.
5. Consider a small diagnostic graph where a standard heuristic performs poorly.
6. Rerun the GNN training and evaluate the learned ordering against the ColPack baselines.

## Best-Available-Ordering Target

Until now, the supervised GNN target has mainly been based on the `SMALLEST_LAST` ordering. This is reasonable because `SMALLEST_LAST` is a strong and commonly useful heuristic.

However, training only against one heuristic mainly teaches the model to reproduce that heuristic. A stronger experiment is to create a target based on the best available ordering among several tested ColPack orderings.

The current baseline orderings are:

- `SMALLEST_LAST`
- `LARGEST_FIRST`
- `NATURAL`

Additional ordering variants such as reverse natural or other supported ColPack orderings may also be considered if they are easy to generate.

For each graph, the best ordering is selected based on the lowest number of colors. The corresponding node ordering can then be used as an alternative supervised target for the GNN.

## Diagnostic Graph Idea

Another useful experiment is to test the learned ordering on a graph where a standard heuristic such as `SMALLEST_LAST` does not perform well.

The purpose of this experiment is not to replace the sparse matrix dataset, but to better understand the behaviour of the GNN. If the GNN performs differently on such a graph, this can provide useful evidence about whether the learned method is only copying a heuristic or whether it can capture a different ordering behaviour.

This diagnostic graph experiment is prioritized before distance-2 coloring, because it directly strengthens the main GNN contribution of the thesis.

## Distance-2 and Star Coloring

Distance-2 coloring remains a possible Jacobian-related extension. Star coloring remains a possible Hessian-related extension or future-work direction.

However, these should only be started after the main distance-1 GNN experiments are sufficiently complete.

## Expected Week 15 Outcome

By the end of Week 15, the expected outcome is:

- an expanded PyTorch Geometric dataset for the 15 graphs,
- validated node-level targets attached to the graph data,
- a clear baseline target based on `SMALLEST_LAST`,
- a first version of the best-available-ordering target,
- and, if time allows, first GNN training results on the expanded dataset.

## Expanded PyG Dataset Preparation

The expanded PyTorch Geometric dataset was rebuilt for all 15 graphs using the Week 15 split file and the Week 14 `SMALLEST_LAST` ordering targets.

Each graph data object contains node features, graph connectivity, graph-level split information, and node-level target scores. The rebuilt dataset was saved under:

```text
data/processed/initial_graph_coloring_dataset/pyg_data_week15_expanded/
```

A summary table was also created:

```text
data/processed/initial_graph_coloring_dataset/pyg_data_week15_expanded/pyg_week15_expanded_dataset_summary.csv
```

The consistency check confirmed that all PyG node counts, edge counts, and target counts match the Week 13 graph summary.


```text
All node counts match: True
All edge counts match: True
All target counts match nodes: True
All PyG consistency checks passed: True
```


## First Expanded GNN Training Run

The first expanded GNN node scorer experiment was run on the Week 15 PyG dataset using the `SMALLEST_LAST` node-level targets.

The experiment used:

- 10 training graphs,
- 2 validation graphs,
- 3 test graphs,
- 5 random seeds,
- 200 training epochs,
- hidden dimension 32.

An initial tensor-shape warning was found because the model predictions had shape `[num_nodes, 1]`, while the target tensor had shape `[num_nodes]`. This was corrected by flattening both tensors during loss computation.

After this correction, the training ran successfully without warnings.

The repeated-run summary was:

```text
Mean validation loss: 0.057407
Mean test loss:       0.061790
Test loss std:        0.007730
Minimum test loss:    0.052874
Maximum test loss:    0.074143
```


## First Learned Coloring Evaluation

The trained Week 15 GNN models were evaluated by converting the predicted node scores into vertex orderings and applying greedy coloring on the test graphs.

The learned orderings produced valid colorings for all test graphs and all random seeds.

The comparison with ColPack baselines was:

```text
bcsstk06:
  Best ColPack result: 13 colors
  Learned GNN result:  13 colors

jac_pat:
  Best ColPack result: 11 colors
  Learned GNN result:  11 colors

sherman1:
  Best ColPack result: 2 colors
  Learned GNN result:  4--5 colors
```


## Best-Available-Ordering Target Preparation

A second supervised target file was prepared using the best available ColPack ordering among the current Week 14 baseline set:

- `SMALLEST_LAST`
- `LARGEST_FIRST`
- `NATURAL`

For each graph, the ordering with the lowest number of colors was selected. In case of ties, the selection used a stable priority order.

The resulting best-available target file was saved as:

```text
data/processed/initial_graph_coloring_dataset/ordering_targets/best_available_ordering_targets_week15.csv
```


## Target Strategy Comparison

The learned coloring results from the two Week 15 target strategies were compared:

- `SMALLEST_LAST`
- `BEST_AVAILABLE_OF_3`

The comparison table was saved as:

```text
results/tables/gnn_node_scorer/week15_target_strategy_comparison.csv
```

The summary was:

```
bcsstk06:
  SMALLEST_LAST target:     13 colors
  BEST_AVAILABLE_OF_3:      13--14 colors

jac_pat:
  SMALLEST_LAST target:     11 colors
  BEST_AVAILABLE_OF_3:      11 colors

sherman1:
  SMALLEST_LAST target:     4--5 colors
  BEST_AVAILABLE_OF_3:      4--5 colors

```

The best-available target did not clearly improve the learned coloring results on the test graphs. This is understandable because the best-of-3 target was very close to the `SMALLEST_LAST` target for most graphs. Only `bcsstk04` selected `LARGEST_FIRST`, while all other graphs selected `SMALLEST_LAST`.

This result suggests that a richer best-ordering target may require additional ordering variants, such as reverse natural or other tie-breaking variants.


## Best-Available-of-5 Target Experiment

After the initial `BEST_AVAILABLE_OF_3` experiment, two additional ColPack-supported deterministic orderings were tested:

- `DYNAMIC_LARGEST_FIRST`
- `INCIDENCE_DEGREE`

Together with the original three orderings, this created a five-ordering candidate set:

- `SMALLEST_LAST`
- `LARGEST_FIRST`
- `NATURAL`
- `DYNAMIC_LARGEST_FIRST`
- `INCIDENCE_DEGREE`

The combined ColPack benchmark table contains 75 rows, corresponding to 15 graphs and 5 orderings:

```text
results/tables/initial_graph_coloring_benchmarks/colpack_week15_five_ordering_benchmark.csv
```

The best available ordering per graph was saved in:
```text
results/tables/initial_graph_coloring_benchmarks/colpack_week15_best_available_of_5_summary.csv
```


The selected best orderings were:
```text
DYNAMIC_LARGEST_FIRST: 10 graphs
INCIDENCE_DEGREE:      5 graphs
```

This showed that the original best-of-3 target was too narrow, because after adding two additional ColPack orderings, none of the graphs selected `SMALLEST_LAST`, `LARGEST_FIRST`, or `NATURAL` as the final best ordering under the chosen tie-breaking rule.

A new node-level target file was created:
```text
data/processed/initial_graph_coloring_dataset/ordering_targets/best_available_of_5_ordering_targets_week15.csv
```
The target validation passed successfully:
```text
Total target rows: 3563
All row counts match vertices: True
All node IDs unique: True
All order positions unique: True
All score ranges valid: True
```

A new PyG dataset was then built:
```text
data/processed/initial_graph_coloring_dataset/pyg_data_week15_best_available_of_5
```

Compared with the previous target strategies:
```text
SMALLEST_LAST:
  mean validation loss = 0.057407
  mean test loss       = 0.061790

BEST_AVAILABLE_OF_3:
  mean validation loss = 0.059134
  mean test loss       = 0.062660

BEST_AVAILABLE_OF_5:
  mean validation loss = 0.056024
  mean test loss       = 0.060817
```


Thus, BEST_AVAILABLE_OF_5 gave the best regression performance among the three target strategies.

The learned coloring comparison was saved in:
```text
results/tables/gnn_node_scorer/week15_all_target_strategy_comparison.csv
```

The final coloring results were:
```text
bcsstk06:
  SMALLEST_LAST target: 13 colors
  BEST_AVAILABLE_OF_3:  13--14 colors
  BEST_AVAILABLE_OF_5:  13--14 colors

jac_pat:
  SMALLEST_LAST target: 11 colors
  BEST_AVAILABLE_OF_3:  11 colors
  BEST_AVAILABLE_OF_5:  11 colors

sherman1:
  SMALLEST_LAST target: 4--5 colors, mean 4.8
  BEST_AVAILABLE_OF_3:  4--5 colors, mean 4.8
  BEST_AVAILABLE_OF_5:  4--5 colors, mean 4.4
```

The main conclusion is that the richer five-ordering target improved the learning objective and slightly improved the learned coloring result on `sherman1`. However, the improvement in regression loss did not strongly transfer to all final greedy coloring results. This suggests that learning useful ordering scores is possible, but the relation between score prediction and final coloring quality is delicate.
