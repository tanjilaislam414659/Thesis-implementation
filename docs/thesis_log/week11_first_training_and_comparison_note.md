# Week 11 Note — First GNN Training and Method Comparison

During Week 11, I extended the learning-based graph coloring pipeline from model setup to a first trainable prototype.

## Resolving the `jac_pat` Graph Representation Issue

At the beginning of Week 11, I first resolved the graph representation mismatch for `jac_pat.mtx`.

The Python/PyTorch Geometric pipeline represents this rectangular matrix as a **column intersection graph**. In this representation:

```text
vertices = matrix columns
edges = pairs of columns that have nonzero entries in the same row
```

Earlier, the stored ColPack output for `jac_pat` had been generated from a different graph interpretation. Because of this, the ColPack output did not match the graph used in the Python/PyTorch Geometric pipeline.

To fix this, I exported the Python column intersection graph as a square Matrix Market graph file and reran ColPack on this aligned graph.

After the correction, `jac_pat` has the same structure in both pipelines:

```text
43 vertices
121 edges
```

The corrected ColPack results for `jac_pat` are:

| Graph | Method | Ordering | Colors |
|---|---|---|---:|
| `jac_pat` | ColPack | `LARGEST_FIRST` | 11 |
| `jac_pat` | ColPack | `SMALLEST_LAST` | 11 |

After this correction, I regenerated the `SMALLEST_LAST` ordering targets and attached the corrected target tensor to `jac_pat.pt`.

---

## Split-Aware Dataset Loading

I then created a split-aware loader for the saved PyTorch Geometric graph objects.

The current split setup is:

| Split | Graphs |
|---|---|
| Train | `ash85`, `can_24`, `hess_pat` |
| Validation | `hess_pat_small` |
| Test | `jac_pat` |

All five graph objects now contain:

```text
x = node feature matrix
edge_index = graph connectivity
y = normalized node-level ordering target
```

The loader was tested successfully and confirmed that all train, validation, and test graphs can be loaded correctly.

---

## First GNN Training Script

I implemented the first supervised training script for the `GNNNodeScorer` model.

The model is trained to predict normalized node-level target scores derived from the ColPack `SMALLEST_LAST` ordering.

The training setup is:

```text
input: node features and edge_index
model: two-layer GCN node scorer
target: normalized SMALLEST_LAST ordering score
loss: mean squared error
```

The training script also tracks the best validation loss and saves the best model checkpoint.

The saved checkpoint is stored under:

```text
results/models/gnn_node_scorer/best_gnn_node_scorer.pt
```

A small training summary CSV is also generated under:

```text
results/tables/gnn_node_scorer/initial_training_summary.csv
```

This makes the first training result reproducible and easier to compare with later experiments.

---

## Prediction and Ordering Inspection

Using the saved model checkpoint, I inspected the predicted node scores on the test graph `jac_pat`.

The predicted scores were converted into a learned vertex ordering by sorting nodes in descending order of their predicted score:

```text
higher predicted score
→ earlier position in the learned ordering
```

I compared this learned ordering with the target `SMALLEST_LAST` ordering.

For the test graph `jac_pat`, the first ordering-quality summary was:

| Metric | Value |
|---|---:|
| Mean absolute position error | 3.628 |
| Max absolute position error | 15 |
| Top-10 overlap | 3 / 10 |
| Top-15 overlap | 12 / 15 |

This shows that the first model does not perfectly reproduce the full target ordering, but it already captures many of the high-priority nodes in the top 15.

---

## Learned Coloring Evaluation

I then evaluated the learned GNN ordering by applying greedy coloring using the predicted ordering.

The evaluation pipeline was:

```text
trained GNN
→ predicted node scores
→ learned vertex ordering
→ greedy coloring
→ number of colors
```

For the test graph `jac_pat`, the learned ordering produced:

```text
11 colors
valid coloring: true
```

This means the learned ordering produced a valid coloring and matched the color count of the current ColPack heuristic baselines on the test graph.

---

## First Method Comparison

The first learned-versus-heuristic comparison is:

| Graph | Method | Colors | Valid |
|---|---|---:|---|
| `jac_pat` | ColPack `LARGEST_FIRST` | 11 | true |
| `jac_pat` | ColPack `SMALLEST_LAST` | 11 | true |
| `jac_pat` | GNN learned ordering | 11 | true |

This is the first full comparison result produced by the learning-based pipeline.

The result is still preliminary because the dataset is very small. However, it confirms that the complete pipeline works end to end:

```text
graph
→ node features
→ GNN training
→ predicted node scores
→ learned ordering
→ greedy coloring
→ comparison with heuristic baselines
```

---

## Files Created or Updated

The main Week 11 files include:

```text
src/training/export_column_intersection_graph_for_colpack.py
src/training/load_pyg_splits.py
src/training/train_gnn_node_scorer.py
src/training/check_gnn_checkpoint.py
src/training/inspect_gnn_predictions.py
src/training/compare_predicted_ordering.py
src/training/summarize_predicted_ordering.py
src/training/learned_ordering.py
src/training/ordered_greedy_coloring.py
src/training/evaluate_learned_coloring.py
src/training/build_method_comparison_table.py
```

The main generated result files include:

```text
results/models/gnn_node_scorer/best_gnn_node_scorer.pt
results/tables/gnn_node_scorer/initial_training_summary.csv
results/tables/gnn_node_scorer/jac_pat_predicted_ordering_comparison.csv
results/tables/gnn_node_scorer/jac_pat_predicted_ordering_summary.csv
results/tables/gnn_node_scorer/learned_coloring_evaluation.csv
results/tables/gnn_node_scorer/method_comparison_jac_pat.csv
```

---

## Current Outcome

The main outcome of Week 11 is that the first trainable GNN-based graph coloring pipeline is now working.

The model can be trained, saved, loaded again, used to predict node scores, and evaluated through greedy coloring.

The first learned ordering result on the test graph `jac_pat` matched the current ColPack heuristic baselines in terms of number of colors.

---

## Next Step

The next step is to improve and extend the comparison.

Planned next work includes:

```text
- repeat training and evaluation more systematically,
- save results from multiple runs if needed,
- compare learned orderings on more graphs,
- add more graph instances to make the learning setup more meaningful,
- begin preparing stronger learned-versus-heuristic evaluation tables.
```

This will help move from a working prototype toward more reliable experimental results.