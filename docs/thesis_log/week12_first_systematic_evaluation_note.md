# Week 12 Note — First Systematic Distance-1 Evaluation

During Week 12, I extended the first learned-vs-heuristic comparison into a more systematic evaluation.

The goal was not yet to produce final thesis-level evidence, because the current dataset is still small. Instead, the goal was to validate the full experimental pipeline more carefully and check whether the learned GNN ordering behaves consistently across repeated training runs.

## Evaluation Setup

The current split is:

| Split | Graphs |
|---|---|
| Train | `ash85`, `can_24`, `hess_pat` |
| Validation | `hess_pat_small` |
| Test | `jac_pat` |

The GNN model is trained to predict normalized node-level scores derived from the ColPack `SMALLEST_LAST` ordering.

The predicted node scores are converted into a learned vertex ordering by sorting nodes in descending order of predicted score:

```text
higher predicted score
→ earlier position in the learned ordering
```

Greedy coloring is then applied using this learned ordering.

The main evaluation metric is:

```text
number of colors
```

Each produced coloring is also checked for validity.

---

## Repeated GNN Training Runs

To check stability, I trained the GNN model five times with different random seeds:

```text
0, 1, 2, 3, 4
```

The training and test losses varied across seeds, which shows that the current small dataset is sensitive to random initialization.

The aggregate loss summary over the five runs was:

| Metric | Value |
|---|---:|
| Number of runs | 5 |
| Mean train loss | 0.125702 |
| Mean validation loss | 0.085873 |
| Mean test loss | 0.446206 |
| Minimum test loss | 0.071342 |
| Maximum test loss | 0.852253 |
| Best test seed | 0 |
| Best validation seed | 2 |

One important observation is that the best validation seed was not the same as the best test seed. This suggests that the current validation setup is still too small to reliably predict test behavior.

---

## Learned Coloring Results

For the test graph `jac_pat`, the learned GNN ordering produced stable coloring results across all five runs.

| Method | Runs | Min Colors | Max Colors | Mean Colors | Valid |
|---|---:|---:|---:|---:|---|
| GNN learned ordering | 5 | 11 | 11 | 11.0 | true |

This means that all five learned orderings produced valid colorings with 11 colors.

Although the score-regression losses varied across random seeds, the final coloring result stayed stable. This is useful because it shows that exact score prediction and final coloring quality are related, but not identical.

---

## Comparison with ColPack and NetworkX Baselines

The learned GNN ordering was compared with both ColPack and NetworkX heuristic baselines.

The full summary on the current test graph `jac_pat` is:

| Method | Runs | Min Colors | Max Colors | Mean Colors | Valid |
|---|---:|---:|---:|---:|---|
| ColPack `LARGEST_FIRST` | 1 | 11 | 11 | 11.0 | true |
| ColPack `SMALLEST_LAST` | 1 | 11 | 11 | 11.0 | true |
| NetworkX `largest_first` | 1 | 11 | 11 | 11.0 | true |
| NetworkX `smallest_last` | 1 | 11 | 11 | 11.0 | true |
| NetworkX `random_sequential` | 1 | 11 | 11 | 11.0 | true |
| GNN learned ordering | 5 | 11 | 11 | 11.0 | true |

The learned GNN ordering matched all current heuristic baselines in terms of number of colors on the held-out test graph.

---

## Runtime Observation

Runtime was also recorded for the learned GNN ordering and Python heuristic baselines.

The current runtime values are very small because the graph is small. Therefore, these numbers should be treated only as rough pipeline measurements, not final runtime evidence.

The mean runtime for the five learned GNN coloring evaluations was approximately:

```text
0.012883 seconds
```

The first GNN run had a higher runtime than the later runs, likely due to one-time initialization or loading overhead. More reliable runtime comparison will require larger graphs and more systematic measurement.

---

## Main Observation

The first systematic comparison shows that the learned GNN ordering matches the heuristic baselines on the current held-out test graph in terms of number of colors.

The most important observation is:

```text
The GNN score-prediction losses vary across seeds,
but the final greedy coloring result remains stable.
```

This suggests that the learned ordering does not need to reproduce the exact target ordering perfectly in order to produce the same color count.

---

## Current Limitations

The main limitation is the very small dataset size.

At the moment, the evaluation uses only one test graph:

```text
jac_pat
```

Therefore, the result should be treated as pipeline validation and early evidence, not as a final conclusion about model quality.

Other current limitations are:

```text
- only one validation graph is available,
- only one test graph is available,
- the GNN is trained on only three graphs,
- the current model is still a simple two-layer GCN,
- runtime measurements are not yet meaningful for very small graphs,
- no larger sparse matrix collection has been included yet.
```

---

## Current Outcome

The current result confirms that the experimental pipeline works end to end:

```text
graph
→ node features
→ GNN training
→ predicted node scores
→ learned ordering
→ greedy coloring
→ comparison with heuristic baselines
```

This is an important milestone because the thesis now has a working learned-vs-heuristic comparison pipeline.

---

## Next Step

The next step is to expand the dataset with more graph instances and repeat the same evaluation on a larger and more diverse set of graphs.

This will make it possible to better judge whether the learned ordering approach can generalize beyond the current small test setup.

Planned next work includes:

```text
- add more sparse matrix graphs,
- generate ColPack baselines and ordering targets for the new graphs,
- update train/validation/test splits,
- rerun the repeated GNN experiments,
- compare learned orderings against heuristic baselines on more test graphs,
- inspect cases where the learned ordering matches or differs from the heuristics.
```