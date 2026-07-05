# Week 17 Additional Analysis: Heuristic Diversity and Best-of-K GNN Inference

## 1. Motivation

After the controlled heuristic-gap experiment, Professor Naumann suggested one additional validation direction.  
The question was whether the GNN can become more reliable in cases where different ColPack heuristics win on different graph instances.

The controlled-gap experiment mainly tested whether a learned ordering can improve over ColPack on hand-crafted graph families where ColPack heuristics are far from the known optimum.  
This additional analysis asks a different question:

> If the best ColPack heuristic varies from graph to graph, can the GNN behave more like a robust learned ordering that tracks the best available heuristic?

This is not the same as beating ColPack on hard cases.  
Here, the comparison target is the **best-of-5 ColPack oracle**, defined as the best coloring result obtained among the five tested ColPack orderings for each graph.

The five ColPack orderings considered are:

- `NATURAL`
- `LARGEST_FIRST`
- `DYNAMIC_LARGEST_FIRST`
- `INCIDENCE_DEGREE`
- `SMALLEST_LAST`

---


## 2. Heuristic-Diversity Scan

The first step was to scan the existing ColPack benchmark results instead of creating new graphs immediately.

For each graph, I recorded:

- the color count obtained by each ColPack ordering,
- the best color count among the five orderings,
- the worst color count among the five orderings,
- the spread between worst and best,
- the ordering or orderings that achieved the best result.

A graph was treated as an interesting heuristic-diversity case if:

```text
worst ColPack color count - best ColPack color count >= 2
```

This filter was used because very small differences, such as 4 vs 5 colors, are technically heuristic variation but not very convincing.

### Scan result

```text
Benchmark CSV rows scanned: 533
Graph cases summarized: 121
Cases with spread >= 2 before deduplication: 28
Clean deduplicated heuristic-diversity cases: 19
```

The clean set was deduplicated so that each graph appears only once.

---

## 3. Winner Counts in the Clean Heuristic-Diversity Set

The clean heuristic-diversity set contains 19 graphs. Different ColPack orderings win or tie across this set.

| Ordering | Winner or tied winner count | Mean spread when winning | Max spread when winning |
|---|---:|---:|---:|
| `SMALLEST_LAST` | 10 | 2.40 | 4 |
| `DYNAMIC_LARGEST_FIRST` | 9 | 2.56 | 4 |
| `INCIDENCE_DEGREE` | 9 | 2.22 | 3 |
| `LARGEST_FIRST` | 5 | 2.80 | 3 |
| `NATURAL` | 3 | 3.00 | 3 |

This confirms that the best heuristic is not always the same. Although `SMALLEST_LAST` and `DYNAMIC_LARGEST_FIRST` are strong overall, other orderings also win or tie on some graphs.

---

## 4. Best-of-5 Oracle on the 19 Clean Diversity Cases

For the 19 clean heuristic-diversity cases, I compared each fixed ColPack ordering against the best-of-5 oracle.

The best-of-5 oracle means:

```text
For each graph:
    choose the smallest color count obtained by any of the five ColPack orderings
```

### Result on 19 clean cases

| Method | Total colors | Gap to best-of-5 oracle | Graphs matching oracle |
|---|---:|---:|---:|
| `BEST_OF_5_ORACLE` | 202 | 0 | 19 |
| `SMALLEST_LAST` | 221 | 19 | 10 |
| `INCIDENCE_DEGREE` | 221 | 19 | 9 |
| `DYNAMIC_LARGEST_FIRST` | 222 | 20 | 9 |
| `LARGEST_FIRST` | 229 | 27 | 5 |
| `NATURAL` | 239 | 37 | 3 |

This shows that no single fixed ColPack heuristic fully matches the best-of-5 oracle across the heterogeneous set. Even the strongest fixed heuristics are 19 colors worse than the oracle over these 19 graphs.

This supports the motivation for a learned method that could, in principle, behave more reliably across graph types.

---

## 5. Coverage by Existing Week 17 Best-of-5 GNN Targets

The existing Week 17 best-of-5 GNN target file was:

```text
data/processed/initial_graph_coloring_dataset/ordering_targets/week17_best_available_of_5_ordering_targets.csv
```

Among the 19 clean heuristic-diversity cases:

```text
Covered by existing best-of-5 target file: 11
Missing from target file: 8
```

### Covered graph IDs

```text
bcsstk04
bcsstk05
bcsstk06
can_24
dwt_361
dwt_419
sherman1
week17_bcsstk08
week17_gr_30_30
week17_lshp_265
west0479
```

### Missing graph IDs

```text
bcsstk10
bcsstk14
bcsstk15
week17_gap_join_c17_join_3
week17_gap_join_c32_join_3
week17_gap_join_c47_join_3
week17_gap_join_c47_join_4
week17_gap_join_c47_join_5
```

The missing graphs were not used in the GNN comparison at this stage, to keep the experiment lightweight.

---

## 6. Common Subset with Existing GNN Evaluation

The existing Week 17 GNN evaluation did not cover all 11 target-covered diversity graphs. The common subset between the heuristic-diversity cases and the existing GNN evaluation contained 6 graphs.

### Common graph IDs

```text
bcsstk06
dwt_419
sherman1
week17_bcsstk08
week17_gr_30_30
week17_lshp_265
```

The following graphs were covered by the target file but missing from the existing GNN evaluation:

```text
bcsstk04
bcsstk05
can_24
dwt_361
west0479
```

For fairness, the GNN and ColPack methods were compared only on the 6 common graphs.

---

## 7. Single-Seed GNN Result on the Common Diversity Subset

First, the existing single-seed GNN results were compared against the fixed ColPack heuristics and the best-of-5 oracle.

### Corrected result on 6 common graphs

| Method | Total colors | Gap to oracle | Graphs matching oracle |
|---|---:|---:|---:|
| `BEST_OF_5_ORACLE` | 36 | 0 | 6 |
| `INCIDENCE_DEGREE` | 38 | 2 | 4 |
| `SMALLEST_LAST` | 38 | 2 | 4 |
| `LARGEST_FIRST` | 43 | 7 | 1 |
| `DYNAMIC_LARGEST_FIRST` | 44 | 8 | 1 |
| `NATURAL` | 47 | 11 | 1 |
| `GNN_LOSS_SELECTED` mean single seed | 47.8 | 11.8 | 0.6 |
| `GNN_COLOR_SELECTED` mean single seed | 48.4 | 12.4 | 1.0 |

The single-seed GNN did not reliably approximate the best-of-5 oracle on this heterogeneous subset. It was worse than the strongest fixed ColPack heuristics and roughly comparable to the weaker fixed heuristics.

This result is important because it separates two different claims:

1. The GNN can perform well on controlled hard cases.
2. The current GNN does not automatically behave like a reliable best-heuristic selector across heterogeneous graphs.

---

## 8. Best-of-K GNN Inference

As a final lightweight inference experiment, I applied best-of-K inference on the same 6-graph heuristic-diversity subset.

The idea is:

```text
For each graph:
    evaluate several trained GNN seeds
    keep the valid coloring with the smallest number of colors
```

This is useful because different trained GNN seeds may produce different vertex orderings. When the model is uncertain, selecting the best valid result across multiple seeds can improve reliability.

Three GNN inference variants were compared:

- `GNN_BEST_OF_5_COLOR_SELECTED_SEEDS`
- `GNN_BEST_OF_5_LOSS_SELECTED_SEEDS`
- `GNN_BEST_OF_10_COLOR_AND_LOSS_SELECTED`

The best-of-10 version uses both the color-selected and loss-selected checkpoint sets.

---

## 9. Best-of-K Result

### Result on 6 common heuristic-diversity graphs

| Method | Total colors | Gap to oracle | Graphs matching oracle |
|---|---:|---:|---:|
| `BEST_OF_5_ORACLE` | 36 | 0 | 6 |
| `INCIDENCE_DEGREE` | 38 | 2 | 4 |
| `SMALLEST_LAST` | 38 | 2 | 4 |
| `GNN_BEST_OF_10_COLOR_AND_LOSS_SELECTED` | 42 | 6 | 2 |
| `GNN_BEST_OF_5_COLOR_SELECTED_SEEDS` | 43 | 7 | 1 |
| `LARGEST_FIRST` | 43 | 7 | 1 |
| `DYNAMIC_LARGEST_FIRST` | 44 | 8 | 1 |
| `GNN_BEST_OF_5_LOSS_SELECTED_SEEDS` | 44 | 8 | 2 |
| `NATURAL` | 47 | 11 | 1 |
| `GNN_LOSS_SELECTED` mean single seed | 47.8 | 11.8 | 0.6 |
| `GNN_COLOR_SELECTED` mean single seed | 48.4 | 12.4 | 1.0 |

Best-of-K inference substantially improved the GNN result.

The best GNN variant was:

```text
GNN_BEST_OF_10_COLOR_AND_LOSS_SELECTED
```

It achieved:

```text
Total colors: 42
Gap to oracle: 6
Graphs matching oracle: 2 / 6
```

Compared with the mean single-seed GNN:

```text
GNN_COLOR_SELECTED mean single seed:
Total colors: 48.4
Gap to oracle: 12.4

GNN_BEST_OF_10:
Total colors: 42
Gap to oracle: 6
```

Thus, best-of-K inference nearly halved the GNN gap to the best-of-5 oracle.

---

## 10. Per-Graph Best-of-K Result

For the best combined GNN inference strategy:

```text
GNN_BEST_OF_10_COLOR_AND_LOSS_SELECTED
```

the selected per-graph results were:

| Graph | Selected candidate | GNN colors | Oracle colors | Gap |
|---|---|---:|---:|---:|
| `bcsstk06` | `GNN_COLOR_SELECTED_seed_1` | 13 | 12 | 1 |
| `dwt_419` | `GNN_COLOR_SELECTED_seed_0` | 6 | 6 | 0 |
| `sherman1` | `GNN_COLOR_SELECTED_seed_0` | 4 | 2 | 2 |
| `week17_bcsstk08` | `GNN_LOSS_SELECTED_seed_0` | 8 | 8 | 0 |
| `week17_gr_30_30` | `GNN_COLOR_SELECTED_seed_1` | 6 | 4 | 2 |
| `week17_lshp_265` | `GNN_COLOR_SELECTED_seed_0` | 5 | 4 | 1 |

The GNN matched the best-of-5 oracle on two graphs:

```text
dwt_419
week17_bcsstk08
```

---

## 11. Interpretation

This additional analysis gives a mixed but useful result.

The heuristic-diversity scan shows that the existing benchmark does contain cases where different ColPack heuristics win, and where the spread between the best and worst heuristic is non-trivial.

The single-seed GNN does not yet behave like a reliable best-of-5 heuristic selector. However, best-of-K inference improves the learned method considerably.

The best-of-10 GNN result:

- improves strongly over the mean single-seed GNN,
- outperforms `NATURAL`,
- outperforms `DYNAMIC_LARGEST_FIRST`,
- matches or slightly improves over `LARGEST_FIRST`,
- but still remains behind the strongest fixed heuristics, `INCIDENCE_DEGREE` and `SMALLEST_LAST`, on this subset.

An important reason is that the strongest fixed heuristics are already very close to the best-of-5 oracle on the common subset. Both `INCIDENCE_DEGREE` and `SMALLEST_LAST` are only 2 colors above the oracle total. Therefore, there is limited remaining headroom for a learned best-of-heuristics selector to improve over these strong fixed heuristics.

This makes the result more understandable: the GNN does not fully match the best fixed heuristics in this setting, but the comparison is already against heuristics that are nearly oracle-level on this subset.

Therefore, the final conclusion should be careful:

> Best-of-K inference improves the reliability of the learned ordering on the heuristic-diversity subset, but the current GNN still does not fully match the best-of-5 ColPack oracle or the strongest fixed heuristics on this small heterogeneous subset. The result also suggests that, when strong fixed heuristics are already close to the oracle, the available improvement margin for a learned selector may be small.

---

## 12. Relation to the Controlled-Gap Experiment

This experiment should be presented separately from the controlled-gap experiment.

### Controlled-gap experiment

Main question:

```text
Can the GNN beat ColPack on hand-crafted cases where ColPack has a verified heuristic gap?
```

Main finding:

```text
Yes. On the controlled gap family, the GNN substantially reduces the gap to the known optimum and beats the best ColPack result.
```

### Heuristic-diversity experiment

Main question:

```text
Can the GNN behave like a robust best-of-5 heuristic selector when different ColPack heuristics win on different graphs?
```

Main finding:

```text
Partially. A single GNN seed does not reliably do this, but best-of-K inference improves the result and makes the GNN competitive with several fixed heuristics.
```

These are two different claims. Keeping them separate makes the thesis interpretation clearer.

---

## 13. Thesis-Ready Wording

A possible thesis/report paragraph:

> As an additional validation, I considered a heuristic-diversity subset in which the best ColPack ordering varies across graphs and the spread between the best and worst ColPack ordering is at least two colors. The scan identified 19 clean graph cases with non-trivial heuristic diversity. On this set, the best-of-five ColPack oracle used 202 colors, while the strongest fixed ColPack heuristics used 221 colors, showing that no single fixed ordering fully matches the per-graph best heuristic behavior.
>
> For the subset already covered by the Week 17 GNN evaluation, the single-seed GNN did not reliably approximate the best-of-five oracle. However, best-of-K inference across multiple trained GNN seeds improved the result substantially, at the cost of multiple inference runs. On the common six-graph subset, the mean single-seed GNN had a gap of 12.4 colors to the oracle, while the best-of-10 GNN reduced this gap to 6 colors. This best-of-K variant outperformed NATURAL, DYNAMIC_LARGEST_FIRST, and LARGEST_FIRST, but remained behind the strongest fixed heuristics, INCIDENCE_DEGREE and SMALLEST_LAST.
>
> This result should be interpreted with the available headroom in mind. On the common subset, the strongest fixed heuristics were already only 2 colors above the best-of-five oracle. Therefore, the learned method was being compared against heuristics that were already close to oracle-level performance. These results suggest that best-of-K inference improves the reliability of learned orderings, while fully matching the per-graph best ColPack heuristic remains an open direction.

---

## 14. Limitations

This additional analysis has several limitations:

1. The GNN comparison was performed only on the 6 common graphs that were already covered by existing GNN evaluation results.
2. The 19-graph clean heuristic-diversity set is useful for scanning ColPack behavior, but not all of these graphs currently have GNN target/evaluation coverage.
3. The best-of-K inference uses multiple trained seeds, so it is more expensive than using a single learned ordering.
4. The best-of-K GNN improves the result but still does not match the strongest fixed ColPack heuristics on this subset.
5. A stronger version of this experiment would require a training objective specifically designed for best-of-heuristics behavior or a larger heterogeneous training set.

---

## 15. Generated Scripts

The following scripts were used for this analysis:

```text
src/experiments/week17_scan_colpack_heuristic_diversity.py
src/experiments/week17_build_clean_heuristic_diversity_set.py
src/experiments/week17_summarize_clean_heuristic_diversity_oracle.py
src/experiments/week17_summarize_gnn_on_heuristic_diversity_set.py
src/experiments/week17_summarize_gnn_on_heuristic_diversity_common_subset.py
src/experiments/week17_best_of_k_gnn_heuristic_diversity.py
```

---

## 16. Generated Result Files

The following result files were generated:

```text
results/tables/gnn_node_scorer/week17_colpack_heuristic_diversity_per_graph.csv
results/tables/gnn_node_scorer/week17_colpack_heuristic_diversity_winner_counts.csv
results/tables/gnn_node_scorer/week17_colpack_heuristic_diversity_interesting_cases.csv

results/tables/gnn_node_scorer/week17_clean_heuristic_diversity_cases.csv
results/tables/gnn_node_scorer/week17_clean_heuristic_diversity_winner_counts.csv
results/tables/gnn_node_scorer/week17_clean_heuristic_diversity_oracle_summary.csv

results/tables/gnn_node_scorer/week17_heuristic_diversity_covered_cases.csv
results/tables/gnn_node_scorer/week17_heuristic_diversity_gnn_vs_oracle_summary.csv
results/tables/gnn_node_scorer/week17_heuristic_diversity_gnn_seed_summary.csv
results/tables/gnn_node_scorer/week17_heuristic_diversity_gnn_per_graph_results.csv

results/tables/gnn_node_scorer/week17_heuristic_diversity_common_gnn_cases.csv
results/tables/gnn_node_scorer/week17_heuristic_diversity_common_gnn_vs_oracle_summary.csv
results/tables/gnn_node_scorer/week17_heuristic_diversity_common_gnn_seed_summary.csv

results/tables/gnn_node_scorer/week17_best_of_k_gnn_heuristic_diversity_summary.csv
results/tables/gnn_node_scorer/week17_best_of_k_gnn_heuristic_diversity_per_graph.csv
```

---

## 17. Final Takeaway

This additional analysis answers the professor’s suggestion in a lightweight way.

The main takeaway is:

> The existing benchmark contains meaningful heuristic diversity. A single fixed ColPack heuristic does not always match the best-of-5 oracle. The current single-seed GNN does not yet reliably act as a best-heuristic selector, but best-of-K inference substantially improves the learned result and makes it competitive with several fixed ColPack orderings. This provides a useful additional validation and a clear direction for future improvement.