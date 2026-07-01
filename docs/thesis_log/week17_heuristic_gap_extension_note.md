# Week 17 Extension: Controlled Heuristic-Gap Experiment

## Motivation

After the Week 15--17 progress report, Prof. Naumann suggested that the GNN may generalize better on instances where the gap between the best known coloring and the ColPack heuristic results is larger.

To test this conjecture, I added a controlled family of hand-crafted graph coloring instances. The goal was not only to test the GNN on more graphs, but to create graph instances where the heuristic gap can be controlled and verified directly.

This experiment is useful because the earlier sparse matrix experiments show the method on realistic sparse derivative-related graphs, while this controlled experiment tests whether the GNN can close a deliberately constructed gap between known good colorings and standard heuristic orderings.

## Graph family

The experiment is based on square cycle graphs \(C_n^2\), especially cases where

\[
n = 3r + 2.
\]

For these graphs, a known 4-coloring can be constructed using the repeated pattern

\[
0, 1, 2
\]

followed by a final correction pattern. This gives a known coloring with 4 colors.

To create larger graphs with larger heuristic gaps, I constructed joins of several copies of these cycle-square graphs. For a join of \(k\) copies, every vertex in one copy is connected to every vertex in every other copy. Therefore, each copy must use a disjoint set of colors.

Since each copy needs 4 colors, the known coloring for a join of \(k\) copies uses

\[
4k
\]

colors.

## Verified controlled gaps

For each generated graph, I ran five ColPack ordering strategies:

- NATURAL
- LARGEST_FIRST
- DYNAMIC_LARGEST_FIRST
- INCIDENCE_DEGREE
- SMALLEST_LAST

The final clean controlled family used the base cycle sizes

\[
20, 26, 29, 35, 38, 41
\]

and gap levels

\[
1, 2, 3, 4, 5.
\]

The verified gaps were:

| Gap level | Known colors | Best ColPack-5 colors | Verified gap |
|---:|---:|---:|---:|
| 1 | 4 | 5 | 1 |
| 2 | 8 | 10 | 2 |
| 3 | 12 | 15 | 3 |
| 4 | 16 | 20 | 4 |
| 5 | 20 | 25 | 5 |

Here, "Best ColPack-5" means the best result among the five tested ColPack ordering strategies.

Some generated graphs were excluded from the clean controlled family. For example, \(C_{23}^2\) had gap 0 because Dynamic Largest First reached the known optimum. Some joins based on \(C_{17}^2\) and \(C_{32}^2\) were also excluded because Dynamic Largest First partially reduced the intended gap.

These exclusions are important because the goal was to build a controlled experiment where the independent variable is the verified heuristic gap. Therefore, I only kept graph families where ColPack produced the intended gap after verification.

## Dataset split

The clean dataset was split by base cycle size:

| Split | Base sizes | Number of graphs |
|---|---|---:|
| Train | 20, 26, 29, 35 | 20 |
| Validation | 38 | 5 |
| Test | 41 | 5 |

This split tests whether the GNN can generalize to an unseen base cycle size while still facing the same controlled gap levels.

## GNN setup

The GNN was trained to predict an exact-optimal color-class ordering. The target ordering was generated from the known coloring by ordering vertices first by color class and then by node id.

The same 25-dimensional Week 17 symmetry-breaking node features were used.

Training was repeated for five random seeds:

\[
0, 1, 2, 3, 4.
\]

The best checkpoint for each seed was selected using validation coloring quality. Validation loss was used only as a tie-breaker.

## Original unseen test result on base size 41

On the unseen test graphs with base cycle size 41, the total known target color count was 60 and the total best ColPack-5 color count was 75.

Across five random seeds, the GNN achieved:

| Metric | Value |
|---|---:|
| Mean GNN total colors | 65.0 |
| Standard deviation | 1.0 |
| Best GNN total colors | 64 |
| Worst GNN total colors | 66 |
| Target total colors | 60 |
| ColPack-5 total colors | 75 |
| Mean colors saved vs ColPack-5 | 10.0 |

This shows that the GNN consistently improved over the best result among the five tested ColPack orderings.

## Additional unseen test size

To strengthen the evaluation, I added two further unseen candidate base sizes:

\[
44 \quad \text{and} \quad 47.
\]

Both satisfy \(n = 3r + 2\), so the same known 4-coloring construction applies to \(C_n^2\).

After generating the graphs, I again verified the heuristic gaps using ColPack. The base size 44 produced the intended clean gaps 1--5 and was therefore kept as an additional unseen test size. The base size 47 was excluded from the clean controlled-gap evaluation because Dynamic Largest First reduced the intended gaps for the joined graphs.

This confirms that the gap levels were not assumed from construction alone, but verified experimentally before evaluation.

The trained GNN models were not retrained. The same five checkpoints trained on base sizes 20, 26, 29, and 35 and selected using validation base size 38 were evaluated directly on the additional unseen base size 44.

## Extra unseen test result on base size 44

On the additional unseen test graphs with base cycle size 44, the total known target color count was again 60 and the total best ColPack-5 color count was 75.

Across five random seeds, the GNN achieved:

| Metric | Value |
|---|---:|
| Mean GNN total colors | 65.8 |
| Standard deviation | 0.84 |
| Best GNN total colors | 65 |
| Worst GNN total colors | 67 |
| Target total colors | 60 |
| ColPack-5 total colors | 75 |
| Mean colors saved vs ColPack-5 | 9.2 |

This confirms that the learned ordering also generalizes to the larger unseen base size 44 without retraining.

## Combined C41 and C44 result

Combining the original unseen test base size 41 with the additional unseen test base size 44, the evaluation contains two unseen base sizes, five gap levels, and five random seeds.

Both test base sizes are larger than all training base sizes, which were

\[
20, 26, 29, 35.
\]

Therefore, this evaluation tests size extrapolation rather than only interpolation.

| Metric | Value |
|---|---:|
| Number of unseen base sizes | 2 |
| Gap levels | 1--5 |
| Test graphs per seed | 10 |
| Total evaluation runs | 50 |
| Target total colors per seed | 120 |
| ColPack-5 total colors per seed | 150 |
| Mean GNN total colors per seed | 130.8 |
| Standard deviation | 0.45 |
| Mean colors saved vs ColPack-5 | 19.2 |

The per-gap comparison gives the clearest interpretation. The following table reports the mean result over both unseen test base sizes, 41 and 44, using five random seeds for each graph.

| Gap level | ColPack error above optimum | Mean GNN error above optimum | Mean colors saved vs ColPack-5 |
|---:|---:|---:|---:|
| 1 | 1 | 0.5 | 0.5 |
| 2 | 2 | 1.0 | 1.0 |
| 3 | 3 | 1.5 | 1.5 |
| 4 | 4 | 0.9 | 3.1 |
| 5 | 5 | 1.5 | 3.5 |

## Interpretation

The most important result is not only that the GNN saves colors compared with ColPack. Since larger joined graphs naturally provide more room for saving colors, the stronger comparison is the error above the known optimum.

For the controlled test graphs, the ColPack error above the known optimum increases directly with the constructed gap level:

\[
1, 2, 3, 4, 5.
\]

In contrast, the GNN error above the known optimum remains roughly around one color when averaged over both unseen test sizes and five random seeds:

\[
0.5, 1.0, 1.5, 0.9, 1.5.
\]

The GNN error is not perfectly monotonic or exactly constant. However, it also does not grow linearly with the constructed heuristic gap. This is the main evidence supporting the advisor's conjecture.

While the fixed ColPack orderings become increasingly far from the known optimum, the learned ordering stays close to the optimum even on larger unseen graph sizes. This suggests that the GNN learned useful structural information for ordering vertices, rather than only memorizing the training graph sizes.

A careful thesis statement would be:

> On the controlled heuristic-gap family, the ColPack error increases directly with the constructed gap level, while the GNN remains roughly around one color above the known optimum across two larger unseen test sizes. Across five random seeds and 50 evaluation runs, the GNN used on average 130.8 colors compared with 150 colors for the best ColPack result and 120 colors for the known constructed optimum. This supports the conjecture that learned orderings are most useful on instances where standard heuristic orderings are farther from the optimum.

## Limitations

This result should be interpreted as supportive evidence rather than a general scaling law. The controlled evaluation uses two unseen base sizes and five gap levels, which is stronger than a single test family, but the graph family is still hand-crafted.

The hand-crafted nature is intentional. The purpose of this experiment is to directly test whether the GNN behaves better when the heuristic gap is controlled and increased.

The GNN error is also not perfectly flat across gap levels. However, the important observation is not exact constancy, but that the GNN error remains small and does not increase in the same linear way as the ColPack error.

The excluded \(C_{47}^2\)-based joins are also important to report, because they show that only graphs with verified ColPack gaps were used in the final controlled evaluation.

## Generated files

Important output files from this experiment:

- `data/processed/initial_graph_coloring_dataset/ordering_targets/week17_heuristic_gap_exact_optimal_ordering_targets.csv`
- `data/processed/initial_graph_coloring_dataset/splits/week17_heuristic_gap_split.csv`
- `data/processed/initial_graph_coloring_dataset/pyg_data_week17_heuristic_gap_symmetry_breaking`
- `data/processed/initial_graph_coloring_dataset/ordering_targets/week17_heuristic_gap_extra_c44_exact_optimal_ordering_targets.csv`
- `data/processed/initial_graph_coloring_dataset/splits/week17_heuristic_gap_extra_c44_split.csv`
- `data/processed/initial_graph_coloring_dataset/pyg_data_week17_heuristic_gap_extra_c44_symmetry_breaking`
- `results/tables/initial_graph_coloring_benchmarks/week17_heuristic_gap_colpack_summary.csv`
- `results/tables/initial_graph_coloring_benchmarks/week17_heuristic_gap_exact_target_summary.csv`
- `results/tables/initial_graph_coloring_benchmarks/week17_heuristic_gap_extra_c44_exact_target_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_gnn_training_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_gnn_per_graph_results.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_gnn_per_gap_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_gnn_overall_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_extra_c44_gnn_per_graph_results.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_extra_c44_gnn_per_gap_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_extra_c44_gnn_overall_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_combined_c41_c44_per_graph_results.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_combined_c41_c44_per_gap_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_combined_c41_c44_per_base_summary.csv`
- `results/tables/gnn_node_scorer/week17_heuristic_gap_combined_c41_c44_overall_summary.csv`

## Summary

This controlled heuristic-gap experiment directly addresses the advisor's conjecture. The results suggest that the GNN learned ordering is especially useful when standard ColPack heuristic orderings are farther from the known optimum.

The strongest finding is that ColPack error grows with the constructed gap level, while the GNN error remains roughly around one color above the known optimum across two larger unseen test sizes.