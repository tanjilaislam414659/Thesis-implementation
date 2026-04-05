# Greedy Coloring Ordering Benchmark
Date: 2026-03-16

## Objective

The goal of this experiment was to verify the behavior of greedy graph coloring under different vertex ordering strategies. This forms the baseline experiment pipeline that will later be used to compare classical heuristics and learning-based ordering methods.

## Implementation

The following modules were implemented:

- src/utils/graph_utils.py
  - greedy coloring
  - coloring validation
  - color counting
  - graph statistics

- src/graphs/ordering_strategies.py
  - natural ordering
  - reverse ordering
  - random ordering
  - descending degree ordering
  - ascending degree ordering
  - smallest-last ordering

- src/training/run_ordering_benchmark.py
  - experiment pipeline
  - runtime measurement
  - result export to CSV

The benchmark script evaluates greedy coloring results across multiple ordering strategies for several test graphs.

## Graphs Used

The following toy graphs were used for the initial experiments:

- cycle_6
- path_6
- complete_5
- star_6
- custom_6

These graphs were selected to represent different structural properties such as regular graphs, bipartite graphs, and dense graphs.

## Observations

1. All generated colorings were valid.
2. For several graphs (cycle, star, complete graph), all ordering strategies produced the same number of colors.
3. For the path graph, one ordering strategy (ascending degree) produced a worse coloring (3 colors instead of 2).

This confirms that greedy coloring results depend on vertex ordering.

## Interpretation

The experiment confirms the core intuition behind the thesis: the quality of greedy graph coloring can vary depending on the vertex processing order.

While some graph structures are insensitive to ordering, other graphs can produce significantly different colorings.

This observation motivates the exploration of learned ordering strategies using Graph Neural Networks.

## Next Steps

1. Extend the benchmark to include additional graph families.
2. Integrate real-world graphs derived from sparse matrices.
3. Install and test ColPack heuristics for comparison.
4. Begin exploratory experiments with graph neural networks.