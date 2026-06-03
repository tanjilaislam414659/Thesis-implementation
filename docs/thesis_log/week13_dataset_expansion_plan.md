# Week 13 Dataset Expansion Plan

## Goal

The goal of Week 13 is to expand the current sparse graph dataset so that the learned distance-1 coloring pipeline can be evaluated on a more meaningful and structurally diverse set of graphs.

The current dataset was sufficient for pipeline validation, but it is too small for strong conclusions about generalization. Therefore, the next step is to add more sparse matrix graphs and use them for a larger learned-versus-heuristic comparison.

## Graph Selection Criteria

New graph instances should satisfy the following criteria:

1. The graph should come from a sparse matrix source such as SuiteSparse or Matrix Market.
2. The matrix should be compatible with the current sparse-matrix-to-graph pipeline.
3. The resulting graph should be small or medium-sized so that ColPack, NetworkX, and GNN experiments remain manageable.
4. The graph should add structural diversity compared to the current dataset.
5. ColPack should be able to generate distance-1 coloring results for the graph.
6. The graph should help evaluate generalization of the learned ordering method on unseen graph structures.

## Preferred Size Range

For the first dataset expansion, the preferred graph size range is:

- minimum number of vertices: around 20
- maximum number of vertices: around 1000

This range keeps the experiments practical while still allowing more meaningful evaluation than the initial five-graph dataset.

## Desired Graph Diversity

The expanded dataset should include:

- small graphs for debugging,
- medium-sized graphs for more meaningful evaluation,
- low-density sparse graphs,
- denser sparse graphs,
- graphs where heuristic orderings produce different color counts,
- graphs where heuristic orderings produce similar color counts,
- mostly square sparse matrices,
- optionally a few rectangular Jacobian-style matrices.

## Important Selection Principle

The most useful graphs are not necessarily the largest graphs. For this thesis, graphs are especially useful when the choice of vertex ordering affects the number of colors. Such graphs make the learned-ordering problem more meaningful.