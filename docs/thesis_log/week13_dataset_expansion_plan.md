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

## Basis for Selecting Candidate Matrices

Candidate matrices are selected from standard sparse matrix benchmark collections such as Matrix Market and SuiteSparse. These collections are suitable for this thesis because they provide real sparse matrix structures that can be converted into graph-coloring instances.

The selection is guided by four main principles:

1. **Pipeline compatibility**  
   The matrix should be compatible with the current sparse-matrix-to-graph pipeline, ColPack baseline generation, target extraction, and PyTorch Geometric dataset construction.

2. **Manageable graph size**  
   The first expansion should focus on small and medium-sized graphs. This keeps debugging practical while still making the evaluation more meaningful than the initial five-graph prototype.

3. **Structural diversity**  
   The expanded dataset should include graphs with different sizes, densities, and application origins. This helps evaluate whether the learned ordering generalizes beyond one narrow graph type.

4. **Ordering relevance**  
   Graphs are especially useful when different vertex orderings lead to different color counts. Such graphs make the learned-ordering problem more meaningful because the ordering has a visible effect on coloring quality.

Candidate matrices are therefore treated as provisional. A matrix is accepted into the expanded dataset only after verifying successful graph conversion, ColPack baseline generation, ordering-target extraction, and basic coloring behavior.


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


## Initial Expansion Target

The first dataset expansion will add approximately 10 new sparse matrix graphs.

The target expanded dataset size is therefore:

- current graphs: 5
- new graphs: approximately 10
- total graphs after expansion: approximately 15

The new graphs should be selected in a controlled way:

| Group | Target Count | Approximate Vertex Range | Purpose |
|---|---:|---:|---|
| Small graphs | 3 | 20--100 | sanity checks and comparison with current graphs |
| Medium-small graphs | 4 | 100--500 | main learning and evaluation expansion |
| Medium graphs | 3 | 500--1000 | scalability and generalization testing |

For the first expansion, square sparse matrices are preferred because the current graph-construction pipeline is more stable for square matrices. Rectangular Jacobian-style matrices remain useful, but they should be added carefully because their graph representation must match the ColPack baseline representation.

The first expansion will therefore prioritize:

- mostly square sparse matrices,
- manageable graph sizes,
- structural diversity,
- graphs where different vertex orderings may lead to different color counts.