# Sparse Derivative Coloring Background Notes

## Paper

Narayanan, S. H. K., Norris, B., Hovland, P., Nguyen, D. C., and Gebremedhin, A. H.  
"Sparse Jacobian Computation Using ADIC2 and ColPack."

## Relevance to the Thesis

This paper is relevant because it connects automatic differentiation, sparse derivative computation, ColPack, and graph coloring. The paper describes how ADIC2 and ColPack can be integrated so that sparse Jacobian computations can exploit graph coloring to reduce storage and computational cost.

The paper also gives a useful overview of coloring models used in ColPack for sparse derivative matrices:

- Jacobian direct recovery: distance-2 coloring
- Hessian direct recovery: star coloring
- Jacobian substitution recovery: acyclic bicoloring
- Hessian substitution recovery: acyclic coloring

This is important for the thesis because the current work focuses on learning-based graph coloring, while sparse Jacobian and Hessian computation provide the motivating application context.

## Connection to Current Thesis Work

The current thesis pipeline focuses first on distance-1 coloring in order to build and validate the full learning-based pipeline:

graph → node features → GNN → node scores → ordering → greedy coloring

Later, the framework may be extended toward distance-2 coloring for Jacobian-related sparse derivative computation. Star coloring can also be considered as an optional Hessian-related extension, following the coloring models discussed in the ADIC2-ColPack paper.

## Practical Planning Note

The current Month 4 work should still focus on expanding the distance-1 dataset and strengthening the learned-vs-heuristic evaluation. However, selected symmetric stiffness or Hessian-like sparse matrices should be kept in the dataset because they may be useful later for a possible star-coloring extension.