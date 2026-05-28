# Note on `jac_pat` Graph Representation Mismatch

During the preparation for Week 10, I compared the graph representation used in the Python/PyTorch Geometric pipeline with the graph representation reflected in the previously stored ColPack output for `jac_pat.mtx`.

The matrix `jac_pat.mtx` is rectangular:

```text
10 rows × 43 columns

In the Python graph pipeline, rectangular matrices are converted into column intersection graphs:

vertices represent matrix columns,
two columns are connected if they have nonzero entries in the same matrix row.

Using this representation, jac_pat produces:

43 vertices
121 undirected edges

However, the stored ColPack output for jac_pat reports:

43 vertices
72 edges


A direct check showed that 72 corresponds exactly to the number of unique undirected edges obtained by interpreting each nonzero matrix entry (row, column) as an undirected edge between those two integer indices.

Therefore, the current ColPack output for jac_pat was generated from a graph interpretation that is not the same as the Python/PyTorch Geometric column intersection graph representation.

Consequence

The previously stored ColPack color assignments for jac_pat should not be used directly as supervision or baseline labels for the current PyTorch Geometric jac_pat graph object.

For future evaluation on rectangular Jacobian-style matrices, the ColPack baseline workflow must be aligned with the same column-intersection representation used in the Python pipeline, or jac_pat should temporarily be excluded from label-based learning/evaluation until this alignment is resolved.

Immediate Impact on Week 10

This issue does not block the immediate learning-target design work, because the current training graphs:

ash85
can_24
hess_pat

are square matrices and their graph representations are aligned between the existing Python pipeline and ColPack baseline workflow.

The jac_pat mismatch must be addressed before final test-set evaluation or before using its ColPack labels in a learned-vs-heuristic comparison.



## Resolution in Week 11

This mismatch was resolved at the beginning of Week 11.

To align the ColPack workflow with the Python/PyTorch Geometric pipeline, the Python column-intersection graph for `jac_pat.mtx` was exported as a square Matrix Market graph file:

```text
data/processed/initial_graph_coloring_dataset/colpack_graph_inputs/jac_pat_column_intersection_graph.mtx


This exported graph contains:

→ 43 vertices
→ 21 undirected edges


ColPack was then rerun on this aligned graph representation. The corrected stored ColPack outputs now report:

→ 43 vertices
→ 121 edges
→ 11 colors for LARGEST_FIRST
→ 11 colors for SMALLEST_LAST


After this correction:

→ the jac_pat SMALLEST_LAST ordering targets were regenerated,
→ the target tensor was attached to jac_pat.pt,
→ the PyTorch Geometric target validation was updated,
→ the ColPack benchmark CSV was regenerated with the corrected jac_pat values.

Therefore, jac_pat is now aligned across:

→ Python graph construction
→ PyTorch Geometric graph data
→ ColPack baseline outputs
→ ordering target extraction
→ benchmark summary table