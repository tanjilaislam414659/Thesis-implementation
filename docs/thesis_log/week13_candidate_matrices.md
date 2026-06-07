# Week 13 Candidate Sparse Matrices

## Purpose

This document records candidate sparse matrices for expanding the graph-coloring dataset in Week 13.

The goal is not only to increase the number of graphs, but to build a more meaningful evaluation set for the learned-ordering pipeline. Candidate matrices should therefore be selected to provide structural diversity, manageable graph sizes, and useful comparisons between heuristic and learned orderings.

## Current Dataset Summary

| Graph ID | Vertices | Edges | Notes |
|---|---:|---:|---|
| ash85 | 85 | 219 | small sparse graph |
| can_24 | 24 | 68 | small graph |
| hess_pat | 43 | 94 | Hessian-style pattern |
| hess_pat_small | 10 | 13 | very small debug/validation graph |
| jac_pat | 43 | 121 | rectangular Jacobian-style graph converted by column intersection |

## Candidate Selection Targets

The first expansion should add approximately 10 new graphs:

| Group | Target Count | Approximate Vertex Range |
|---|---:|---:|
| Small graphs | 3 | 20--100 |
| Medium-small graphs | 4 | 100--500 |
| Medium graphs | 3 | 500--1000 |

## Candidate Matrix Table

| Candidate ID | Source | Matrix Name | Square/Rectangular | Expected Size | Reason for Selection | Status |
|---|---|---|---|---:|---|---|
| C01 | Matrix Market / HB | bcsstk01 | square | 48 | small structural engineering matrix; useful sanity-check graph close to current dataset size | accepted after graph-conversion check: 48 vertices, 176 edges |
| C02 | Matrix Market / HB | bcsstm01 | square | 48 | companion mass matrix to bcsstk01; useful for checking structurally related matrices |  | rejected for main dataset: graph conversion produced 48 vertices and 0 edges |
| C03 | Matrix Market / HB | bcsstk03 | square | 112 | small-medium structural engineering matrix; slightly larger than current graphs | accepted after graph-conversion check: 112 vertices, 264 edges |
| C04 | Matrix Market / HB | bcsstk04 | square | 132 | small-medium structural engineering matrix; adds another sparse structural pattern | accepted after graph-conversion check: 132 vertices, 1758 edges |
| C05 | Matrix Market / HB | bcsstk05 | square | 153 | small-medium structural engineering matrix; useful for expanding beyond tiny graphs | accepted after graph-conversion check: 153 vertices, 1135 edges |
| C06 | Matrix Market / HB | dwt_234 | square | 234 | medium-small symmetric structural matrix; useful for graph-size expansion | accepted after graph-conversion check: 234 vertices, 300 edges |
| C07 | Matrix Market / HB | dwt_361 | square | 361 | medium-small structural matrix; useful for generalization testing | accepted after graph-conversion check: 361 vertices, 1296 edges |
| C08 | Matrix Market / HB | dwt_419 | square | 419 | medium-small structural matrix; adds size and structure variation | accepted after graph-conversion check: 419 vertices, 1572 edges |
| C09 | Matrix Market / HB | west0479 | square | 479 | chemical process simulation matrix; adds application-domain diversity and an unsymmetric pattern | accepted after graph-conversion check: 479 vertices, 1889 edges |
| C10 | Matrix Market / HB | dwt_503 | square | 503 | medium graph; useful for testing scalability beyond 500 vertices | candidate |
| C11 | Matrix Market / HB | bcsstk06 | square | 420 | symmetric stiffness matrix; useful as a Hessian-like candidate for possible star-coloring extension | accepted after graph-conversion check: 420 vertices, 3720 edges |
 C12 | Matrix Market / HB | bcsstk07 | square | 420 | symmetric stiffness matrix; useful as another Hessian/star-coloring-ready sparse graph candidate |  rejected for main dataset: graph structure is identical to bcsstk06 under current conversion |
 | C13 | Matrix Market / HB | sherman1 | square | 1000 | oil reservoir simulation matrix; adds application-domain diversity and tests scalability near the upper Week 13 size limit | accepted after graph-conversion check: 1000 vertices, 1375 edges |


## First Candidate Verification Notes

The first three downloaded candidates were checked by loading the Matrix Market files with SciPy and converting them using the current graph-construction pipeline.

Results:

| Matrix | Matrix Shape | Nonzeros | Graph Vertices | Graph Edges | Decision |
|---|---:|---:|---:|---:|---|
| bcsstk01 | 48 x 48 | 400 | 48 | 176 | accepted |
| bcsstm01 | 48 x 48 | 48 | 48 | 0 | rejected for main dataset |
| bcsstk03 | 112 x 112 | 640 | 112 | 264 | accepted |

The mass matrix `bcsstm01` produced an edgeless graph after removing diagonal entries. Since this would lead to a trivial distance-1 coloring, it is not useful for evaluating ordering-sensitive graph coloring behavior in the main learned-ordering dataset.


## Second Candidate Verification Notes

The second candidate batch contained `bcsstk04`, `bcsstk05`, and `dwt_234`.

Results:

| Matrix | Matrix Shape | Nonzeros | Graph Vertices | Graph Edges | Decision |
|---|---:|---:|---:|---:|---|
| bcsstk04 | 132 x 132 | 3648 | 132 | 1758 | accepted |
| bcsstk05 | 153 x 153 | 2423 | 153 | 1135 | accepted |
| dwt_234 | 234 x 234 | 834 | 234 | 300 | accepted |

The matrices `bcsstk04` and `bcsstk05` were accepted because they loaded correctly and produced non-trivial graph-coloring instances. Compared with the earlier accepted graphs, these two graphs are denser and therefore useful for testing the learned-ordering pipeline on structurally different sparse graphs.

The matrix `dwt_234` was accepted because it provides a larger but relatively sparse graph. This is useful for testing whether the learned-ordering pipeline behaves consistently beyond the very small graph instances used in the initial prototype.


## Third Candidate Verification Notes

The third candidate batch contained `dwt_361`, `dwt_419`, and `west0479`.

Results:

| Matrix | Matrix Shape | Nonzeros | Graph Vertices | Graph Edges | Decision |
|---|---:|---:|---:|---:|---|
| dwt_361 | 361 x 361 | 2953 | 361 | 1296 | accepted |
| dwt_419 | 419 x 419 | 3563 | 419 | 1572 | accepted |
| west0479 | 479 x 479 | 1910 | 479 | 1889 | accepted |

The matrices `dwt_361` and `dwt_419` extend the dataset with medium-sized sparse graphs from the DWT family. These graphs are useful for testing whether the learned-ordering pipeline remains stable as graph size increases beyond the initial small examples.

The matrix `west0479` adds application-domain diversity because it comes from a chemical/process simulation setting. This helps reduce the risk that the expanded dataset is dominated only by structural engineering matrices.

## Fourth Candidate Verification Notes

The fourth candidate batch contained `bcsstk06` and `bcsstk07`.

Results:

| Matrix | Matrix Shape | Nonzeros | Graph Vertices | Graph Edges | Decision |
|---|---:|---:|---:|---:|---|
| bcsstk06 | 420 x 420 | 7860 | 420 | 3720 | accepted |
| bcsstk07 | 420 x 420 | 7860 | 420 | 3720 | rejected for main dataset |

Both matrices loaded correctly and produced non-trivial graph-coloring instances. However, a duplicate-structure check showed that `bcsstk06` and `bcsstk07` produce the same vertex set and the same edge set under the current sparsity-pattern graph conversion.

Since the current thesis experiments are based on graph structure, keeping both would not add meaningful graph diversity. Therefore, `bcsstk06` is kept as a Hessian/star-coloring-ready stiffness-matrix candidate, while `bcsstk07` is excluded from the main expanded dataset.


## Fifth Candidate Verification Notes

The fifth candidate batch contained `sherman1`, which was selected as a replacement for the duplicate `bcsstk07`.

Results:

| Matrix | Matrix Shape | Nonzeros | Graph Vertices | Graph Edges | Decision |
|---|---:|---:|---:|---:|---|
| sherman1 | 1000 x 1000 | 3750 | 1000 | 1375 | accepted |

The matrix `sherman1` was accepted because it loaded correctly, produced a non-trivial graph, and added application-domain diversity beyond the structural and DWT matrix families. It also tests scalability near the upper size limit chosen for the first dataset expansion.



## Derivative- and Optimization-Motivated Candidates

In addition to general sparse benchmark matrices, the expanded dataset should include a small number of derivative- or optimization-motivated sparse matrix patterns if suitable manageable examples are available.

This is useful because the thesis is motivated by sparse Jacobian and Hessian computation. However, the main experimental object remains the graph induced by the sparsity pattern. Therefore, a matrix does not need to be used as an actual derivative matrix during the experiments; it only needs to provide a meaningful sparse graph-coloring instance.

Potential candidates in this category should be accepted only if they satisfy the same checks as the other matrices:

- valid Matrix Market loading,
- successful graph construction,
- non-trivial graph structure,
- manageable size,
- later ColPack baseline generation.

The current dataset already contains small derivative-motivated examples through `jac_pat`, `hess_pat`, and `hess_pat_small`. These are useful because they directly reflect Jacobian- and Hessian-style sparsity patterns, even though they are small. The Week 13 expansion mainly adds structurally diverse benchmark matrices, while later work may add further derivative- or optimization-related matrices if suitable manageable examples are found.


## Hessian and Star-Coloring Readiness

The thesis mainly focuses on learned vertex orderings for graph coloring. However, star coloring has been discussed as a possible optional extension for Hessian-related sparse derivative computation.

For this reason, the expanded dataset should also record which matrices may be useful later for Hessian- or star-coloring-related experiments.

A matrix is considered potentially useful for the Hessian/star-coloring extension if it has one or more of the following properties:

- it is symmetric or structurally symmetric,
- it comes from a stiffness, finite-element, PDE, structural mechanics, or optimization-related source,
- it produces a non-trivial graph after sparsity-pattern conversion,
- it has a manageable number of vertices and edges,
- it can be processed by the same ColPack/graph pipeline.

This does not mean that star coloring is part of the current Week 13 work. The current goal remains distance-1 dataset expansion. The purpose of this note is only to avoid selecting matrices that are useless for the possible later Hessian-related extension.


## Accepted Week 13 Expansion Graphs

The Week 13 dataset expansion accepted 10 new sparse matrix graphs.

| Graph ID | Source | Matrix Shape | Nonzeros | Graph Vertices | Graph Edges | Main Reason for Inclusion |
|---|---|---:|---:|---:|---:|---|
| bcsstk01 | Matrix Market / HB | 48 x 48 | 400 | 48 | 176 | small structural stiffness graph; useful sanity-check case |
| bcsstk03 | Matrix Market / HB | 112 x 112 | 640 | 112 | 264 | small-medium structural graph; expands beyond current graph sizes |
| bcsstk04 | Matrix Market / HB | 132 x 132 | 3648 | 132 | 1758 | denser structural graph; useful for testing ordering behavior |
| bcsstk05 | Matrix Market / HB | 153 x 153 | 2423 | 153 | 1135 | denser structural graph; adds size and density variation |
| dwt_234 | Matrix Market / HB | 234 x 234 | 834 | 234 | 300 | larger but sparse graph; useful for generalization testing |
| dwt_361 | Matrix Market / HB | 361 x 361 | 2953 | 361 | 1296 | medium sparse graph; tests scalability beyond small examples |
| dwt_419 | Matrix Market / HB | 419 x 419 | 3563 | 419 | 1572 | medium sparse graph; adds further size variation |
| west0479 | Matrix Market / HB | 479 x 479 | 1910 | 479 | 1889 | chemical/process simulation matrix; adds application-domain diversity |
| bcsstk06 | Matrix Market / HB | 420 x 420 | 7860 | 420 | 3720 | symmetric stiffness graph; useful for Hessian/star-coloring readiness |
| bcsstk07 | Matrix Market / HB | 420 x 420 | 7860 | 420 | 3720 | symmetric stiffness graph; useful for Hessian/star-coloring readiness |

After this expansion, the dataset contains approximately 15 graph instances in total when combined with the original five graphs. This is still manageable for debugging, but it is substantially more meaningful than the initial five-graph prototype for evaluating learned orderings.

The accepted matrices also improve the dataset in three ways:

1. **Larger graph sizes**  
   The dataset now includes graphs up to 479 vertices.

2. **More structural variation**  
   The accepted graphs include sparse, denser, structural, and process-simulation patterns.

3. **Better future extension readiness**  
   Several accepted matrices are symmetric stiffness-type matrices, which may be useful later if the optional Hessian/star-coloring extension is investigated.


## Final Week 13 Dataset Status

After candidate verification and duplicate checking, the Week 13 expansion resulted in 10 accepted new graph instances.

The accepted new graphs are:

- bcsstk01
- bcsstk03
- bcsstk04
- bcsstk05
- bcsstk06
- dwt_234
- dwt_361
- dwt_419
- west0479
- sherman1

Together with the original five graphs, the expanded dataset contains 15 graph instances.

The final graph-size range is:

- smallest graph: 10 vertices (`hess_pat_small`)
- largest graph: 1000 vertices (`sherman1`)

The expansion improves the dataset in several ways:

1. It increases the number of graph instances from 5 to 15.
2. It increases the maximum graph size from 85 vertices to 1000 vertices.
3. It adds denser structural graphs, sparse medium-sized graphs, and simulation-related graphs.
4. It includes symmetric stiffness-type matrices that may be useful later for the optional Hessian/star-coloring extension.
5. It avoids including duplicate graph structures, as shown by the rejection of `bcsstk07`.

The generated summary table is stored at:

```text
data/processed/initial_graph_coloring_dataset/graph_metadata/week13_expanded_graph_summary.csv
```

## Metadata Update Decision

The original metadata file from the initial five-graph dataset is not overwritten during Week 13.

Instead, the expanded graph summary is stored separately as:

```text
data/processed/initial_graph_coloring_dataset/graph_metadata/week13_expanded_graph_summary.csv
```

## Reproducibility Script

The expanded graph summary table is generated by the script:

```text
src/training/build_week13_expanded_graph_summary.py
```

This script scans all .mtx files stored in:

```text
data/raw/matrices/
```


For each matrix file, the script loads the matrix, converts it into the graph representation used in the current pipeline, and records both matrix-level and graph-level information.

The recorded fields are:
- graph ID,
- matrix filename,
- number of matrix rows,
- number of matrix columns,
- number of matrix nonzeros,
- number of graph vertices,
- number of graph edges.

The generated output file is:

```text
data/processed/initial_graph_coloring_dataset/graph_metadata/week13_expanded_graph_summary.csv
```


This script makes the Week 13 dataset expansion reproducible. It also provides a clean overview of the expanded raw graph dataset before generating new ColPack baselines in Week 14.

The script can be run from the project root using:

```text
python -m src.training.build_week13_expanded_graph_summary
```


## File Naming and Storage Plan

All accepted matrix files will be stored under:

```text
data/raw/matrices/
```