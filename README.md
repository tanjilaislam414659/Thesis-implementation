# Graph Neural Networks for Learning-Based Graph Coloring

Implementation and experimental framework for the master's thesis:

**Graph Neural Networks for Learning-Based Graph Coloring:  
An Empirical Study in Sparse Derivative Computation**

The project investigates whether a graph neural network can learn **vertex orderings** that improve sequential greedy distance-1 graph coloring.

The motivating application is sparse derivative computation, particularly sparse Jacobian compression. The learned model does not assign colors directly. Instead, it predicts one scalar priority score for each vertex, the vertices are sorted by these scores, and a deterministic greedy coloring algorithm constructs the final coloring.

---

## Overview

Greedy graph coloring is fast and always produces a valid coloring, but the number of colors can depend strongly on the order in which vertices are processed.

Classical ordering strategies use predefined structural rules. This project studies whether a graph convolutional network (GCN) can instead learn useful vertex priorities from graph data.

The main pipeline is:

```text
Graph
  ↓
Node features
  ↓
GCN node scorer
  ↓
Vertex priority scores
  ↓
Descending score ordering
  ↓
Sequential greedy coloring
  ↓
Validity check and color-count evaluation
```

The learned orderings are compared with five ordering strategies provided by ColPack:

- Natural
- Largest First
- Dynamic Largest First
- Incidence Degree
- Smallest Last

The final thesis focuses exclusively on **distance-1 coloring**.

---

## Model

The learned node scorer is a two-layer graph convolutional network implemented with PyTorch Geometric.

Each vertex is represented by a 25-dimensional feature vector consisting of:

- 17 structural features,
- 7 symmetry-breaking features, and
- 1 constant feature.

The model architecture is:

```text
25 input features
      ↓
GCNConv(25, 32)
      ↓
ReLU
      ↓
GCNConv(32, 32)
      ↓
ReLU
      ↓
Linear(32, 1)
      ↓
one scalar score per vertex
```

No graph-level pooling or output activation is used.

At inference time, vertices are sorted in descending order of their predicted scores. Sequential greedy coloring is then applied to that ordering, followed by an explicit validity check.

---

## Learning Objectives

Two principal supervision objectives are implemented.

### Mean-Squared-Error Regression

The GCN learns normalized vertex-position targets derived from heuristic or verified exact orderings.

### Equivalence-Aware Pairwise Ranking

For controlled graphs with verified color classes, the model learns only the required ordering relations between different target color classes.

Vertices belonging to the same target color class are treated as equivalent and are not assigned a pairwise ordering preference.

The final ranking experiment uses 256 cross-class vertex pairs per training graph and epoch.

---

## Experimental Data

The implementation contains experiments on three main types of graph data.

### Sparse-Matrix Graphs

Sparse matrices are converted into simple undirected graph representations.

The final benchmark contains 15 sparse-matrix graphs. Most square matrices are used as generic sparsity-pattern graphs. The rectangular `jac_pat` example is converted explicitly into a column-intersection graph.

### Controlled Graphs

Controlled experiments use cycle-square graphs and complete joins with known target color counts.

These datasets make it possible to compare learned orderings against verified exact targets and against ColPack orderings that are known to require more colors on the selected instances.

### Heterogeneous Synthetic Graphs

The heterogeneous dataset contains retained instances from:

- Gilbert `G(n,p)` random graphs,
- crown graphs,
- Barabási-Albert graphs,
- Watts-Strogatz graphs, and
- stochastic block models.

Graphs were filtered for ordering sensitivity before training and evaluation.

---

## Final Controlled Comparison

The final frozen comparison uses:

- 125 training graphs,
- 5 validation graphs,
- 5 held-out test graphs,
- 25 node features,
- a two-layer GCN with 32 hidden units,
- Adam with learning rate `0.01`,
- 500 training epochs, and
- five random seeds: `0, 1, 2, 3, 4`.

The five controlled test graphs have:

- verified exact total: **60 colors**
- best-of-five ColPack total: **75 colors**

Across five seeds:

| Objective | Mean total | Population std. dev. | Exact outcomes |
|---|---:|---:|---:|
| MSE regression | 65.8 | 2.482 | 7/25 |
| Equivalence-aware ranking | 64.6 | 0.490 | 14/25 |

All reported learned colorings passed the validity checks.

---

## Repository Structure

```text
.
├── src/
│   ├── graphs/          # Graph loading, conversion, and ordering utilities
│   ├── models/          # GNN node-scoring model
│   ├── training/        # Shared training and coloring utilities
│   ├── experiments/     # Experiment-specific pipelines
│   └── utils/           # Supporting utilities
│
├── data/
│   ├── raw/             # Original or generated graph inputs
│   └── processed/       # Targets, splits, metadata, and generated datasets
│
├── results/
│   └── tables/          # Tracked experimental result tables
│
├── docs/                # Development notes and thesis workflow records
├── notebooks/           # Historical exploratory notebook material
├── configs/             # Reserved configuration directory
│
├── generate_sparse_jacobian_compression.py
├── generate_symmetry_breaking_node_features.py
├── plot_vertex_ordering_effect.py
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

The `week13_` to `week19_` prefixes in experiment filenames reflect the chronological development of the thesis implementation.

---

## Environment

The final experiments were conducted using:

- Windows 11
- Python 3.11.15
- PyTorch 2.10.0
- PyTorch Geometric 2.7.0
- NetworkX 3.6.1
- NumPy 2.4.3
- SciPy 1.17.1
- pandas 3.0.1
- scikit-learn 1.8.0
- Matplotlib 3.10.8

A CPU-only PyTorch environment was used for the reported experiments.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tanjilaislam414659/Thesis-implementation.git
cd Thesis-implementation
```

Create and activate a Python environment, for example with Conda:

```bash
conda create --name thesis python=3.11.15
conda activate thesis
```

Install the recorded Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch Geometric was installed separately in the final environment:

```bash
python -m pip install torch-geometric==2.7.0
```

### Note on `requirements.txt`

The current dependency snapshot contains one machine-specific `packaging` entry generated by the original Conda environment. That entry should be removed or replaced before installing the requirements on another machine.

---

## ColPack

ColPack is used for the five classical ordering and coloring baselines.

The experiments used a locally compiled ColPack installation together with a custom executable named:

```text
test_colpack.exe
```

Its interface is:

```text
test_colpack.exe <matrix-file> <ordering-name>
```

The five ordering identifiers used by the experiments are:

```text
NATURAL
LARGEST_FIRST
DYNAMIC_LARGEST_FIRST
INCIDENCE_DEGREE
SMALLEST_LAST
```

The custom runner source and exact linking command are not included in this repository. Reproducing the ColPack stages therefore requires a compatible executable linked against ColPack.

Some historical experiment scripts also contain machine-specific `PROJECT_ROOT` and `COLPACK_EXE` paths that must be adapted to the local system.

---

## Running the Experiments

The repository is organized as a research implementation rather than as a single-command software package. Individual experiments are reproduced through their corresponding scripts.

For example, the heterogeneous pipeline uses:

```bash
python -m src.experiments.week18_build_heterogeneous_pyg_dataset
python -m src.experiments.week18_train_heterogeneous_gnn
python -m src.experiments.week18_evaluate_heterogeneous_by_family
```

The final equivalence-aware comparison uses:

```bash
python -m src.experiments.week19_build_controlled_color_class_targets
```

Validate the final configuration without training:

```bash
python -m src.experiments.week19_train_equivalence_aware_objective_comparison --validate-only
```

Run both final objectives:

```bash
python -m src.experiments.week19_train_equivalence_aware_objective_comparison --objective all
```

An individual seed can also be selected:

```bash
python -m src.experiments.week19_train_equivalence_aware_objective_comparison --objective ranking --seed 0
```

---

## Results

Compact result tables used in the thesis are stored primarily under:

```text
results/tables/
```

Classical ColPack benchmark results are located mainly under:

```text
results/tables/initial_graph_coloring_benchmarks/
```

Learned-ordering results are located mainly under:

```text
results/tables/gnn_node_scorer/
```

Important final result files include:

```text
week19_equivalence_aware_final_comparison.csv
week19_equivalence_aware_per_graph_comparison.csv
```

The tracked CSV files allow the main numerical results to be inspected without retraining the models.

---

## Reproducing Thesis Figures

The repository root contains scripts for reproducing the figures created specifically for the thesis:

```bash
python generate_sparse_jacobian_compression.py
python generate_symmetry_breaking_node_features.py
python plot_vertex_ordering_effect.py
```

Each script produces its corresponding figure in PDF format.

The vertex-ordering illustration is adapted from an external source and is identified accordingly in the thesis.

---

## Reproducibility Notes

This repository contains the implementation, experiment scripts, split and target metadata, and compact result tables required to inspect the thesis experiments.

Some generated or machine-specific artifacts are intentionally not tracked, including:

- most raw source matrices,
- trained model checkpoints,
- generated PyTorch Geometric objects, and
- many raw ColPack output files.

A complete rerun may therefore require regenerating these artifacts and adapting local paths.

The repository should be regarded as a research implementation rather than a fully portable software package.

---

## Thesis Scope

The final thesis studies:

**learned vertex ordering for sequential greedy distance-1 graph coloring.**

Distance-2 coloring and other coloring formulations are not part of the final experimental evaluation.

Sparse derivative computation provides the main application motivation, with particular emphasis on column-intersection coloring for sparse Jacobian compression.

---

## License

The original implementation in this repository is distributed under the MIT License.

Third-party software and source datasets remain subject to their respective licenses and terms.

See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use this repository in academic work, please cite it using the metadata provided in:

[`CITATION.cff`](CITATION.cff)

Repository:

https://github.com/tanjilaislam414659/Thesis-implementation
