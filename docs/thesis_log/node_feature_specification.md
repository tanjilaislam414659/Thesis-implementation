# Node Feature Specification for Initial GNN Pipeline

## Purpose

This document describes the initial node feature pipeline used for the learning-based graph coloring experiments.

The purpose of the feature pipeline is to convert each graph in the initial graph-coloring dataset into a node feature representation suitable for input to a Graph Neural Network.

The current pipeline follows the structure:

```text
Matrix Market file
→ graph construction
→ node feature extraction
→ PyTorch Geometric Data object
```

At this stage, the focus is not yet on model training. The goal is to prepare consistent graph inputs for the future GNN-based ordering pipeline.

---

## Dataset Graphs

The initial dataset contains the following graph instances:

| Graph ID | Source matrix | Split |
|---|---|---|
| ash85 | ash85.mtx | train |
| can_24 | can_24.mtx | train |
| hess_pat | hess_pat.mtx | train |
| hess_pat_small | hess_pat_small.mtx | validation |
| jac_pat | jac_pat.mtx | test |

Square matrices are converted into undirected sparsity-pattern graphs.

Rectangular matrices are converted into column intersection graphs. In this representation, vertices correspond to matrix columns, and two columns are connected if they have nonzero entries in the same matrix row.

---

## Node Features

The initial feature set is intentionally simple and structural. Each node receives the following five features:

| Feature | Description |
|---|---|
| degree | Number of neighbors of the node |
| normalized_degree | Degree divided by the maximum possible degree in the graph |
| clustering_coefficient | Local clustering coefficient of the node |
| core_number | Core number from the graph's core decomposition |
| constant_bias | Constant value `1.0` for every node |

The feature vector for each node is therefore:

```text
[degree, normalized_degree, clustering_coefficient, core_number, constant_bias]
```

Thus, each graph has a node feature matrix with shape:

```text
[num_nodes, 5]
```

---

## Motivation for Feature Choices

The first feature set is designed to be simple, interpretable, and consistent with the thesis proposal.

Degree is included because classical graph coloring heuristics often depend on vertex degree, and the thesis proposal mentions simple structural node features such as degree as an initial representation.

Normalized degree is included so that degree information is easier to compare across graphs of different sizes.

Clustering coefficient gives a simple measure of local neighborhood density.

Core number gives a simple measure of the structural importance of a node within the graph.

The constant bias feature gives every node a baseline feature value.

---

## PyTorch Geometric Representation

Each graph is converted into a PyTorch Geometric `Data` object containing:

| Attribute | Meaning |
|---|---|
| `x` | Node feature matrix |
| `edge_index` | Graph connectivity in COO format |
| `graph_id` | Identifier of the graph |
| `split` | Train/validation/test split |
| `num_nodes` | Number of nodes |

For undirected graphs, both edge directions are stored in `edge_index`.

Therefore, a graph with `m` undirected edges has `2m` directed edges in the PyTorch Geometric representation.

Example:

```text
NetworkX graph with 219 undirected edges
→ PyTorch Geometric edge_index with 438 directed edges
```

---

## Generated Files

The PyTorch Geometric graph objects are saved under:

```text
data/processed/initial_graph_coloring_dataset/pyg_data/
```

The generated `.pt` files are:

```text
ash85.pt
can_24.pt
hess_pat.pt
hess_pat_small.pt
jac_pat.pt
```

A human-readable summary is stored in:

```text
data/processed/initial_graph_coloring_dataset/pyg_data/pyg_dataset_summary.csv
```

The summary CSV contains:

```text
graph_id, split, num_nodes, num_directed_edges, num_features, pyg_file
```

---

## Why `.pt` Files Are Used

The `.pt` file format is used because the processed graph data is intended for PyTorch and PyTorch Geometric.

A PyTorch Geometric graph is not just a simple table. Each graph object contains several pieces of information, including:

```text
x          = node feature matrix
edge_index = graph connectivity
graph_id   = graph identifier
split      = train/validation/test assignment
num_nodes  = number of nodes
```

A `.csv` file is useful for human-readable tables such as metadata, benchmark results, color counts, and summaries.

However, a `.csv` file is not ideal for storing full graph objects because each graph can have a different number of nodes and edges. The `edge_index` structure is also a tensor representation of graph connectivity, which is easier to save and reload directly using PyTorch.

Therefore, the project uses both formats:

```text
.csv files = metadata, benchmark tables, split files, and readable summaries
.pt files  = PyTorch/PyTorch Geometric training-ready graph objects
```

The `.pt` files are not meant to be edited manually. They are saved and loaded directly in Python using PyTorch:

```python
import torch

data = torch.load(
    "data/processed/initial_graph_coloring_dataset/pyg_data/ash85.pt",
    weights_only=False,
)

print(data.x)
print(data.edge_index)
```

This makes the processed graphs easy to reuse in later training and evaluation scripts without rebuilding the feature pipeline every time.

---

## Implemented Scripts

The following scripts were created for the Week 9 feature pipeline:

```text
src/training/node_features.py
src/training/check_node_features.py
src/training/build_pyg_dataset.py
src/training/check_saved_pyg_data.py
src/training/build_pyg_dataset_summary.py
```

### `node_features.py`

Contains the main node feature extraction utilities:

```text
NetworkX graph
→ node feature matrix
```

### `check_node_features.py`

Checks feature extraction on all current dataset graphs.

### `build_pyg_dataset.py`

Builds PyTorch Geometric `Data` objects from the current dataset graphs and saves them as `.pt` files.

### `check_saved_pyg_data.py`

Loads the saved `.pt` files and verifies that they are still valid.

### `build_pyg_dataset_summary.py`

Creates a human-readable CSV summary of the saved PyTorch Geometric dataset.

---

## Consistency Checks

The following checks are performed:

- each matrix file can be loaded as a graph,
- each graph produces a valid feature matrix,
- the feature matrix has shape `[num_nodes, 5]`,
- all graphs have the same feature dimension,
- no NaN or infinite values occur,
- `edge_index` has shape `[2, num_directed_edges]`,
- all edge indices refer to valid node IDs,
- each graph has a valid graph ID,
- each graph has a train/validation/test split assignment,
- saved `.pt` files can be loaded again successfully.

---

## Current Dataset Summary

The current PyTorch Geometric dataset contains five graphs:

| Graph ID | Split | Nodes | Directed edges | Features |
|---|---|---:|---:|---:|
| ash85 | train | 85 | 438 | 5 |
| can_24 | train | 24 | 136 | 5 |
| hess_pat | train | 43 | 188 | 5 |
| hess_pat_small | validation | 10 | 26 | 5 |
| jac_pat | test | 43 | 242 | 5 |

The directed edge count is twice the undirected edge count because PyTorch Geometric stores both directions for undirected graphs.

---

## Current Limitations

The current feature set is intentionally simple.

It does not yet include more expensive or global graph features such as:

- PageRank,
- betweenness centrality,
- closeness centrality,
- spectral features,
- learned positional encodings.

These may be considered later if the first learning pipeline requires stronger structural information.

The current pipeline prepares GNN input data only. It does not yet define training targets, loss functions, or a GNN model. These will be addressed in the next stage of the thesis work.

---

## Week 9 Outcome

The Week 9 feature pipeline successfully converts all current dataset graphs into PyTorch Geometric `Data` objects with consistent node features and split information.

The resulting pipeline is:

```text
raw matrix
→ graph
→ node features
→ edge_index
→ PyTorch Geometric Data object
→ saved .pt file
```

This prepares the dataset for the next stage: defining the learning target and implementing the first GNN model.