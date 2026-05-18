# Initial GNN Node Scorer Design

## Purpose

This document records the first Graph Neural Network model implemented for the learning-based graph coloring pipeline.

The overall thesis pipeline is:

```text
graph
→ node features
→ GNN
→ node-level scores
→ vertex ordering
→ greedy coloring
```

The purpose of the first GNN model is to take a graph with structural node features and produce **one scalar score per node**. These scores will later be sorted to define a learned vertex ordering.

---

## Model Input

The model operates on the PyTorch Geometric graph objects prepared in Week 9.

Each graph provides:

| Input | Meaning |
|---|---|
| `x` | Node feature matrix with shape `[num_nodes, 5]` |
| `edge_index` | Graph connectivity with shape `[2, num_directed_edges]` |

The five current node features are:

```text
degree
normalized_degree
clustering_coefficient
core_number
constant_bias
```

---

## Model Output

The model returns:

```text
one scalar score per node
```

with shape:

```text
[num_nodes, 1]
```

The intended later interpretation is:

```text
higher predicted score
→ earlier position in the learned ordering
```

This matches the target design based on normalized `SMALLEST_LAST` ordering scores.

---

## Implemented Architecture

The first model is implemented in:

```text
src/models/gnn_node_scorer.py
```

The architecture is:

```text
Input node features
→ GCNConv layer
→ ReLU
→ GCNConv layer
→ ReLU
→ Linear projection
→ scalar node score
```

In more detail:

| Component | Role |
|---|---|
| first `GCNConv` | maps input node features into hidden node representations |
| ReLU | nonlinear activation |
| second `GCNConv` | performs another message-passing step |
| ReLU | nonlinear activation |
| linear output layer | maps hidden representation to one scalar per node |

The current default hidden dimension is:

```text
32
```

and the output dimension is:

```text
1
```

---

## Why a Simple GCN Was Chosen First

The first model is intentionally simple.

The goal at this stage is not yet to optimize architecture quality, but to verify that the GNN pipeline works end to end:

```text
graph data
→ model input
→ message passing
→ node-level scalar output
```

A simple two-layer GCN is suitable for this first prototype because it:

- is easy to implement and debug,
- is standard for graph-structured learning,
- performs local message passing over node neighborhoods,
- matches the thesis goal of learning node-level graph-structural scores.

More advanced architectures can be considered later only if the first learned-ordering pipeline requires improvement.

---

## Forward-Pass Verification

Two verification scripts were created:

```text
src/models/test_gnn_node_scorer.py
src/models/check_gnn_node_scorer_all_graphs.py
```

The first script tests one representative graph:

```text
ash85.pt
```

The second script tests all currently aligned graphs with attached ordering targets:

```text
ash85
can_24
hess_pat
hess_pat_small
```

For every tested graph, the output shape was verified as:

```text
[num_nodes, 1]
```

and was confirmed to match the attached target tensor shape:

```text
data.y.shape
```

---

## Current Limitation

The model has only been tested through forward passes.

At this stage, it has not yet been trained, and no learned ordering or coloring evaluation has been performed.

Those steps belong to the next stage of the learning pipeline.

---

## Current Outcome

The initial GNN node-scoring model is now implemented and successfully produces one scalar score per node for all aligned graph instances.

This completes the core Week 10 model-output milestone:

```text
PyTorch Geometric graph data
→ GNN node scorer
→ scalar score per node
```