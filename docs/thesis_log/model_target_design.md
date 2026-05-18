# Model Target Design for the First GNN Ordering Prototype

## Purpose

This document records the first learning-target decision for the GNN-based graph coloring pipeline.

The thesis pipeline follows the proposal-aligned structure:

```text
graph
→ node features
→ GNN
→ node-level scores
→ vertex ordering
→ greedy graph coloring
```

The purpose of the learning target is therefore to train the GNN to produce node-level scalar scores that can later induce a useful vertex ordering.

---

## Selected Learning Target

For the first learning prototype, the GNN will learn to imitate the **ColPack `SMALLEST_LAST` vertex ordering**.

The target is derived from the actual ordering printed by the updated ColPack runner:

```text
Order Position 0 -> Vertex ...
Order Position 1 -> Vertex ...
...
```

For each node, the parser stores:

```text
graph_id
node_id
order_position
target_score
ordering_name
output_file
```

The target file is:

```text
data/processed/initial_graph_coloring_dataset/ordering_targets/smallest_last_ordering_targets.csv
```

---

## Why `SMALLEST_LAST` Was Chosen

Two ColPack ordering strategies are currently available in the initial benchmark dataset:

```text
LARGEST_FIRST
SMALLEST_LAST
```

The current distance-1 baseline results are:

| Graph | LARGEST_FIRST | SMALLEST_LAST |
|---|---:|---:|
| ash85 | 5 colors | 4 colors |
| can_24 | 5 colors | 4 colors |
| hess_pat | 6 colors | 6 colors |
| hess_pat_small | 3 colors | 3 colors |
| jac_pat | 5 colors | 5 colors |

`SMALLEST_LAST` is selected because it is:

- never worse than `LARGEST_FIRST` on the current benchmark set,
- better on `ash85` and `can_24`,
- a well-established graph coloring ordering heuristic,
- a reasonable first behavior for the GNN to imitate.

---

## Target Score Definition

The raw ColPack output provides an ordering position:

```text
Order Position 0 = first vertex in the heuristic ordering
Order Position 1 = second vertex
...
```

For GNN training, this is converted into a normalized scalar target score:

```text
target_score = (num_vertices - 1 - order_position) / (num_vertices - 1)
```

for graphs with more than one vertex.

This produces:

```text
1.0 = earliest vertex in the SMALLEST_LAST ordering
0.0 = latest vertex in the SMALLEST_LAST ordering
```

The target is normalized to `[0, 1]` so that target scales remain comparable across graphs with different numbers of vertices.

---

## Alignment with the GNN Output

The first GNN model will output:

```text
one scalar score per node
```

During later inference:

```text
higher predicted score
→ earlier vertex in the learned ordering
```

Thus, the supervision target is directly aligned with the planned learned-ordering pipeline:

```text
GNN predicted node score
→ sort scores in descending order
→ learned vertex ordering
→ greedy coloring
```

---

## Why Ordering Targets Are Used Instead of Color Labels

The stored ColPack outputs also contain final per-vertex color assignments. However, the first GNN prototype uses **ordering targets**, not color labels.

This choice is made because the thesis proposal describes the learned method as:

```text
GNN → scalar node scores → ordering → greedy coloring
```

Training on the actual heuristic ordering is more directly aligned with this operational pipeline than training on final color IDs.

Color labels may be useful later for analysis or alternative formulations, but they are not the primary target in the first learning prototype.

---

## Implemented Target Extraction

The ordering targets are extracted by:

```text
src/training/parse_colpack_orderings.py
```

The resulting target CSV is checked by:

```text
src/training/check_ordering_targets.py
```

The validation checks confirm that:

- each graph has one target row per node,
- node IDs are unique,
- ordering positions are consecutive,
- target scores remain in `[0, 1]`.

---

## Important Note on `jac_pat`

A mismatch was identified for `jac_pat.mtx`:

- the Python/PyTorch Geometric pipeline represents it as a **column intersection graph** with 121 undirected edges,
- the current stored ColPack output reports a different graph representation with 72 edges.

Therefore, the existing `jac_pat` ColPack ordering target should **not** be attached to or used as supervision for the current `jac_pat.pt` PyTorch Geometric graph object until the rectangular-matrix ColPack workflow is aligned with the Python column-intersection representation.

The issue is documented separately in:

```text
docs/thesis_log/jac_pat_colpack_graph_mismatch_note.md
```

This does not block the initial learning-target design because the currently aligned square-matrix graphs are:

```text
ash85
can_24
hess_pat
hess_pat_small
```

---

## Current Outcome

The first GNN learning target is now defined as:

```text
normalized node-level target scores derived from the
ColPack SMALLEST_LAST vertex ordering
```

This provides the supervision needed for the next implementation stage:

```text
PyTorch Geometric graph data
+ ordering target scores
→ first GNN node-scoring model
```