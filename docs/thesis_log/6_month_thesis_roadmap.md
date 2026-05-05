# Revised 6-Month Thesis Roadmap — Version 4

**Thesis title:**  
*Graph Neural Networks for Learning-Based Graph Coloring: An Empirical Study in Sparse Derivative Computation*

---

## Overall Thesis Focus

The thesis investigates whether a **Graph Neural Network-based learned strategy** can produce graph colorings that are competitive with classical heuristic methods, especially ColPack baselines.

The main pipeline is:

```text
sparse matrix
→ graph construction
→ heuristic coloring labels / baselines
→ node features
→ GNN
→ node scores
→ vertex ordering
→ greedy coloring
→ comparison against heuristics
```

The thesis remains primarily at the **graph-coloring level**. Sparse Jacobian and Hessian computation provide the motivating application context, but the core experimental work is the comparison of coloring strategies.

The proposed learned method follows the thesis proposal design:

```text
GNN → scalar node scores → vertex ordering → greedy coloring
```

---

## Core Thesis Question

Can a GNN-based learned ordering strategy produce valid graph colorings that are competitive with classical heuristic methods, especially ColPack, when evaluated on sparse graphs derived from sparse matrices?

The main evaluation criteria are:

- number of colors
- coloring validity
- runtime / overhead
- generalization to unseen graphs

---

## Planning Principles for Version 4

### 1. Data comes first

The thesis needs a clean dataset where each instance contains:

```text
graph structure
+ graph metadata
+ heuristic coloring result
+ color count
+ train/validation/test assignment
```

### 2. ColPack is central, not secondary

ColPack is used for:

- baseline generation
- heuristic label generation
- comparison
- validation reference

### 3. Distance-1 comes first

The first full learning pipeline should be built and validated for **distance-1 coloring**.

Distance-2 is important, but it should come after the distance-1 pipeline is working.

### 4. Distance-2 appears early in design, later in implementation

Distance-2 should be considered in schema and documentation early, but actual implementation should begin after the core distance-1 learned pipeline exists.

### 5. The learned method should follow the proposal-consistent design

The main learned approach should be:

```text
GNN → scalar node scores → vertex ordering → greedy coloring
```

---

# What Is Already Completed

## Month 1 — Foundations and Baseline Infrastructure

**Status:** Completed

Already completed:

- development environment and repository setup
- greedy graph coloring framework
- coloring validity verification
- color count computation
- graph statistics utilities
- ordering strategies
- benchmark experiment pipeline
- CSV result export
- initial experiments on simple graph families
- expanded synthetic graph families
- ColPack setup and first validation runs
- first PyTorch Geometric / GNN exploration
- sparse matrix to graph pipeline
- comparison between Python pipeline and ColPack on shared examples

---

## Month 2 — Dataset Backbone and ColPack Integration

**Status:** Completed

Already completed:

- dataset schema
- benchmark schema
- ColPack output mapping
- reproducible ColPack command workflow
- ColPack outputs for initial matrix set
- ColPack output parser
- benchmark CSV table
- graph metadata generation
- rectangular matrix support through column intersection graph
- graph-level train/validation/test split
- distance-2 preparation notes

So Version 4 does **not** repeat Month 1 and Month 2 as active work. It starts from the current position.

---

# Month 3 — Core Learning Pipeline for Distance-1

## Goal

Build the first complete learning-based distance-1 pipeline:

```text
graph
→ node features
→ PyTorch Geometric Data object
→ GNN
→ node scores
→ ordering
→ greedy coloring
→ validation and comparison
```

The expected Month 3 outcome is a first trainable and testable learned-ordering prototype for distance-1 graph coloring.

---

## Week 9 — Feature Pipeline

### Focus

Prepare the input side of the learning pipeline.

This week should bridge the Month 2 dataset work with the upcoming GNN model.

The Week 9 pipeline should be:

```text
graph metadata / matrix file
→ graph construction
→ node feature extraction
→ feature matrix X
→ edge_index
→ PyTorch Geometric Data object
```

### Tasks

- implement a node feature extraction module
- start with simple structural node features
- make feature generation consistent across all current dataset graphs
- convert each graph into a PyTorch Geometric Data object
- connect feature generation with the existing train/validation/test split
- add basic consistency checks
- document the feature design

### Recommended node features

Start simple:

- degree
- normalized degree
- clustering coefficient
- core number
- optional constant bias feature

Degree should be included first because the proposal explicitly mentions simple structural features such as degree.

### Suggested files

```text
src/training/node_features.py
src/training/build_pyg_dataset.py
docs/thesis_log/node_feature_specification.md
```

Alternative feature module location:

```text
src/features/node_features.py
```

### Consistency checks

For each graph, verify:

- feature matrix has shape `[num_nodes, num_features]`
- all graphs have the same feature dimension
- `edge_index` has shape `[2, num_edges]` or `[2, 2*num_edges]` for explicitly undirected PyG representation
- no NaN values appear in the feature matrix
- every graph ID matches the metadata and split files
- train/validation/test graphs are loaded separately

### Deliverables

- node feature extraction module
- feature specification note
- feature generation test on all current dataset graphs
- PyTorch Geometric Data object creation
- consistency check across train/validation/test graphs

### Week 9 outcome statement

By the end of Week 9, the current graph-coloring dataset should be usable as input for a GNN. Each graph should have node features and a PyTorch Geometric representation containing at least `x`, `edge_index`, and `graph_id`.

---

## Week 10 — Learning Target and First GNN Model

### Focus

Define exactly what the model learns.

The proposal describes the learned approach as:

```text
GNN → node scores → ordering → greedy coloring
```

So Week 10 should formalize the operational version of this idea.

### Main design decision

Use the proposal-consistent design:

```text
The GNN predicts one scalar score per node.
Nodes are sorted by this score.
The sorted order is passed to greedy coloring.
The final coloring is checked for validity.
```

### Tasks

- define model input and output
- decide how ColPack heuristic information becomes a learning target
- choose the first training target
- choose the first loss function
- implement a first simple GNN model
- confirm that the model outputs one scalar per node
- document the design choice

### Possible training target options

- learn heuristic ordering positions, if available
- learn ranking scores derived from heuristic color assignments
- learn a proxy score based on color class or ordering behavior
- start with supervised imitation of a selected heuristic

The safest first version is probably:

```text
imitate a heuristic ordering or heuristic-derived node rank
```

This is easier to train and debug than directly optimizing color count.

### Suggested files

```text
src/training/gnn_model.py
src/training/build_training_targets.py
docs/thesis_log/model_target_design.md
```

### Deliverables

- model target design note
- first GNN model implementation
- target construction script or function
- test showing model output shape is `[num_nodes, 1]`

### Week 10 outcome statement

By the end of Week 10, the learning task should be clearly defined, and a first GNN should be able to take a graph with node features and output one scalar score per node.

---

## Week 11 — End-to-End Training Pipeline

### Focus

Connect dataset, model, training, inference, ordering, and greedy coloring.

This is where the first full learned pipeline becomes executable.

### Tasks

- implement a PyTorch Geometric dataset loader
- support train/validation/test graph loading
- support batching if needed
- implement training loop
- implement validation loop
- run first training experiments
- convert predicted node scores into vertex orderings
- apply greedy coloring to predicted orderings
- verify coloring validity
- save first learned coloring outputs

### Suggested files

```text
src/training/dataset_loader.py
src/training/train_gnn.py
src/training/evaluate_learned_ordering.py
results/tables/learned_ordering_initial_results/
```

### Checks

For every inference result, verify:

- node scores exist for all vertices
- ordering contains each vertex exactly once
- greedy coloring produces a color for every vertex
- coloring is valid
- color count is recorded

### Deliverables

- trainable end-to-end distance-1 pipeline
- first training logs
- first validation outputs
- first learned orderings
- first learned greedy colorings

### Week 11 outcome statement

By the end of Week 11, the project should have a working prototype that trains a GNN, predicts node scores, converts them into an ordering, applies greedy coloring, and verifies validity.

---

## Week 12 — First Distance-1 Learned-vs-Heuristic Comparison

### Focus

Run the first prototype comparison.

This should be treated as **pipeline validation**, not final thesis-quality evidence yet.

Because the current dataset is still small, the goal is to confirm that the learned pipeline works end-to-end.

### Tasks

- evaluate the learned model on held-out graphs
- compare learned orderings against ColPack baselines
- compare learned orderings against Python heuristic orderings
- measure number of colors
- verify validity of all colorings
- measure runtime / overhead
- inspect success and failure cases
- identify whether more data or better targets are needed

### Metrics

- graph ID
- method
- ordering source
- number of colors
- validity
- runtime
- comparison to `SMALLEST_LAST`
- comparison to `LARGEST_FIRST`

### Suggested output table

```text
graph_id | method | colors | valid | runtime | notes
```

### Deliverables

- first learned-vs-heuristic comparison table
- first distance-1 evaluation note
- list of limitations and next improvements

### Week 12 outcome statement

By the end of Week 12, there should be a first complete distance-1 comparison showing how the learned ordering prototype performs against heuristic baselines on unseen graphs.

---

# Month 4 — Strengthen Distance-1 Results and Start Distance-2 Implementation

## Goal

Improve the distance-1 learned pipeline and begin real distance-2 implementation.

Month 4 should not immediately jump to final conclusions. It should first improve data, features, model stability, and evaluation quality.

---

## Week 13 — Improve Distance-1 Model and Expand Dataset

### Focus

Strengthen the distance-1 pipeline before making larger claims.

Version 4 adds an important adjustment here:

```text
Expand the sparse matrix dataset before relying heavily on learned performance results.
```

The initial dataset is good for pipeline development, but too small for strong generalization claims.

### Tasks

- add more sparse matrix instances
- generate ColPack baselines for the new graphs
- update graph metadata
- update benchmark tables
- update train/validation/test split
- rerun feature extraction
- test improved feature combinations
- clean up training issues from Week 12

### Possible feature improvements

Only add these if the simple pipeline works:

- local triangle count
- average neighbor degree
- component ID or component size
- graph-normalized structural features

Avoid making the feature set too complicated too early.

### Deliverables

- expanded distance-1 dataset
- updated ColPack benchmark table
- updated metadata and split files
- improved feature pipeline
- stronger distance-1 model version

### Week 13 outcome statement

By the end of Week 13, the distance-1 learning pipeline should run on a larger and cleaner dataset, making the upcoming analysis more meaningful.

---

## Week 14 — Distance-1 Generalization and Analysis

### Focus

Analyze where the learned ordering works, fails, or behaves similarly to heuristics.

The proposal emphasizes performance across structurally diverse sparse graphs and generalization to unseen instances.

### Tasks

- evaluate on unseen test graphs
- analyze results by graph size and structure
- compare learned method to ColPack orderings
- compare learned method to Python greedy orderings
- inspect cases where learned ordering improves
- inspect cases where learned ordering fails
- analyze consistency across repeated runs

### Deliverables

- distance-1 generalization analysis
- distance-1 result summary
- plots or tables for color-count comparison
- notes for thesis results chapter

### Week 14 outcome statement

By the end of Week 14, the thesis should have a clearer picture of whether the learned distance-1 ordering is competitive, unstable, or mainly useful as a prototype.

---

## Week 15 — Implement Distance-2 Baseline Workflow

### Focus

Begin actual distance-2 implementation.

Earlier work prepared distance-2 conceptually, but Week 15 is where it becomes part of the implementation.

### Tasks

- extend coloring validity checker for distance-2
- extend benchmark schema usage for `coloring_distance = 2`
- modify ColPack runner to accept coloring distance as an argument
- run ColPack distance-2 baselines on selected graphs
- parse distance-2 ColPack outputs
- store distance-2 benchmark results
- test distance-2 on small graphs first

### Important checks

Distance-2 coloring must enforce:

```text
vertices at graph distance 1 or 2 cannot share the same color
```

### Deliverables

- distance-2 validity checker
- distance-2 ColPack runner workflow
- first distance-2 benchmark table
- updated documentation

### Week 15 outcome statement

By the end of Week 15, the project should have a working distance-2 baseline pipeline using ColPack and validity checks.

---

## Week 16 — First Learned Distance-2 Prototype

### Focus

Test whether the learned ordering idea can be reused for distance-2 coloring.

This does not need to be final. It is a first prototype.

### Tasks

- reuse the distance-1 learned ordering pipeline
- apply greedy distance-2 coloring using learned orderings
- compare with distance-2 ColPack baselines
- check validity carefully
- record number of colors and runtime
- document limitations

### Deliverables

- first learned distance-2 prototype
- first baseline-vs-learned distance-2 results
- notes on whether separate distance-2 training is needed

### Week 16 outcome statement

By the end of Week 16, the thesis should have a first indication of whether the learned ordering pipeline can transfer from distance-1 to distance-2 evaluation.

---

# Month 5 — Full Experimental Study

## Goal

Produce thesis-quality experiments and comparisons.

By the end of Month 5, most technical work should be finished.

---

## Week 17 — Larger Distance-1 and Distance-2 Experiments

### Focus

Run the main experiment batches.

### Tasks

- finalize selected graph dataset
- run ColPack baselines
- run Python heuristic baselines
- run learned distance-1 evaluation
- run learned distance-2 evaluation, if stable enough
- collect all results into consistent tables
- save logs and configurations

### Deliverables

- expanded distance-1 result tables
- expanded distance-2 result tables
- saved experiment logs
- reproducible experiment configuration

### Week 17 outcome statement

By the end of Week 17, the main experimental results should be collected in a structured form.

---

## Week 18 — Comparative Analysis

### Focus

Interpret the results.

### Tasks

- compare learned method with ColPack baselines
- compare learned method with Python heuristics
- analyze color-count differences
- analyze validity
- analyze runtime and overhead
- identify graph types where the learned method is competitive
- identify graph types where heuristics remain better

### Deliverables

- core comparison notes
- main findings draft
- result tables prepared for thesis writing

### Week 18 outcome statement

By the end of Week 18, the thesis should have its main experimental story.

---

## Week 19 — Ablation, Robustness, and Cleanup

### Focus

Check what matters in the learned system.

### Tasks

- run simple feature ablations
- compare degree-only vs extended features
- compare small model variations
- test robustness across random seeds
- verify reproducibility
- clean result folders and scripts

### Suggested ablations

- degree only
- degree + normalized degree
- degree + clustering + core number
- one GNN layer vs two GNN layers

Keep ablations small and thesis-useful.

### Deliverables

- ablation result table
- robustness notes
- cleaned experiment artifacts

### Week 19 outcome statement

By the end of Week 19, the thesis should have supporting evidence about which feature/model choices matter.

---

## Week 20 — Optional Application-Level Interpretation

### Focus

Connect the color-count results back to sparse derivative computation.

This should remain optional and analytical unless time allows a concrete implementation.

### Tasks

- interpret color counts as function-evaluation counts
- discuss implications for Jacobian/Hessian accumulation
- explain how distance-1 and distance-2 relate to sparse derivative computation
- optionally include a small illustrative example

### Deliverables

- application-oriented interpretation note
- final experiment package

### Week 20 outcome statement

By the end of Week 20, the thesis should have a clear explanation of why color-count improvements matter in the sparse derivative context.

---

# Month 6 — Thesis Writing and Finalization

## Goal

Write the thesis around the actual completed story:

```text
dataset construction
→ ColPack baselines
→ node feature pipeline
→ GNN learned ordering
→ greedy coloring
→ empirical comparison
→ interpretation in sparse derivative context
```

---

## Week 21 — Methodology Chapter

### Write

- problem definition
- graph coloring background
- sparse derivative motivation
- dataset construction
- sparse matrix to graph conversion
- ColPack baseline generation
- feature extraction
- GNN learned-ordering design
- evaluation metrics

### Deliverables

- methodology chapter draft
- pipeline diagram
- dataset schema explanation

---

## Week 22 — Experimental Setup and Results

### Write

- dataset description
- train/validation/test split
- baseline methods
- learned method setup
- distance-1 experiments
- distance-2 experiments, if completed
- result tables
- runtime comparison

### Deliverables

- experimental setup chapter draft
- results chapter draft
- finalized tables and plots

---

## Week 23 — Discussion and Conclusion

### Write

- what the learned approach achieved
- when it was competitive
- when heuristics performed better
- limitations of the dataset and model
- relation to sparse derivative computation
- distance-2 limitations or findings
- future work

### Deliverables

- discussion chapter draft
- conclusion chapter draft

---

## Week 24 — Final Editing and Submission

### Finish

- formatting
- bibliography
- figure and table captions
- consistency checks
- proofreading
- final supervisor review preparation
- final submission package

### Deliverables

- complete thesis manuscript
- final code repository cleanup
- final experiment archive
- final submission version

---

# Core Milestones in Roadmap Version 4

## End of Month 2

Already completed:

- clean graph-coloring dataset backbone
- ColPack-generated baseline labels
- benchmark CSV table
- graph metadata
- train/validation/test split
- distance-2 readiness notes

## End of Month 3

Expected outputs:

- node feature extraction pipeline
- PyTorch Geometric graph objects
- first GNN model
- first training loop
- first learned node-score ordering
- first distance-1 learned-vs-heuristic comparison

## End of Month 4

Expected outputs:

- improved distance-1 learned pipeline
- expanded dataset
- distance-1 generalization analysis
- working distance-2 baseline workflow
- first learned distance-2 prototype

## End of Month 5

Expected outputs:

- thesis-quality experiment results
- comparative analysis
- ablation or robustness checks
- optional sparse-derivative interpretation

## End of Month 6

Expected outputs:

- completed thesis manuscript
- cleaned code repository
- final tables and plots
- final submission package

---

# What Changed from Roadmap Version 3 to Version 4

## 1. Week 9 now includes PyTorch Geometric conversion

Version 3 only mentioned feature extraction and a feature note.

Version 4 expands Week 9 to:

```text
graph → features → edge_index → PyG Data object
```

This is important because the GNN pipeline needs PyG-compatible graph objects, not just standalone feature tables.

## 2. Week 10 now explicitly includes training target construction

Version 3 said “define learning target,” but Version 4 makes this more concrete:

```text
ColPack heuristic output → node-level training target
```

This is a crucial design step because the model must learn some node-level signal before it can produce orderings.

## 3. Week 11 now explicitly includes dataset loader and batching

Version 4 adds:

- PyG dataset loader
- train/validation/test graph loading
- batching support

This makes the end-to-end training pipeline more realistic.

## 4. Week 12 is framed as prototype validation

Version 4 avoids overclaiming from the first small dataset.

Week 12 should produce a first comparison, but not yet final evidence.

## 5. Week 13 adds dataset expansion

This is the biggest future adjustment.

The initial five-matrix dataset is excellent for pipeline development, but probably too small for strong learning-based generalization claims. So Version 4 adds dataset expansion before deeper analysis.

## 6. Distance-2 remains in Month 4

This is unchanged from v3 in spirit.

Distance-2 should not distract from Month 3. First build the distance-1 learning pipeline; then extend.

---

# Practical Month 3 Sequence

The most important practical sequence is:

```text
Week 9: graph → node features → PyG Data
Week 10: PyG Data → GNN → node scores
Week 11: node scores → ordering → greedy coloring
Week 12: learned coloring → compare with heuristics
```

This gives a clean and realistic Month 3 progression.
