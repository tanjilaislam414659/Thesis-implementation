# Thesis Roadmap

## Thesis Title
**Graph Neural Networks for Learning-Based Graph Coloring: An Empirical Study in Sparse Derivative Computation**

## Thesis Focus
Build and evaluate a pipeline where:

**graph data + heuristic colorings from ColPack → dataset → GNN-based learned ordering/coloring pipeline → comparison against heuristics**

The essential thesis level remains at the **graph-coloring level**, with **Jacobian/Hessian accumulation as the motivating application context**, and any direct downstream accumulation comparison treated as optional.

## Core Thesis Question
Can a **GNN-based learned strategy** produce graph colorings that are competitive with **classical heuristic methods**, especially ColPack baselines, when evaluated primarily by the **number of colors** on sparse graphs derived from sparse matrices?

---

## Planning Principles for v3

### 1. Data comes first
The first essential step is to create data, where each data item is a **graph plus a corresponding heuristic coloring**.

### 2. ColPack is not just a validator
ColPack is part of:
- baseline generation
- label generation
- dataset construction

not merely a side comparison tool.

### 3. The thesis stays mainly at graph level
The application motivation is sparse derivative computation, but the thesis should mainly compare graph colorings rather than becoming a full Jacobian/Hessian software project.

### 4. Distance-1 is the main first milestone
The work should first establish and validate the full pipeline on **distance-1 coloring**, then extend to **distance-2**.

### 5. Distance-2 should appear early in design, but later in experiments
So we do **light preparation earlier**, but the **main distance-2 implementation and experiments** belong later.

---

## What Is Already Completed Before This v3 Plan Starts

From the completed work so far, the following foundations are already in place:

- environment and repository structure
- baseline greedy coloring pipeline with ordering strategies and CSV export
- expanded synthetic graph benchmarks
- sparse matrix to graph pipeline on initial examples
- ColPack build and first validation runs
- first GNN exploration and graph → score → ordering → greedy-coloring concept bridge

So v3 should not repeat those as if unfinished. It should build on them.

---

# MONTH 1 — Foundations and Baseline Infrastructure
## Status: Completed

## Goal
Build the initial experimental playground:
- greedy coloring framework
- ordering strategies
- graph benchmarks
- first ColPack setup
- first GNN exploration

## Outcome Already Achieved
This month is essentially done based on the current progress reports.

---

# MONTH 2 — Dataset Backbone + ColPack Integration
## Goal
Turn the current prototype work into a **clean dataset-generation and baseline pipeline**.

This month should answer:
- What is one dataset instance?
- How do we generate heuristic labels with ColPack?
- How do we store graph data, heuristic results, and metadata so they can later be used for training and evaluation?

This is now the most important month.

## Week 5 — Formalize the Dataset Design
### Focus
Define the core thesis dataset:
- graph
- corresponding heuristic coloring
- number of colors
- graph metadata

### Tasks
- define a standard graph instance format
- define which metadata to store
- define which ColPack outputs are needed
- decide whether to store only color counts or also full color assignments
- define benchmark result schema

### Deliverables
- dataset schema document
- result schema document
- ColPack output mapping notes

## Week 6 — Clean ColPack Execution Pipeline
### Focus
Move from “ColPack works” to “ColPack is integrated and reproducible.”

### Tasks
- standardize how graphs / matrices are passed to ColPack
- standardize how outputs are parsed
- run official and custom examples in a repeatable way
- identify which heuristics/orderings will be used as baselines
- add support for storing heuristic outputs cleanly

### Deliverables
- stable ColPack runner workflow
- reproducible baseline-generation script
- short ColPack usage notes

## Week 7 — Build the First Real Dataset
### Focus
Construct the first substantial graph dataset using sparse matrices as the main source.

### Tasks
- collect sparse matrix instances
- convert them to graph form
- compute graph statistics
- generate heuristic colorings with ColPack
- store graphs, labels, and metrics in the standard format

### Deliverables
- first graph-coloring dataset
- first benchmark table with ColPack baselines
- stored graph metadata

## Week 8 — Prepare Evaluation Structure + Light Distance-2 Preparation
### Focus
Prepare for later learning and evaluation, while also introducing distance-2 at the design level.

### Tasks
- define graph-level train / validation / test split
- ensure unseen-graph evaluation is possible
- study distance-2 conceptually
- inspect how distance-2 affects ColPack usage and data representation
- confirm that the storage format will also work later for distance-2

### Deliverables
- graph-level split policy
- split files
- distance-2 preparation notes
- pipeline readiness check for distance-2

---

# MONTH 3 — Core Learning Pipeline for Distance-1
## Goal
Build the essential learned system:

**graph → node features → GNN → node scores / learned decision signal → ordering or coloring output → valid coloring**

This month should produce the first trainable and testable learning pipeline for the thesis’s essential level.

## Week 9 — Feature Pipeline
### Focus
Start simple and proposal-aligned.

### Tasks
- implement node feature extraction
- start with degree and a few simple structural features
- make feature generation consistent across all dataset graphs

### Deliverables
- feature extraction module
- feature specification note

## Week 10 — Define Learning Target and Model Output
### Focus
This is a crucial design week.

The main design should be:
- GNN predicts node scores
- sort nodes
- run greedy coloring

### Tasks
- formalize model input/output
- define training targets from heuristic data
- decide loss function
- implement first model

### Deliverables
- model design note
- first GNN model implementation

## Week 11 — End-to-End Training Pipeline
### Focus
Connect dataset, model, and evaluation.

### Tasks
- batching / loading graphs
- training loop
- validation loop
- end-to-end inference
- convert model outputs into orderings
- run greedy coloring
- verify validity of all results

### Deliverables
- trainable end-to-end pipeline
- first training runs
- first inference outputs on held-out graphs

## Week 12 — First Distance-1 Comparison
### Focus
Reach the first essential thesis milestone.

### Tasks
- evaluate trained model on unseen graphs
- compare learned outputs with heuristic baselines
- measure:
  - number of colors
  - validity
  - runtime / overhead
- inspect success and failure cases

### Deliverables
- first learned-vs-heuristic comparison table
- first distance-1 evaluation note

---

# MONTH 4 — Strengthen Distance-1 Results + Start Distance-2 Implementation
## Goal
Stabilize the main distance-1 study and begin the real extension to distance-2.

This is the right time for distance-2 to become active, because by now the distance-1 core should exist.

## Week 13 — Improve Distance-1 Model and Experiments
### Focus
Strengthen the distance-1 pipeline before broadening scope.

### Tasks
- clean up training issues
- improve feature set if needed
- test a small number of model/design variations
- improve experiment reproducibility

### Deliverables
- stronger distance-1 model version
- refined distance-1 benchmark outputs

## Week 14 — Generalization and Analysis for Distance-1
### Focus
Analyze behavior across structurally diverse sparse graphs and unseen instances.

### Tasks
- analyze results by graph type / structure
- inspect where learned ordering helps or fails
- compare consistency against heuristics

### Deliverables
- generalization analysis
- distance-1 results summary draft

## Week 15 — Implement Distance-2 Baseline Workflow
### Focus
Now begin full distance-2 work, not just conceptual notes.

### Tasks
- extend data / benchmark logic for distance-2 validity
- run ColPack distance-2 baselines
- define distance-2 dataset representation
- test on selected graph instances

### Deliverables
- working distance-2 baseline pipeline
- first distance-2 benchmark data

## Week 16 — First Learned Distance-2 Prototype
### Focus
Try the learned pipeline in the distance-2 setting.

### Tasks
- adapt learned pipeline to distance-2 evaluation
- test whether the same ordering idea can be reused
- run first small distance-2 experiments

### Deliverables
- first distance-2 learned prototype
- first baseline-vs-learned distance-2 results

---

# MONTH 5 — Full Experimental Study
## Goal
Produce thesis-quality experiments and comparisons.

By the end of this month, most of the technical work should be done.

## Week 17 — Larger Distance-1 and Distance-2 Runs
### Focus
Expand the experiment set.

### Tasks
- run larger benchmark batches
- finalize test set usage
- collect tables for both settings

### Deliverables
- expanded results tables
- experiment logs and plots

## Week 18 — Comparative Analysis
### Focus
Interpretation, not only execution.

### Tasks
- compare heuristics, ColPack baselines, and learned method
- analyze color-count differences
- analyze validity and runtime
- summarize where learned methods are competitive

### Deliverables
- comparison notes
- draft of core findings

## Week 19 — Ablation / Robustness / Cleanup
### Focus
Test what really matters in the learned system.

### Tasks
- simple ablations on features or model choices
- robustness checks across graph families
- confirm reproducibility
- finalize tables and plots

### Deliverables
- ablation results
- cleaned final experiment artifacts

## Week 20 — Optional Application-Level Extension
### Focus
This week is explicitly the optional layer.

If time permits, connect colorings more directly to the sparse derivative computation context.

### Tasks
- interpret color-count results in terms of Jacobian/Hessian accumulation
- optionally compare downstream implications more concretely
- if full implementation is too much, provide a clear analytical discussion instead

### Deliverables
- optional application-oriented note
- final experiment package

---

# MONTH 6 — Thesis Writing and Finalization
## Goal
Write the thesis around the actual story:

**dataset creation → heuristic baselines from ColPack → GNN learning pipeline → comparison on unseen graphs → interpretation in sparse derivative context**

## Week 21 — Methodology Chapter
### Write
- problem definition
- graph coloring background
- sparse derivative motivation
- dataset construction
- ColPack baseline generation
- learning pipeline design

## Week 22 — Experimental Setup and Results
### Write
- dataset and split design
- evaluation metrics
- distance-1 experiments
- distance-2 experiments
- main comparison tables

## Week 23 — Discussion and Conclusion
### Write
- what was learned
- when learned method helps
- limits of the approach
- relation to sparse derivative computation
- what remained optional / out of scope

## Week 24 — Final Editing and Submission
### Finish
- consistency checks
- figures/tables polishing
- bibliography
- formatting
- proofreading
- final submission package

---

# Core Milestones in v3

## End of Month 2
You should have:
- a clean graph-coloring dataset
- ColPack-generated baseline labels
- graph-level split design
- distance-2 readiness at design level

## End of Month 3
You should have:
- a first full learning pipeline for distance-1
- first learned-vs-heuristic comparisons on unseen graphs

## End of Month 4
You should have:
- stabilized distance-1 results
- first working distance-2 baseline and prototype experiments

## End of Month 5
You should have:
- thesis-quality comparative experiments
- optional application-level interpretation if time allows

## End of Month 6
You should have:
- completed thesis manuscript ready for submission

---

# What Changed from the Old Roadmap

This v3 plan makes a few important corrections.

## ColPack is moved into the center
Not just as software to “understand,” but as a:
- label generator
- baseline generator
- core part of the dataset workflow

## Data pipeline is brought earlier
Because dataset creation is the real first major task.

## Distance-2 is handled in two stages
- **Month 2:** conceptual/design preparation
- **Month 4 onward:** actual implementation and experiments

## The graph-level thesis story is protected
The plan keeps the thesis mainly at the graph-coloring level, with sparse derivative computation as the motivating application context.

## The plan respects real progress
It does not waste time pretending that environment setup or first ColPack execution still need to begin from scratch.
