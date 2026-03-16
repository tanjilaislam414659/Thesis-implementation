# Thesis Roadmap

## Thesis Title
Learning Vertex Ordering for Greedy Graph Coloring using Graph Neural Networks

## Objective

The goal of this thesis is to investigate whether **Graph Neural Networks (GNNs)** can learn vertex orderings that improve greedy graph coloring compared to classical heuristics and algorithms implemented in ColPack.

The work is divided into two main phases.

### Phase 1 — Distance-1 Coloring

The first phase focuses on **distance-1 graph coloring**.  
The objective is to establish and validate the full experimental pipeline, including:

- graph datasets
- heuristic ordering strategies
- greedy coloring evaluation
- comparison with ColPack algorithms
- learning-based ordering using Graph Neural Networks

This phase establishes the **dataset–training–evaluation pipeline**.

### Phase 2 — Distance-2 Coloring

In the second phase, the framework will be extended to **distance-2 coloring**, which reflects structural constraints encountered in **Jacobian accumulation problems** in sparse derivative computation.

While derivative computation provides the motivation, the thesis remains focused on **graph coloring algorithms**, with Jacobian and Hessian accumulation serving as the application context.

---

# Phase 1 — Distance-1 Coloring Pipeline  
*(Months 1–3)*

Goal: Build and validate the complete experimental pipeline.

---

# Month 1 — Experimental Playground

Goal: Establish the research environment and begin parallel exploration of graph benchmarks, ColPack, and Graph Neural Networks.

---

## Week 1 — Development Environment 

### Tasks

- Set up Python environment using Conda
- Configure development environment in VS Code
- Initialize Git and connect the project to GitHub
- Install required libraries:
  - NetworkX
  - NumPy
  - SciPy
  - pandas
  - matplotlib
  - PyTorch

### Project Structure

The repository is organized into modular components including source code, data, results, documentation, and configuration files.

### Verification

- Graph generation using NetworkX
- Graph visualization tests
- Environment verification script

### Deliverable

A stable and reproducible development environment.

---

## Week 2 — Baseline Graph Coloring Framework

### Implemented Components

#### Graph Utilities

- greedy graph coloring
- coloring validity verification
- color count computation
- graph statistics utilities

#### Ordering Strategies

- natural ordering
- reverse ordering
- random ordering
- descending degree ordering
- ascending degree ordering
- smallest-last ordering

#### Benchmark Pipeline

The benchmark script evaluates different ordering strategies by:

- executing greedy coloring
- verifying coloring validity
- measuring runtime
- exporting results to CSV files

### Initial Test Graphs

- cycle graphs
- path graphs
- star graphs
- complete graphs
- custom example graphs

### Deliverables

- functioning experiment framework
- benchmark results stored in CSV format

---

## Week 3 — Parallel Research Begins

Three research tracks will run simultaneously.

### Track A — Expanded Graph Benchmarks

Tasks:

- add additional graph families:
  - wheel graphs
  - ladder graphs
  - balanced trees
  - barbell graphs
  - grid graphs
- run benchmark experiments on all graph families

Deliverables:

- expanded graph benchmark dataset
- updated benchmark results

---

### Track B — ColPack Setup

Tasks:

- obtain ColPack source code
- inspect repository structure
- study build instructions
- attempt compilation
- explore example programs
- document installation steps

Deliverables:

- working or partially built ColPack installation
- setup documentation

---

### Track C — Initial GNN Exploration

Study topics:

- graph representation for machine learning
- node features
- adjacency representations
- message passing
- node embeddings

Experiments:

- run simple GNN models on small graphs
- inspect node embeddings and outputs

Deliverables:

- first GNN exploration experiments
- notes on graph learning concepts

---

## Week 4 — Real Graph Structures

### Track A — Sparse Matrix Graphs

Tasks:

- explore SuiteSparse Matrix Collection
- load sparse matrices using SciPy
- convert matrices into graphs
- run greedy coloring experiments

Deliverables:

- sparse graph loader
- first experiments on sparse graphs

---

### Track B — ColPack Progress

Tasks:

- resolve compilation issues
- run ColPack example programs
- study available coloring algorithms

Deliverables:

- first successful ColPack execution

---

### Track C — GNN Playground

Develop an initial experimental pipeline connecting graph neural networks with vertex ordering and greedy coloring.

Deliverables:

- conceptual GNN-based ordering experiment

---

# Month 2 — Baseline Integration

Goal: Integrate classical graph coloring methods and learning-based approaches into a unified experimental framework.

---

## Week 5

Study ColPack algorithms including:

- distance-1 coloring
- largest-first ordering
- incidence-degree ordering

Document algorithm behavior and expected outputs.

---

## Week 6

Finalize ColPack installation and run coloring algorithms on benchmark graphs.

Verify correctness of results.

---

## Week 7

Integrate ColPack into the experiment pipeline and compare results against heuristic ordering strategies.

---

## Week 8 — First GNN Ordering Prototype

Develop a prototype pipeline where:

1. Graphs are represented with node features  
2. A Graph Neural Network produces node scores  
3. Nodes are ordered according to predicted scores  
4. Greedy coloring is executed using the learned ordering  

Deliverable: first prototype of a learning-based ordering system.

---

# Month 3 — Learning Pipeline Development

Goal: Develop and train the GNN ordering model.

---

## Week 9 — Node Feature Engineering

Potential features include:

- node degree
- clustering coefficient
- neighborhood statistics
- structural graph properties

Deliverable: feature extraction pipeline.

---

## Week 10 — Training Dataset Generation

Generate datasets consisting of:

- graph structures
- node features
- target vertex orderings

Deliverable: training dataset.

---

## Week 11 — Training Pipeline

Implement the GNN training system including:

- batching graphs
- loss functions
- training loop
- evaluation metrics

Deliverable: working training pipeline.

---

## Week 12 — First Trained Model

Train the initial GNN ordering model and evaluate it against heuristic strategies.

Deliverable: first trained model results.

---

# Phase 2 — Distance-2 Coloring Extension  
*(Months 4–5)*

Goal: Extend the validated pipeline to **distance-2 coloring**, reflecting structural constraints in Jacobian accumulation problems.

---

# Month 4 — Distance-2 Coloring Framework

---

## Week 13

Study theoretical foundations of distance-2 coloring and its relation to Jacobian compression.

---

## Week 14

Implement distance-2 graph representation methods.

---

## Week 15

Extend the greedy coloring framework to support distance-2 constraints.

---

## Week 16

Run comparative experiments between:

- heuristic orderings
- ColPack algorithms
- GNN-based ordering

for distance-2 coloring.

Deliverable: first distance-2 benchmark results.

---

# Month 5 — Large-Scale Experiments

Goal: Produce thesis-quality experimental evaluation.

---

## Week 17

Run experiments on larger sparse graphs derived from SuiteSparse matrices.

---

## Week 18

Perform ablation studies analyzing the impact of:

- node features
- model architectures
- training configurations

---

## Week 19

Analyze performance patterns and relationships between graph structure and coloring effectiveness.

---

## Week 20 — Final Evaluation

Generate final benchmark tables and visualizations comparing:

- heuristic ordering methods
- ColPack algorithms
- GNN-based ordering

---

## Optional Extension — Jacobian Accumulation Analysis

If time permits, analyze how improvements in graph coloring translate into potential improvements in **Jacobian accumulation efficiency**.

This may include:

- interpreting the number of colors as the number of required seed vectors
- estimating compression potential
- comparing different methods in terms of evaluation cost

This extension connects the graph coloring results to the motivating application domain while keeping the thesis focused on graph algorithms.

---

# Month 6 — Thesis Writing

---

## Week 21

Write background and methodology chapters.

---

## Week 22

Write experimental setup and implementation details.

---

## Week 23

Write results and discussion chapters.

---

## Week 24

Final revision, proofreading, and thesis submission.

---

# Key Milestones

End of Month 1  
Experimental playground established.

End of Month 2  
ColPack integrated and GNN prototype implemented.

End of Month 3  
Trainable GNN ordering pipeline completed.

End of Month 4  
Distance-2 coloring framework implemented.

End of Month 5  
Full experimental evaluation completed.

End of Month 6  
Thesis written and submitted.