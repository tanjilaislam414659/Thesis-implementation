# Learning Vertex Ordering for Greedy Graph Coloring using Graph Neural Networks

## Overview

This repository contains the implementation and experimental framework for my master's thesis on **learning vertex orderings for greedy graph coloring using Graph Neural Networks (GNNs)**.

The objective of this research is to investigate whether **learning-based approaches** can improve graph coloring results compared to classical heuristic ordering strategies and algorithms implemented in **ColPack**.

The study focuses on the **graph coloring problem itself**, while **Jacobian and Hessian accumulation in sparse derivative computation** serve as the motivating application domain.

---

# Research Problem

Greedy graph coloring is widely used in sparse numerical algorithms, including **Jacobian and Hessian compression** in automatic differentiation.

The number of colors produced by greedy coloring strongly depends on the **vertex ordering**. Classical algorithms rely on hand-designed heuristics such as:

- largest-first ordering
- smallest-last ordering
- degree-based strategies

This thesis explores whether **Graph Neural Networks can learn vertex orderings directly from graph structure**, potentially producing better colorings than traditional heuristics.

---

# Research Approach

The thesis investigates graph coloring through a systematic empirical comparison between:

- classical heuristic ordering strategies
- algorithms implemented in **ColPack**
- a **Graph Neural Network–based ordering model**

The experimental framework follows two main stages.

### Stage 1 — Distance-1 Graph Coloring

The first stage establishes a complete experimental pipeline:

- graph dataset generation
- heuristic vertex ordering strategies
- greedy coloring evaluation
- ColPack baseline comparison
- learning-based vertex ordering using Graph Neural Networks

This stage validates the **dataset–training–evaluation pipeline**.

### Stage 2 — Distance-2 Graph Coloring

After the pipeline is validated, the framework is extended to **distance-2 coloring**, which reflects structural constraints encountered in **Jacobian accumulation problems**.

Although derivative computation motivates the problem, the thesis remains focused on **graph coloring algorithms and vertex ordering strategies**.

---

# Experimental Framework

The experimental pipeline follows this structure:

Graph → Vertex Ordering → Greedy Coloring → Evaluation

The evaluation measures:

- number of colors
- coloring validity
- runtime performance

The framework supports comparisons between:

- heuristic ordering methods
- ColPack algorithms
- GNN-based learned ordering

---

# Repository Structure
src/
│
├── graphs/ Graph generation and ordering strategies
├── models/ Graph neural network models
├── training/ Experiment pipelines and training scripts
└── utils/ Graph utilities and helper functions

data/
│
├── raw/ Original graph datasets
└── processed/ Processed graph datasets

notebooks/ Experimental notebooks

results/
│
├── tables/ Benchmark results
└── figures/ Plots and visualizations

docs/
└── thesis_log/
└── roadmap.md Thesis roadmap and planning document

configs/ Experiment configurations


---

# Implemented Components

### Graph Utilities

The repository includes utilities for:

- greedy graph coloring
- coloring validation
- graph statistics
- graph preprocessing

### Ordering Strategies

Several classical vertex ordering strategies are implemented:

- natural ordering
- reverse ordering
- random ordering
- descending degree ordering
- ascending degree ordering
- smallest-last ordering

These strategies serve as baseline methods for comparison with learned ordering approaches.

---

# Technologies

The implementation uses the following tools and libraries:

- Python 3
- NetworkX
- PyTorch
- NumPy
- SciPy
- pandas
- matplotlib

Planned integrations include:

- ColPack
- PyTorch Geometric

---

# Project Roadmap

The full thesis development roadmap can be found in:
docs/thesis_log/roadmap.md


This document describes the planned stages of the project, including:

- experimental framework development
- baseline comparisons
- graph neural network modeling
- distance-2 coloring extension
- large-scale experimental evaluation

---

# Purpose

This repository serves as the **research implementation environment** for the thesis and supports reproducible experimentation on graph coloring and vertex ordering methods.