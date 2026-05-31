# Star Coloring as a Possible Thesis Extension

## Background

Prof. Naumann mentioned that Paul Hovland from Argonne National Laboratory pointed out current interest in **star coloring for Hessian matrices**.

I will treat this as a possible extension of the current thesis work. The main thesis pipeline currently focuses on graph-level coloring and a GNN-based learned ordering approach. The primary evaluation is still based on the number of colors compared with heuristic baselines.

The current learning pipeline is:

```text
graph
→ node features
→ GNN
→ node scores
→ learned ordering
→ greedy coloring
→ comparison with heuristic baselines
```

Star coloring may fit naturally into this direction because it is also a graph coloring problem, but with stronger coloring constraints.

---

## What Star Coloring Means

In ordinary graph coloring, adjacent vertices must receive different colors.

In star coloring, the coloring must also avoid any path of four vertices that uses only two colors. In simple terms, a color pattern like:

```text
A - B - A - B
```

along a path of four vertices is not allowed.

Therefore, star coloring is stricter than ordinary distance-1 coloring and may require more colors.

---

## Relevance to Hessian Matrices

Star coloring is relevant for sparse Hessian computation because Hessian recovery can require stronger coloring conditions than ordinary graph coloring.

Since the thesis is motivated by sparse derivative computation, especially Jacobian and possibly Hessian accumulation, star coloring could be a meaningful Hessian-related extension.

The connection is:

```text
sparse Hessian structure
→ graph representation
→ star coloring problem
→ number of colors as evaluation metric
```

This keeps the extension at the graph level, which is consistent with the original thesis scope.

---

## How It Could Fit into the Current Thesis

The current main work focuses on learning orderings for distance-1 graph coloring.

A possible extension would be:

```text
Hessian-related graph
→ star coloring heuristic baseline
→ GNN-based learned ordering
→ star coloring procedure
→ compare number of colors
```

The GNN component could remain similar:

```text
graph
→ node features
→ GNN
→ node scores
→ ordering
```

The main difference would be in the coloring/evaluation stage, where the ordinary greedy coloring step would be replaced or extended by a star-coloring procedure.

---

## Possible Investigation Steps

To decide whether this extension can be included naturally, I should first investigate:

```text
- whether ColPack supports star coloring in a usable way,
- what input graph representation is required for Hessian-related star coloring,
- whether the current GNN ordering pipeline can be reused,
- how to verify that a produced coloring satisfies the star-coloring constraint,
- whether enough time remains after the main distance-1 evaluation.
```

A small first technical step could be to check whether ColPack can produce star coloring outputs for a simple Hessian-style graph.

---

## Priority and Scope

This extension should not replace the main thesis direction.

The main priority remains:

```text
distance-1 graph coloring
→ GNN-based learned ordering
→ learned-vs-heuristic comparison
```

Star coloring should be considered only after the current learning-based comparison pipeline is stable.

Therefore, the current plan is:

```text
1. complete the current distance-1 learned-vs-heuristic evaluation,
2. expand the dataset and improve the comparison,
3. investigate star coloring as an optional Hessian-related extension,
4. include it only if it fits naturally within the remaining thesis time.
```

---

## Possible Thesis Framing

If included, star coloring could be framed as an optional extension rather than the main thesis contribution.

A possible framing is:

```text
The main thesis studies a GNN-based learned ordering approach for graph coloring
in sparse derivative computation. After validating the distance-1 coloring pipeline,
the same idea may be investigated for Hessian-related coloring variants such as
star coloring.
```

This would keep the GNN pipeline central while also connecting the thesis to current interest in Hessian matrix coloring.

---

## Current Decision

For now, I will continue with the current learning-based pipeline and first focus on the distance-1 learned-vs-heuristic comparisons.

At the same time, I will keep star coloring in mind as a possible later extension and check whether it can be included naturally after the current pipeline is further developed.