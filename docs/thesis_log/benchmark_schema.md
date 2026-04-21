# Benchmark Schema

## Purpose

This document defines the structure of benchmark result tables used in the thesis experiments.
A benchmark result table stores the outcome of running one coloring method on one graph instance.

## What is one benchmark row?

One benchmark row represents one method run on one graph.
It stores the information needed to compare methods consistently across graphs and experiments.

## Core fields

- graph_id
- source_type
- source_name
- num_vertices
- num_edges
- coloring_distance
- method_family
- method_name
- ordering_name
- num_colors
- runtime
- valid

## Field notes

- `graph_id`: unique identifier of the graph instance
- `source_type`: origin category such as sparse_matrix or synthetic_graph
- `source_name`: original graph or matrix name
- `num_vertices`: number of vertices in the graph
- `num_edges`: number of edges in the graph
- `coloring_distance`: usually 1 now, later extensible to 2
- `method_family`: broad source of the result, for example python_heuristic, colpack, or gnn
- `method_name`: specific algorithm or pipeline name
- `ordering_name`: ordering strategy used before greedy coloring, if applicable
- `num_colors`: number of colors produced
- `runtime`: runtime of the method execution
- `valid`: whether the produced coloring is valid

## Notes

The benchmark schema is separate from the dataset schema.
The dataset schema describes stored graph instances and labels, while the benchmark schema describes experiment outputs.