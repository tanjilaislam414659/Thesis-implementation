# Distance-2 Preparation Notes

## Purpose

This note summarizes the planned role of distance-2 coloring in the thesis workflow.
The goal at this stage is not to implement full distance-2 experiments, but to understand how distance-2 will affect the dataset, ColPack usage, and evaluation pipeline later.


## Distance-1 vs Distance-2 coloring

In distance-1 coloring, adjacent vertices are not allowed to share the same color.
In distance-2 coloring, vertices with graph distance at most two are not allowed to share the same color.

Therefore, distance-2 coloring imposes stronger constraints than distance-1 coloring and will usually require at least as many colors.

## Relevance for sparse derivative computation

Distance-2 coloring is relevant because additional structural dependencies can appear in sparse derivative computation, especially in Jacobian and Hessian-related settings.
In this thesis, distance-2 coloring is treated as a planned extension after the distance-1 pipeline is established.

## Expected pipeline changes

The current pipeline is built around distance-1 coloring.
To support distance-2 coloring later, the pipeline will need to allow the coloring distance to be selected explicitly.

Expected changes include:
- running ColPack with `DISTANCE_TWO`
- storing `coloring_distance = 2` in benchmark and dataset records
- checking validity according to distance-2 constraints
- comparing distance-2 results separately from distance-1 results

## Storage format readiness

The current dataset and benchmark schemas already include a `coloring_distance` field.
This means the storage format can distinguish distance-1 and distance-2 results without needing a major redesign.

For distance-2 experiments, new benchmark rows can be added with `coloring_distance = 2`, while keeping the same general table structure.

## Current limitations

Distance-2 coloring has not yet been implemented in the Python benchmark pipeline.
The current ColPack runner is still hardcoded to use `DISTANCE_ONE`, so it must be updated later to accept the coloring distance as a command-line argument.

Distance-2 validity checking is also not yet implemented.
For now, distance-2 is only considered at the design and preparation level. 