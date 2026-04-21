# ColPack Output Mapping Notes

## Purpose

This note explains which parts of the current ColPack runner output can be stored in the dataset and benchmark schemas.

## Current runner output

The current runner prints:
- coloring method / ordering information
- total number of colors
- vertex count
- edge count
- ordering time
- coloring time
- number of colored vertices
- per-vertex color assignments

It also prints a placeholder message noting that explicit validity checking is not yet implemented in the runner output.

## Mapping to dataset fields

The following runner outputs can be used directly for dataset storage:
- `source_name`
- `num_vertices`
- `num_edges`
- `coloring_distance`
- `heuristic_method`
- `coloring_assignment`
- `num_colors`

The coloring assignment is obtained from `GetVertexColors(...)`.

## Mapping to benchmark fields

The following runner outputs can be used directly for benchmark rows:
- `graph_id`
- `source_type`
- `source_name`
- `num_vertices`
- `num_edges`
- `coloring_distance`
- `method_family`
- `method_name`
- `ordering_name`
- `num_colors`
- `runtime`
- `valid`

## Current limitation

The current runner does not yet perform or print an explicit validity check.
For now, validity is treated as expected for ColPack output, but this should later be made explicit in the pipeline.