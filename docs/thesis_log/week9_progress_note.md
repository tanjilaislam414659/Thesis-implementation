# Week 9 Progress Note — Feature Pipeline

During Week 9, I implemented the first node feature pipeline for the learning-based graph coloring part of the thesis.

The goal was to prepare the current graph-coloring dataset for later use with a Graph Neural Network.

I created a node feature extraction module that computes simple structural features for each node: degree, normalized degree, clustering coefficient, core number, and a constant bias feature.

The feature pipeline was tested on all current dataset graphs: `ash85`, `can_24`, `hess_pat`, `hess_pat_small`, and `jac_pat`.

All graphs produced valid node feature matrices with consistent shape `[num_nodes, 5]`.

I then converted each graph into a PyTorch Geometric `Data` object containing `x`, `edge_index`, `graph_id`, `split`, and `num_nodes`.

The generated graph objects were saved as `.pt` files under `data/processed/initial_graph_coloring_dataset/pyg_data/`, and a readable summary CSV was also created.

Additional checks confirmed that the saved `.pt` files can be loaded again successfully.

The main outcome of Week 9 is that the input side of the learning pipeline is ready:

```text
raw matrix → graph → node features → PyTorch Geometric Data object → saved .pt file