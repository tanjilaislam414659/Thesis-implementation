"""
First GNN node-scoring model for the learning-based graph coloring pipeline.

The model takes:
- x: node feature matrix
- edge_index: graph connectivity

and returns:
- one scalar score per node

These predicted scores will later be used to induce a vertex ordering.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GNNNodeScorer(nn.Module):
    """
    A simple two-layer GCN model that outputs one scalar score per node.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, out_channels)

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix with shape [num_nodes, num_features].
        edge_index : torch.Tensor
            Graph connectivity with shape [2, num_directed_edges].

        Returns
        -------
        torch.Tensor
            Node score tensor with shape [num_nodes, 1].
        """

        x = self.conv1(x, edge_index)
        x = self.activation(x)

        x = self.conv2(x, edge_index)
        x = self.activation(x)

        scores = self.output_layer(x)

        return scores