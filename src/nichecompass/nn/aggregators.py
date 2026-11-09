"""
This module contains gene expression aggregators used by the NicheCompass model.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax


def get_gcn_normalized_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the edges and weights of the symmetrically normalized adjacency matrix
    with self loops, as `gcn_norm` computes them for a torch_sparse
    SparseTensor: existing self loops are replaced by one self loop of weight 1
    per node, and the entry (i, j) is scaled by deg(i)^-1/2 * deg(j)^-1/2,
    where deg is the row sum.

    Parameters
    ----------
    edge_index:
        Tensor containing the node indices of edges.
        (Size: 2 x n_edges)
    num_nodes:
        Number of nodes.
    dtype:
        Data type of the edge weights.

    Returns
    ----------
    row:
        Row index of each edge, the edges without self loops in input order
        followed by one self loop per node.
    col:
        Column index of each edge.
    value:
        Normalized weight of each edge.
    """
    row, col = edge_index
    not_self_loop = row != col
    nodes = torch.arange(num_nodes, device=edge_index.device)
    row = torch.cat([row[not_self_loop], nodes])
    col = torch.cat([col[not_self_loop], nodes])
    deg = torch.bincount(row, minlength=num_nodes).to(dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)
    value = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return row, col, value


def aggregate_to_columns(row: torch.Tensor,
                         col: torch.Tensor,
                         value: torch.Tensor,
                         x: torch.Tensor) -> torch.Tensor:
    """
    Multiply the transposed sparse adjacency matrix with `x`, that is sum
    value * x[row] into position col, as `adj.t().matmul(x)` does for a
    torch_sparse SparseTensor. The transposed matrix is built in CSR format
    with sorted indices, which avoids sorting the edges again inside the
    sparse matrix product.

    Parameters
    ----------
    row:
        Row index of each edge.
    col:
        Column index of each edge.
    value:
        Weight of each edge.
    x:
        Tensor containing the node features.
        (Size: n_nodes x n_node_features)

    Returns
    ----------
    x_aggregated:
        Tensor containing the aggregated node features.
        (Size: n_nodes x n_node_features)
    """
    num_nodes = x.shape[0]
    order = torch.argsort(col * num_nodes + row)
    crow_indices = torch.zeros(num_nodes + 1, dtype=torch.long, device=x.device)
    crow_indices[1:] = torch.cumsum(torch.bincount(col, minlength=num_nodes), 0)
    adj_t = torch.sparse_csr_tensor(crow_indices,
                                    row[order],
                                    value[order].to(x.dtype),
                                    (num_nodes, num_nodes))
    return adj_t @ x


class OneHopAttentionNodeLabelAggregator(MessagePassing):
    """
    One-hop Attention Node Label Aggregator class that uses a weighted sum
    of the omics features of a node's 1-hop neighbors to build an
    aggregated neighbor omics feature vector for a node. The weights are
    determined by an additivite attention mechanism with learnable weights.
    It returns a concatenation of the node's own omics feature vector and
    the attention-aggregated neighbor omics feature vector as node labels
    for the omics reconstruction task. 
    
    Parts of the implementation are inspired by
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py#L16
    (01.10.2022).

    Parameters
    ----------
    modality:
        Omics modality that is aggregated. Can be either `rna` or `atac`.
    n_input:
        Number of omics features used for the Node Label Aggregation.
    n_heads:
        Number of attention heads for multi-head attention.
    leaky_relu_negative_slope:
        Slope of the leaky relu activation function.
    dropout_rate:
        Dropout probability of the normalized attention coefficients which
        exposes each node to a stochastically sampled neighborhood during 
        training.
    """
    def __init__(self,
                 modality: Literal["rna", "atac"],
                 n_input: int,
                 n_heads: int=4,
                 leaky_relu_negative_slope: float=0.2,
                 dropout_rate: float=0.):
        super().__init__(node_dim=0)
        self.n_input = n_input
        self.n_heads = n_heads
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.linear_l_l = Linear(n_input,
                                 n_input * n_heads,
                                 bias=False,
                                 weight_initializer="glorot")
        self.linear_r_l = Linear(n_input,
                                 n_input * n_heads,
                                 bias=False,
                                 weight_initializer="glorot")
        self.attn = nn.Parameter(torch.Tensor(1, n_heads, n_input))
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.dropout = nn.Dropout(p=dropout_rate)
        self._alpha = None
        self.reset_parameters()

        print(f"ONE HOP ATTENTION {modality.upper()} NODE LABEL AGGREGATOR -> "
              f"n_input: {n_input}, "
              f"n_heads: {n_heads}")

    def reset_parameters(self):
        """
        Reset weight parameters.
        """
        self.linear_l_l.reset_parameters()
        self.linear_r_l.reset_parameters()
        glorot(self.attn)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_agg_weights: bool=False) -> torch.Tensor:
        """
        Forward pass of the One-hop Attention Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the omics features of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)
        return_agg_weights:
            If ´True´, also return the aggregation weights (attention weights).

        Returns
        ----------
        x_neighbors:
            Tensor containing the node labels of the nodes in the current node 
            batch excluding sampled neighbors. These labels are used for the 
            omics feature reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        alpha:
            Aggregation weights for edges in ´edge_index´.
        """
        x_l = x_r = x
        g_l = self.linear_l_l(x_l).view(-1, self.n_heads, self.n_input)
        g_r = self.linear_r_l(x_r).view(-1, self.n_heads, self.n_input)
        x_l = x_l.repeat(1, self.n_heads).view(-1, self.n_heads, self.n_input)

        output = self.propagate(edge_index, x=(x_l, x_r), g=(g_l, g_r))
        x_neighbors = output.mean(dim=1)
        alpha = self._alpha
        self._alpha = None
        if return_agg_weights:
            return x_neighbors, alpha
        return x_neighbors, None

    def message(self,
                x_j: torch.Tensor,
                x_i: torch.Tensor,
                g_j: torch.Tensor,
                g_i: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        """
        Message method of the MessagePassing parent class. Variables with "_i" 
        suffix refer to the central nodes that aggregate information. Variables
        with "_j" suffix refer to the neigboring nodes.

        Parameters
        ----------
        x_j:
            Gene expression of neighboring nodes (dim: n_index x n_heads x
            n_node_features).
        g_i:
            Key vector of central nodes (dim: n_index x n_heads x
            n_node_features).
        g_j:
            Query vector of neighboring nodes (dim: n_index x n_heads x
            n_node_features).     
        """
        g = g_i + g_j
        g = self.activation(g)
        alpha = (g * self.attn).sum(dim=-1)
        alpha = softmax(alpha, index) # index is 2nd dim of edge_index (index of
                                      # central node over which softmax should
                                      # be applied)
        self._alpha = alpha
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)


class OneHopGCNNormNodeLabelAggregator(nn.Module):
    """
    One-hop GCN Norm Node Label Aggregator class that uses a symmetrically
    normalized sum of the omics feature vector of a node's 1-hop neighbors to
    build an aggregated neighbor omics feature vector for a node. It returns a 
    concatenation of the node's own omics feature vector and the gcn-norm
    aggregated neighbor omics feature vector as node labels for the omics
    reconstruction task.

    modality:
        Omics modality that is aggregated. Can be either `rna` or `atac`.
    """
    def __init__(self,
                 modality: Literal["rna", "atac"]):
        super().__init__()
        print(f"ONE HOP GCN NORM {modality.upper()} NODE LABEL AGGREGATOR")

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_agg_weights: bool=False) -> torch.Tensor:
        """
        Forward pass of the One-hop GCN Norm Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the omics features of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)
        return_agg_weights:
            If ´True´, also return the aggregation weights (norm weights).
            
        Returns
        ----------
        x_neighbors:
            Tensor containing the node labels of the nodes in the current node 
            batch. These labels are used for the omics reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        alpha:
            Neighbor aggregation weights.
        """
        row, col, value = get_gcn_normalized_edges(edge_index,
                                                   num_nodes=x.shape[0],
                                                   dtype=x.dtype)
        x_neighbors = aggregate_to_columns(row, col, value, x)
        if return_agg_weights:
            # Return the weights in row-major order, as SparseTensor stores them
            order = torch.argsort(row * x.shape[0] + col, stable=True)
            alpha = value[order]
            return x_neighbors, alpha
        return x_neighbors, None


class OneHopSumNodeLabelAggregator(nn.Module):
    """
    One-hop Sum Node Label Aggregator class that sums up the omics features of
    a node's 1-hop neighbors to build an aggregated neighbor omics feature
    vector for a node. It returns a concatenation of the node's own omics
    feature vector and the sum-aggregated neighbor omics feature vector as node 
    labels for the omics reconstruction task.

    Parameters
    ----------
    modality:
        Omics modality that is aggregated. Can be either `rna` or `atac`.
    """
    def __init__(self,
                 modality: Literal["rna", "atac"]):
        super().__init__()
        print(f"ONE HOP SUM {modality.upper()} NODE LABEL AGGREGATOR")

    def forward(self,
                x: torch.Tensor,
                edge_index:torch.Tensor,
                return_agg_weights: bool=False) -> torch.Tensor:
        """
        Forward pass of the One-hop Sum Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the omics features of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)

        Returns
        ----------
        x_neighbors:
            Tensor containing the node labels of the nodes in the current node 
            batch excluding sampled neighbors. These labels are used for the 
            omics reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        """
        row, col = edge_index
        value = torch.ones(row.numel(), dtype=x.dtype, device=x.device)
        x_neighbors = aggregate_to_columns(row, col, value, x)
        return x_neighbors, None
    