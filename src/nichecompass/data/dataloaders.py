"""
This module contains dataloaders for the training of an NicheCompass model.
"""

import copy
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.sampler import NeighborSampler, SamplerOutput


def get_induced_edges(colptr: torch.Tensor,
                      row: torch.Tensor,
                      node: torch.Tensor,
                      local_index: torch.Tensor) -> tuple[torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor]:
    """
    Get all edges between the sampled nodes, in the order torch_sparse returns
    them for induced subgraphs: for each sampled node in sampling order, its
    incoming edges in CSC order whose source node was sampled as well.

    Parameters
    ----------
    colptr:
        CSC column pointers of the full graph.
    row:
        CSC row indices (source nodes) of the full graph.
    node:
        Global indices of the sampled nodes in sampling order.
    local_index:
        Buffer with one entry per node of the full graph that is -1 for every
        node. It is used to map global to local node indices and is -1 again
        when the function returns.

    Returns
    ----------
    source:
        Local index of the source node of each induced edge.
    target:
        Local index of the target node of each induced edge.
    edge:
        CSC position of each induced edge.
    """
    local_index[node] = torch.arange(node.numel())
    start = colptr[node]
    count = colptr[node + 1] - start
    n_candidates = int(count.sum())
    target = torch.repeat_interleave(torch.arange(node.numel()), count)
    group_start = torch.cumsum(count, 0) - count
    edge = (torch.repeat_interleave(start - group_start, count)
            + torch.arange(n_candidates))
    source = local_index[row[edge]]
    local_index[node] = -1
    keep = source >= 0
    return source[keep], target[keep], edge[keep]


class InducedNeighborSampler(NeighborSampler):
    """
    Neighbor sampler that returns the induced subgraph of the sampled nodes
    for homogeneous graphs.

    PyG samples induced subgraphs only with torch_sparse, which has no wheels
    for recent torch versions. This sampler samples the nodes with the
    directional sampler, which pyg-lib supports, and then adds all edges
    between the sampled nodes. The nodes, edges and their order are the same
    as with torch_sparse when all neighbors are sampled (`num_neighbors` of
    -1).
    """
    def __init__(self, data: Data, num_neighbors: list, **kwargs):
        if not data.edge_index.is_contiguous():
            # pyg-lib sorts the edges into CSC format and needs a contiguous
            # edge index.
            data = copy.copy(data)
            data.edge_index = data.edge_index.contiguous()
        super().__init__(data, num_neighbors, subgraph_type="directional",
                         **kwargs)
        self._local_index = torch.full((self.num_nodes,), -1, dtype=torch.long)

    def _sample(self,
                seed: torch.Tensor,
                seed_time: Optional[torch.Tensor]=None,
                **kwargs) -> SamplerOutput:
        if isinstance(seed, dict):
            raise NotImplementedError(
                "InducedNeighborSampler supports homogeneous graphs only.")
        out = super()._sample(seed, seed_time, **kwargs)
        source, target, edge = get_induced_edges(
            self.colptr, self.row, out.node, self._local_index)
        return SamplerOutput(node=out.node,
                             row=source,
                             col=target,
                             edge=edge,
                             batch=out.batch)


def initialize_dataloaders(node_masked_data: Data,
                           edge_train_data: Optional[Data]=None,
                           edge_val_data: Optional[Data]=None,
                           edge_batch_size: Optional[int]=64,
                           node_batch_size: int=64,
                           n_direct_neighbors: int=-1,
                           n_hops: int=1,
                           shuffle: bool=True,
                           edges_directed: bool=False,
                           neg_edge_sampling_ratio: float=1.) -> dict:
    """
    Initialize edge-level and node-level training and validation dataloaders.

    Parameters
    ----------
    node_masked_data:
        PyG Data object with node-level split masks.
    edge_train_data:
        PyG Data object containing the edge-level training set.
    edge_val_data:
        PyG Data object containing the edge-level validation set.
    edge_batch_size:
        Batch size for the edge-level dataloaders.
    node_batch_size:
        Batch size for the node-level dataloaders.
    n_direct_neighbors:
        Number of sampled direct neighbors of the current batch nodes to be 
        included in the batch. Defaults to ´-1´, which means to include all 
        direct neighbors.
    n_hops:
        Number of neighbor hops / levels for neighbor sampling of nodes to be 
        included in the current batch. E.g. ´2´ means to not only include 
        sampled direct neighbors of current batch nodes but also sampled 
        neighbors of the direct neighbors.
    shuffle:
        If `True`, shuffle the dataloaders.
    edges_directed:
        If `False`, both symmetric edge index pairs are included in the same 
        edge-level batch (1 edge has 2 symmetric edge index pairs).
    neg_edge_sampling_ratio:
        Negative sampling ratio of edges. This is currently implemented in an
        approximate way, i.e. negative edges may contain false negatives.

    Returns
    ----------
    loader_dict:
        Dictionary containing training and validation PyG LinkNeighborLoader 
        (for edge reconstruction) and NeighborLoader (for gene expression 
        reconstruction) objects.
    """
    loader_dict = {}

    # Node-level dataloaders
    loader_dict["node_train_loader"] = NeighborLoader(
        node_masked_data,
        num_neighbors=[n_direct_neighbors] * n_hops,
        batch_size=node_batch_size,
        shuffle=shuffle,
        input_nodes=node_masked_data.train_mask,
        neighbor_sampler=InducedNeighborSampler(
            node_masked_data, num_neighbors=[n_direct_neighbors] * n_hops))
    if node_masked_data.val_mask.sum() != 0:
        loader_dict["node_val_loader"] = NeighborLoader(
            node_masked_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=node_batch_size,
            shuffle=shuffle,
            input_nodes=node_masked_data.val_mask,
            neighbor_sampler=InducedNeighborSampler(
                node_masked_data, num_neighbors=[n_direct_neighbors] * n_hops))
        
    # Edge-level dataloaders
    if edge_train_data is not None:
        loader_dict["edge_train_loader"] = LinkNeighborLoader(
            edge_train_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=edge_train_data.edge_label_index[:, edge_train_data.edge_label.bool()], # limit the edges to the ones from the edge_label_adj
            shuffle=shuffle,
            neighbor_sampler=(None if edges_directed else InducedNeighborSampler(
                edge_train_data, num_neighbors=[n_direct_neighbors] * n_hops)),
            neg_sampling_ratio=neg_edge_sampling_ratio)
    if edge_val_data is not None and edge_val_data.edge_label.sum() != 0:
        loader_dict["edge_val_loader"] = LinkNeighborLoader(
            edge_val_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=edge_val_data.edge_label_index[:, edge_val_data.edge_label.bool()], # limit the edges to the ones from the edge_label_adj
            shuffle=shuffle,
            neighbor_sampler=(None if edges_directed else InducedNeighborSampler(
                edge_val_data, num_neighbors=[n_direct_neighbors] * n_hops)),
            neg_sampling_ratio=neg_edge_sampling_ratio)

    return loader_dict