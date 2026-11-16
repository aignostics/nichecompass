import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from nichecompass.data.dataloaders import InducedNeighborSampler, get_induced_edges


def test_get_induced_edges_keeps_edges_between_sampled_nodes_in_csc_order():
    # CSC of the path graph 0 - 1 - 2 - 3: incoming edges per target node.
    colptr = torch.tensor([0, 1, 3, 5, 6])
    row = torch.tensor([1, 0, 2, 1, 3, 2])
    local_index = torch.full((4,), -1)
    source, target, edge = get_induced_edges(colptr, row, torch.tensor([1, 0, 2]), local_index)
    # Targets in sampling order (1, 0, 2), sources in CSC order, node 3 left out.
    assert target.tolist() == [0, 0, 1, 2]
    assert source.tolist() == [1, 2, 0, 0]
    assert edge.tolist() == [1, 2, 0, 3]
    assert local_index.tolist() == [-1, -1, -1, -1]


def test_induced_neighbor_sampler_returns_the_induced_subgraph():
    generator = torch.Generator().manual_seed(0)
    edge_index = torch.randint(0, 50, (2, 200), generator=generator)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], 1).unique(dim=1)
    data = Data(edge_index=edge_index, num_nodes=50)
    out = InducedNeighborSampler(data, num_neighbors=[-1])._sample(torch.tensor([3, 7]))
    assert out.node[:2].tolist() == [3, 7]
    induced, _ = subgraph(out.node, edge_index, num_nodes=50)
    sampled = torch.stack([out.node[out.row], out.node[out.col]])
    assert sorted(map(tuple, sampled.t().tolist())) == sorted(map(tuple, induced.t().tolist()))
