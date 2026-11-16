import numpy as np
import scipy.sparse as sp
import torch

from nichecompass.data.utils import sparse_mx_to_sparse_tensor
from nichecompass.nn.aggregators import OneHopGCNNormNodeLabelAggregator, OneHopSumNodeLabelAggregator


def test_gcn_norm_aggregator_matches_hand_computed_weights():
    # Path graph 0 - 1 - 2 with a self loop on node 0 that gets replaced.
    edge_index = torch.tensor([[0, 0, 1, 1, 2],
                               [0, 1, 0, 2, 1]])
    x = torch.tensor([[1., 0.], [0., 1.], [2., 2.]])
    aggregator = OneHopGCNNormNodeLabelAggregator(modality="rna")
    x_neighbors, alpha = aggregator(x, edge_index, return_agg_weights=True)

    # Degrees with one self loop each: 2, 3, 2.
    deg_inv_sqrt = 1 / np.sqrt([2., 3., 2.])
    adj = np.array([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    np.testing.assert_allclose(x_neighbors.numpy(), adj_norm.T @ x.numpy(), rtol=1e-6)
    # Weights in row-major order of the non-zero entries.
    np.testing.assert_allclose(alpha.numpy(), adj_norm[adj > 0], rtol=1e-6)


def test_sum_aggregator_sums_source_features_into_targets():
    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 1]])
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    x_neighbors, _ = OneHopSumNodeLabelAggregator(modality="rna")(x, edge_index)
    expected = torch.tensor([[0., 0.], [6., 8.], [3., 4.]])
    assert torch.equal(x_neighbors, expected)


def test_sparse_mx_to_sparse_tensor_keeps_entries_in_row_major_order():
    matrix = sp.csr_matrix(np.array([[0., 2., 0.], [1., 0., 3.], [0., 4., 0.]]))
    tensor = sparse_mx_to_sparse_tensor(matrix)
    assert tensor.is_coalesced()
    assert tensor.indices().tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    np.testing.assert_array_equal(tensor.to_dense().numpy(), matrix.toarray())
