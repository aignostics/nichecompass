import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from nichecompass.data.datasets import SpatialAnnTorchDataset


def make_adata(adj: np.ndarray) -> ad.AnnData:
    adata = ad.AnnData(X=np.ones((adj.shape[0], 2), dtype=np.float32))
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(adj)
    return adata


def test_dataset_accepts_symmetric_adjacency():
    adj = np.array([[0., 1., 0.], [1., 0., 2.], [0., 2., 0.]])
    dataset = SpatialAnnTorchDataset(make_adata(adj), [], counts_key=None, self_loops=False)
    assert dataset.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]


def test_dataset_rejects_asymmetric_adjacency():
    adj = np.array([[0., 1., 0.], [0., 0., 1.], [0., 1., 0.]])
    with pytest.raises(ValueError, match="symmetric"):
        SpatialAnnTorchDataset(make_adata(adj), [], counts_key=None, self_loops=False)
