"""Test using B2Dataset to enable fast access to Blosc2 compressed dataset.
"""

import h5py
import hdf5plugin
import numpy as np
import pytest

from b2h5py import B2Dataset


@pytest.fixture
def hdf5_test_dataset(tmp_path):
    """Text fixture creating a hdf5 file with a "data" dataset compression with Blosc2"""
    shape = 3500, 300
    chunks = 1747, 150
    data = np.arange(np.prod(shape), dtype="u2").reshape(shape)

    with h5py.File(tmp_path / "test_file.h5", "w") as f:
        compression = hdf5plugin.Blosc2(cname='lz4', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)
        f.create_dataset('data', data=data, chunks=chunks, **compression)

    with h5py.File(tmp_path / "test_file.h5", "r") as f:
        yield f["data"]


def test_B2Dataset(hdf5_test_dataset):
    b2dataset = B2Dataset(hdf5_test_dataset)
    assert b2dataset.is_b2_fast_access

    # Access h5py.Dataset attribute
    assert b2dataset.chunks == hdf5_test_dataset.chunks

    # Access whole array
    assert np.array_equal(b2dataset[()], hdf5_test_dataset[()])
    # Unoptimized access
    assert np.array_equal(b2dataset[::2], hdf5_test_dataset[::2])
