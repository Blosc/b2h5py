"""Example of Blosc2 NDim optimized slicing for 1-dimensional datasets.

.. _Optimized Hyper-slicing in PyTables with Blosc2 NDim:
   https://www.blosc.org/posts/pytables-b2nd-slicing/
"""

import h5py
import numpy as np

from b2h5py import B2Dataset
from hdf5plugin import Blosc2 as B2Comp

item = slice(2, 4)
with h5py.File("test.h5", "w") as h5f:
    a = np.arange(100, dtype=np.int8)
    dset1 = h5f.create_dataset("1d-blosc2", data=a, chunks=(5,), **B2Comp())

    a = np.arange(100, dtype=np.int8).reshape(10, 10)
    dset2 = h5f.create_dataset("2d-blosc2", data=a, chunks=(5,5), **B2Comp())

    slice_ = dset1[item]
    b2slice_ = B2Dataset(dset1)[item]
    print("dset1 via h5py", slice_)
    print("dset1 via b2h5py", b2slice_)
    np.testing.assert_array_equal(slice_, b2slice_)

    slice_ = dset2[item]
    b2slice_ = B2Dataset(dset2)[item]
    print("dset1 via h5py", slice_)
    print("dset1 via b2h5py", b2slice_)
    np.testing.assert_array_equal(slice_, b2slice_)
