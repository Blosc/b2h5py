"""Example of Blosc2 NDim optimized slicing.

It creates a 2-dimensional dataset made of different chunks, compressed with
Blosc2.  Then it proceeds to slice the dataset in ways that may and may not
benefit from Blosc2 optimized slicing.  Some hints about forcing the use of
the HDF5 filter pipeline are included, as well as comments on the Python
package dependencies required for the different use cases.

Optimized slicing can provide considerable speed-ups in certain use cases,
please see `this benchmark`__ which evaluates applying the same technique in
PyTables, and the post `Optimized Hyper-slicing in PyTables with Blosc2
NDim`_, which presents the results of the benchmark.

__ https://github.com/PyTables/PyTables/blob/master/bench/b2nd_compare_getslice.py

.. _Optimized Hyper-slicing in PyTables with Blosc2 NDim:
   https://www.blosc.org/posts/pytables-b2nd-slicing/
"""

import os

import h5py
import numpy as np


# The array created below is to be stored in a dataset
# 3 chunks top-down, 3 chunks accross,
# with fringe chunks partially filled by array data::
#
#      <- 100 -X- 100 -X-50->···
#    ^ +-------+-------+----+··+
#    1 |       |       |    |  ·
#    0 |  #0   |  #1   | #2 |  ·
#    0 |       |       |    |  ·
#    X +-------+-------+----+··+
#    1 |       |       |    |  ·
#    0 |  #3   |  #4   | #5 |  ·
#    0 |       |       |    |  ·
#    X +-------+-------+----+··+
#    5 |       |       |    |  ·
#    0 |  #6   |  #7   | #8 |  ·
#    v +-------+-------+----+  ·
#    · +·······+·······+·······+
#
shape = (250, 250)
chunks = (100, 100)
data = np.arange(np.prod(shape)).reshape(shape)

file_name = 'b2nd-example.h5'
dataset_name = 'data'

# Creating a Blosc2-compressed dataset
# ------------------------------------
with h5py.File(file_name, 'w') as f:
    # This import is needed to declare Blosc2 compression parameters
    # for a newly created dataset.
    # For the moment, all writes to Blosc2-compressed datasets
    # use the HDF5 filter pipeline, so only hdf5plugin is needed.
    import hdf5plugin as h5p
    comp = h5p.Blosc2(cname='lz4', clevel=5, filters=h5p.Blosc2.SHUFFLE)
    dataset = f.create_dataset(dataset_name, data=data, **comp)

# Benefitting from Blosc2 optimized slicing
# -----------------------------------------
# After importing `b2h5py`,
# support for Blosc2 optimized slicing is enabled by default.
print("# Using Blosc2 optimized slicing")
with h5py.File(file_name, 'r') as f:
    import b2h5py
    assert(b2h5py.is_dataset_class_patched())
    # One just uses slicing as usual.
    dataset = f[dataset_name]
    # Slices with step == 1 may be optimized.
    slice_ = dataset[150:, 150:]
    print("Contiguous slice from dataset (optimized):", slice_, sep='\n')
    print("Contiguous slice from input array:", data[150:, 150:], sep='\n')
    # Slices with step != 1 (or with datasets of a foreign endianness)
    # are not optimized, but still work
    # (via the HDF5 filter pipeline and hdf5plugin).
    slice_ = dataset[150::2, 150::2]
    print("Sparse slice from dataset (filter):", slice_, sep='\n')
    print("Sparse slice from input array:", data[150::2, 150::2], sep='\n')
    print()

# Disabling Blosc2 optimized slicing
# ----------------------------------
# Utility functions are provided to enable and disable optimization globally.
print("# Disabling Blosc2 optimized slicing globally")
with h5py.File(file_name, 'r') as f:
    import b2h5py
    assert(b2h5py.is_dataset_class_patched())
    b2h5py.unpatch_dataset_class()
    assert(not b2h5py.is_dataset_class_patched())
    dataset = f[dataset_name]
    slice_ = dataset[150:, 150:]
    print("Slice from dataset (filter):", slice_, sep='\n')
    print("Slice from input array:", data[150:, 150:], sep='\n')
    b2h5py.patch_dataset_class()  # back to normal
    assert(b2h5py.is_dataset_class_patched())
    print()

# Enabling Blosc2 optimized slicing temporarily
# ---------------------------------------------
# If you have disabled optimization,
# you may use a context manager to enable it only for a part of your code.
print("# Enabling Blosc2 optimized slicing temporarily")
with h5py.File(file_name, 'r') as f:
    import b2h5py
    b2h5py.unpatch_dataset_class()
    assert(not b2h5py.is_dataset_class_patched())
    dataset = f[dataset_name]
    slice_ = dataset[150:, 150:]
    print("Slice from dataset (filter):", slice_, sep='\n')
    with b2h5py.patching_dataset_class():
        assert(b2h5py.is_dataset_class_patched())
        slice_ = dataset[150:, 150:]
        print("Slice from dataset (optimized):", slice_, sep='\n')
    assert(not b2h5py.is_dataset_class_patched())
    print("Slice from input array:", data[150:, 150:], sep='\n')
    print()
