"""Example of Blosc2 NDim optimized slicing.

It creates a 2-dimensional dataset made of different chunks, compressed with
Blosc2.  Then it proceeds to slice the dataset in ways that may and may not
benefit from Blosc2 optimized slicing.  Examples of different ways to enable
Blosc2 optimized slicing are shown.

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


def printl(*args, **kwargs):
    print(*args, **kwargs, sep='\n')

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
    assert(b2h5py.is_fast_slicing_enabled())
    # One just uses slicing as usual.
    dataset = f[dataset_name]
    # Slices with step == 1 may be optimized.
    printl("Contiguous slice from dataset (optimized):", dataset[150:, 150:])
    printl("Contiguous slice from input array:", data[150:, 150:])
    # Slices with step != 1 (or with datasets of a foreign endianness)
    # are not optimized, but still work
    # (via the HDF5 filter pipeline and hdf5plugin).
    printl("Sparse slice from dataset (filter):", dataset[150::2, 150::2])
    printl("Sparse slice from input array:", data[150::2, 150::2])
    print()

# Disabling Blosc2 optimized slicing
# ----------------------------------
# Utility functions are provided to enable and disable optimization globally.
print("# Disabling Blosc2 optimized slicing globally")
with h5py.File(file_name, 'r') as f:
    import b2h5py
    assert(b2h5py.is_fast_slicing_enabled())
    b2h5py.disable_fast_slicing()
    assert(not b2h5py.is_fast_slicing_enabled())
    dataset = f[dataset_name]
    printl("Slice from dataset (filter):", dataset[150:, 150:])
    printl("Slice from input array:", data[150:, 150:])
    b2h5py.enable_fast_slicing()  # back to normal
    assert(b2h5py.is_fast_slicing_enabled())
    print()

# Enabling Blosc2 optimized slicing temporarily
# ---------------------------------------------
# If you have disabled optimization,
# you may use a context manager to enable it only for a part of your code.
print("# Enabling Blosc2 optimized slicing temporarily")
with h5py.File(file_name, 'r') as f:
    import b2h5py
    b2h5py.disable_fast_slicing()
    assert(not b2h5py.is_fast_slicing_enabled())
    dataset = f[dataset_name]
    printl("Slice from dataset (filter):", dataset[150:, 150:])
    with b2h5py.fast_slicing():
        assert(b2h5py.is_fast_slicing_enabled())
        printl("Slice from dataset (optimized):", dataset[150:, 150:])
    assert(not b2h5py.is_fast_slicing_enabled())
    printl("Slice from input array:", data[150:, 150:])
    print()
