"""Test using B2Dataset to enable fast access to Blosc2 compressed dataset.
"""

import os.path
import tempfile
from unittest import TestCase

import h5py
import hdf5plugin
import numpy as np

from b2h5py import B2Dataset


class TestB2Dataset(TestCase):
    def setUp(self):
        shape = 3500, 300
        chunks = 1747, 150
        self.data = np.arange(np.prod(shape), dtype="u2").reshape(shape)

        self.tempdir = tempfile.TemporaryDirectory()
        filename = os.path.join(self.tempdir.name, "test_file.h5")

        with h5py.File(filename, "w") as f:
            compression = hdf5plugin.Blosc2(cname='lz4', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)
            f.create_dataset('data', data=self.data, chunks=chunks, **compression)

        self.h5file = h5py.File(filename, "r")
        self.h5dataset = self.h5file["data"]

    def tearDown(self):
        self.h5dataset = None
        self.h5file.close()
        self.tempdir.cleanup()

    def testB2Dataset(self):
        b2dataset = B2Dataset(self.h5dataset)
        self.assertTrue(b2dataset.is_b2_fast_access)

        # Access h5py.Dataset attribute
        self.assertEqual(b2dataset.chunks, self.h5dataset.chunks)

        # Access whole array
        self.assertTrue(np.array_equal(b2dataset[()], self.data))
        # Unoptimized access
        self.assertTrue(np.array_equal(b2dataset[::2], self.data[::2]))

