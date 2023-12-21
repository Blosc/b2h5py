"""Test using B2Dataset to enable fast access to Blosc2 compressed dataset.
"""

import h5py
import hdf5plugin
import numpy as np

from b2h5py import B2Dataset
from h5py.tests.common import TestCase


class TestB2Dataset(TestCase):
    def setUp(self):
        super().setUp()

        shape = 3500, 300
        chunks = 1747, 150
        self.data = np.arange(np.prod(shape), dtype="u2").reshape(shape)

        compression = hdf5plugin.Blosc2(cname='lz4', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)
        self.f.create_dataset('data', data=self.data, chunks=chunks, **compression)

        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, "r")
        self.h5dataset = self.f["data"]

    def tearDown(self):
        self.h5dataset = None
        super().tearDown()

    def testB2Dataset(self):
        b2dataset = B2Dataset(self.h5dataset)
        self.assertTrue(b2dataset.is_b2_fast_access)

        # Access h5py.Dataset attribute
        self.assertEqual(b2dataset.chunks, self.h5dataset.chunks)

        # Access whole array
        self.assertArrayEqual(b2dataset[()], self.data)
        # Unoptimized access
        self.assertArrayEqual(b2dataset[::2], self.data[::2])

