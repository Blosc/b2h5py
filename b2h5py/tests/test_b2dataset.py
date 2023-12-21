"""Test using B2Dataset to enable fast access to Blosc2 compressed dataset.
"""

import numpy as np

from b2h5py import B2Dataset
from b2h5py.tests.common import StoreArrayMixin
from h5py.tests.common import TestCase


class TestB2Dataset(TestCase, StoreArrayMixin):
    def setUp(self):
        TestCase.setUp(self)

        shape = 3500, 300
        self.chunks = 1747, 150
        self.arr = np.arange(np.prod(shape), dtype="u2").reshape(shape)
        StoreArrayMixin.setUp(self)

    def testB2Dataset(self):
        b2dataset = B2Dataset(self.dset)
        self.assertTrue(b2dataset.is_b2_fast_access)

        # Access h5py.Dataset attribute
        self.assertEqual(b2dataset.chunks, self.dset.chunks)

        # Access whole array
        self.assertArrayEqual(b2dataset[()], self.arr)
        # Unoptimized access
        self.assertArrayEqual(b2dataset[::2], self.arr[::2])
