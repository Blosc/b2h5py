"""Dataset Blosc2 optimized slicing test module.

Tests slice read operations for the cases where Blosc2 optimized slicing can
be used.
"""

import os
import random

import hdf5plugin
import b2h5py
import h5py
import numpy as np

from b2h5py.tests.common import (Blosc2OptNotUsedError,
                                 StoreArrayMixin,
                                 check_opt_slicing)
from h5py.tests.common import TestCase


class SimpleSlicingTestCase(TestCase):
    def setUp(self):
        # Create test file and datasets
        b2comp = hdf5plugin.Blosc2()
        self.filename = "test_slicing.h5"
        with h5py.File(self.filename, "w") as h5f:
            # 1D dataset
            a1d = np.arange(100, dtype=np.int64)
            h5f.create_dataset("1d-blosc2", data=a1d, chunks=(5,), **b2comp)

            # 2D dataset
            a2d = np.arange(100, dtype=np.int64).reshape(10, 10)
            h5f.create_dataset("2d-blosc2", data=a2d, chunks=(5, 5), **b2comp)

    def tearDown(self):
        # Clean up test file
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_1d_slicing(self):
        """Test slicing on 1D dataset"""
        with h5py.File(self.filename, "r") as h5f:
            dset = h5f["1d-blosc2"]
            b2dset = b2h5py.B2Dataset(dset)

            # Test different slice patterns
            slices = [
                slice(2, 4),
                slice(0, 10),
                slice(10, 20),
                slice(None, None, 2),
                slice(50, None),
                slice(None, 50)
            ]

            for item in slices:
                regular_slice = dset[item]
                b2_slice = b2dset[item]
                np.testing.assert_array_equal(
                    regular_slice, b2_slice,
                    f"Slices don't match for {item}"
                )

    def test_2d_slicing(self):
        """Test slicing on 2D dataset"""
        with h5py.File(self.filename, "r") as h5f:
            dset = h5f["2d-blosc2"]
            b2dset = b2h5py.B2Dataset(dset)

            # Test with integer index (row selection)
            for i in range(3):
                regular_slice = dset[i]
                b2_slice = b2dset[i]
                np.testing.assert_array_equal(
                    regular_slice, b2_slice,
                    f"Row slices don't match for index {i}"
                )

            # Test with slice on first dimension
            slices = [
                (slice(2, 4), slice(None)),
                (slice(0, 5), slice(None)),
                (slice(None, None, 2), slice(None))
            ]

            for item in slices:
                regular_slice = dset[item]
                b2_slice = b2dset[item]
                np.testing.assert_array_equal(
                    regular_slice, b2_slice,
                    f"Slices don't match for {item}"
                )


class Blosc2OptSlicingTestCase(TestCase, StoreArrayMixin):
    """Blosc2 optimized slicing by patching dataset class"""

    blosc2_force_filter = False

    def setUp(self):
        TestCase.setUp(self)

        shape = (3500, 300)
        self.chunks = (1747, 150)
        self.arr = np.arange(np.prod(shape), dtype="u2").reshape(shape)
        StoreArrayMixin.setUp(self)

        self.blosc2_filter_env = os.environ.get('BLOSC2_FILTER', '0')
        os.environ['BLOSC2_FILTER'] = '1' if self.blosc2_force_filter else '0'
        b2h5py.enable_fast_slicing()

    def tearDown(self):
        b2h5py.disable_fast_slicing()
        os.environ['BLOSC2_FILTER'] = self.blosc2_filter_env
        super().tearDown()

    def should_enable_opt(self):
        return not self.blosc2_force_filter

    def test_disabled(self):
        """Non-use detection working."""
        if not self.should_enable_opt():
            return

        @check_opt_slicing
        def test(self):
            self.dset[::2]  # step != 1 not supported currently

        with self.assertRaises(Blosc2OptNotUsedError):
            test(self)

    # Test the data of the returned object.

    @check_opt_slicing
    def test_whole_array(self):
        """Reading a slice covering the whole array"""
        self.assertArrayEqual(self.dset[:], self.arr)

    @check_opt_slicing
    def test_cross_chunk_1dim(self):
        """Reading a slice crossing chunk boundaries (1-dim)"""
        slc = slice(self.dset.chunks[0] - 5, self.dset.chunks[0] + 5)
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_cross_chunk_ndim(self):
        """Reading a slice crossing chunk boundaries (n-dim)"""
        slc = (slice(self.dset.chunks[0] - 5, self.dset.chunks[0] + 5),
               slice(self.dset.chunks[1] - 5, self.dset.chunks[1] + 5))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_last_chunk_1dim(self):
        """Reading a slice going past the last chunk (1-dim)"""
        slc = slice(self.dset.shape[0] - 5, self.dset.shape[0] + 5)
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_last_chunk_ndim(self):
        """Reading a slice going past the last chunk (n-dim)"""
        slc = (slice(self.dset.shape[0] - 5, self.dset.shape[0] + 5),
               slice(self.dset.shape[1] - 5, self.dset.shape[1] + 5))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    # Test the attributes of the returned object.

    @check_opt_slicing
    def test_scalar_inside(self):
        """Reading a scalar inside of the array"""
        coord = tuple(random.randrange(0, c) for c in self.dset.shape)
        self.assertEqual(self.dset[coord], self.arr[coord])

    @check_opt_slicing
    def test_scalar_outside(self):
        """Reading a scalar outside of the array"""
        shape = self.dset.shape
        coords = [(shape[0] * 2, 0), (0, shape[1] * 2),
                  tuple(c * 2 for c in shape)]
        for coord in coords:
            with self.assertRaises(IndexError):
                self.dset[coord]

    @check_opt_slicing
    def test_slice_outside(self):
        """Reading a slice outside the array (empty)"""
        shape = self.dset.shape
        slcs = [(slice(shape[0] * 2, shape[0] * 3), ...),
                (..., slice(shape[1] * 2, shape[1] * 3)),
                tuple(slice(c * 2, c * 3) for c in shape)]
        for slc in slcs:
            self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_slice_1dimless(self):
        """Reading a slice with one dimension less than the array"""
        idxs = [random.randrange(0, dim) for dim in self.dset.shape]
        for idx in idxs:
            self.assertArrayEqual(self.dset[idx], self.arr[idx])

    @check_opt_slicing
    def test_astype(self):
        """Reading a slice converted to another type"""
        alt_dtype = np.dtype('u4')
        self.assertTrue(self.dset.dtype < alt_dtype)
        alt_arr = self.arr.astype(alt_dtype)
        alt_dset = self.dset.astype(alt_dtype)
        slc = slice(10, 20)
        self.assertArrayEqual(alt_dset[slc], alt_arr[slc])


class Blosc2FiltSlicingTestCase(Blosc2OptSlicingTestCase):
    """Blosc2 filter slicing forced by environment variable"""

    blosc2_force_filter = True


class Blosc2UnpatchTestCase(Blosc2OptSlicingTestCase):
    """Blosc2 filter slicing (original h5py) by unpatching dataset class"""

    def setUp(self):
        super().setUp()
        b2h5py.disable_fast_slicing()

    def should_enable_opt(self):
        return False


class Blosc2OptSlicingMinTestCase(TestCase, StoreArrayMixin):
    """Blosc2 optimized slicing with chunks on inner dimension"""

    # Minimal test which can be figured out manually::
    #
    #     z  Data: 1   Chunk0:   Chunk1: 1   Slice:
    #    /        /|\                    |\
    #   |\       0 5 3       0           5 3        5
    #   x y      |X X|       |\           \|       / \
    #            4 2 7       4 2           7      4   7
    #             \|/         \|                   \ /
    #              6           6                    6
    #
    #                  Chunk0 & Slice: 4   Chunk1 & Slice: 5
    #                                   \                   \
    #                                    6                   7
    #
    # This is mainly a test for the assemblage of the returned slice
    # from its parts in different chunks.

    def setUp(self):
        TestCase.setUp(self)

        shape = (2, 2, 2)
        self.chunks = (2, 2, 1)
        self.arr = np.arange(np.prod(shape), dtype="u1").reshape(shape)
        StoreArrayMixin.setUp(self)
        b2h5py.enable_fast_slicing()

    def tearDown(self):
        b2h5py.disable_fast_slicing()
        TestCase().tearDown()

    def should_enable_opt(self):
        return True

    @check_opt_slicing
    def test_slice(self):
        """Reading a slice perpendicular to chunks"""
        slc = (slice(1, 2), slice(0, 2), slice(0, 2))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])


class Blosc2OptSlicingCompTestCase(TestCase, StoreArrayMixin):
    """Blosc2 optimized slicing with compound dtypes"""

    def setUp(self):
        TestCase.setUp(self)

        dtype = np.dtype('i4,f8')
        shape = (5, 5)
        self.chunks = (2, 2)
        arr = np.zeros(dtype=dtype, shape=shape)
        arr[0, 0] = (1, 1)
        arr[0, 2] = (2, 2)
        arr[0, 4] = (3, 3)
        arr[2, 0] = (4, 4)
        arr[2, 2] = (5, 5)
        arr[2, 4] = (6, 6)
        arr[4, 0] = (7, 7)
        arr[4, 2] = (8, 8)
        arr[4, 4] = (9, 9)
        self.arr = arr
        StoreArrayMixin.setUp(self)
        b2h5py.enable_fast_slicing()

    def tearDown(self):
        b2h5py.disable_fast_slicing()
        TestCase().tearDown()

    def should_enable_opt(self):
        return True

    @check_opt_slicing
    def test_whole_array(self):
        """Reading a slice covering the whole array"""
        self.assertArrayEqual(self.dset[:], self.arr)

    @check_opt_slicing
    def test_chunk(self):
        """Reading a slice matching a chunk"""
        slcs = [tuple(slice(mult * cl, (mult + 1) * cl)
                      for cl in self.dset.chunks)
                for mult in range(3)]
        for slc in slcs:
            self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_cross_chunk(self):
        """Reading a slice crossing chunk boundaries"""
        slc = (slice(1, 1), slice(3, 3))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_last_chunk(self):
        """Reading a slice going past the last chunk"""
        slc = (slice(-1, self.dset.shape[0] + 10),
               slice(-1, self.dset.shape[1] + 10))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])

    @check_opt_slicing
    def test_cross_last_chunk(self):
        """Reading a slice crossing chunk boundaries past last chunk"""
        slc = (slice(-2, self.dset.shape[0] + 10),
               slice(-2, self.dset.shape[1] + 10))
        self.assertArrayEqual(self.dset[slc], self.arr[slc])
