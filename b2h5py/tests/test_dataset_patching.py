"""Dataset patching test module.

Tests that monkey-patching ``h5py.Dataset`` works as expected.
"""

import functools

import b2h5py  # monkey-patches h5py.Dataset

from h5py import Dataset
from h5py.tests.common import TestCase


class Blosc2DatasetPatchingTestCase(TestCase):
    def setUp(self):
        super().setUp()
        b2h5py.patch_dataset_class()

    def tearDown(self):
        b2h5py.patch_dataset_class()
        super().tearDown()

    def test_default(self):
        """Dataset class is patched by default"""
        self.assertTrue(b2h5py.is_dataset_class_patched())

    def test_unpatch_patch(self):
        """Unpatching and patching dataset class again"""
        b2h5py.unpatch_dataset_class()
        self.assertFalse(b2h5py.is_dataset_class_patched())

        b2h5py.patch_dataset_class()
        self.assertTrue(b2h5py.is_dataset_class_patched())

    def test_patch_again(self):
        """Patching the dataset class twice"""
        b2h5py.patch_dataset_class()
        getitem1 = Dataset.__getitem__
        b2h5py.patch_dataset_class()
        getitem2 = Dataset.__getitem__

        self.assertIs(getitem1, getitem2)

    def test_unpatch_again(self):
        """Unpatching the dataset class twice"""
        b2h5py.unpatch_dataset_class()
        getitem1 = Dataset.__getitem__
        b2h5py.unpatch_dataset_class()
        getitem2 = Dataset.__getitem__

        self.assertIs(getitem1, getitem2)

    def test_patch_patched(self):
        """Patching when already patched by someone else"""
        b2h5py.unpatch_dataset_class()

        @functools.wraps(Dataset.__getitem__)
        def foreign_getitem(self, args, new_dtype=None):
            return 42

        Dataset.__getitem__ = foreign_getitem

        try:
            b2h5py.patch_dataset_class()
            self.assertTrue(b2h5py.is_dataset_class_patched())
            self.assertIs(Dataset.__getitem__.__wrapped__, foreign_getitem)

            b2h5py.unpatch_dataset_class()
            self.assertFalse(b2h5py.is_dataset_class_patched())
            self.assertIs(Dataset.__getitem__, foreign_getitem)
        finally:
            b2h5py.unpatch_dataset_class()
            Dataset.__getitem__ = foreign_getitem.__wrapped__

    def test_unpatch_foreign(self):
        """Unpatching when patched over by someone else"""

        @functools.wraps(Dataset.__getitem__)
        def foreign_getitem(self, args, new_dtype=None):
            return 42

        Dataset.__getitem__ = foreign_getitem

        try:
            with self.assertRaises(ValueError):
                b2h5py.unpatch_dataset_class()
        finally:
            Dataset.__getitem__ = foreign_getitem.__wrapped__
