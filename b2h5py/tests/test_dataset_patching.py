"""Dataset patching test module.

Tests that monkey-patching ``h5py.Dataset`` works as expected.
"""

import contextlib
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


class CMTestError(Exception):
    pass


class ContextManagerTestCase(TestCase):
    """Blosc2 patching context manager (no exception raised)"""

    shall_raise = False

    def setUp(self):
        super().setUp()
        b2h5py.unpatch_dataset_class()

    def tearDown(self):
        b2h5py.patch_dataset_class()
        super().tearDown()

    def patching_cmgr(self):
        """Checks for error if `self.shall_raise`, patches dataset class"""
        test_case = self

        class CMTestContextManager(contextlib.ExitStack):
            def __enter__(self):
                if test_case.shall_raise:
                    self.enter_context(test_case.assertRaises(CMTestError))
                self.enter_context(b2h5py.patching_dataset_class())
                return super().__enter__()

        return CMTestContextManager()

    def maybe_raise(self):
        if self.shall_raise:
            raise CMTestError

    def test_default(self):
        """Dataset class is patched then unpatched"""
        self.assertFalse(b2h5py.is_dataset_class_patched())
        with self.patching_cmgr():
            self.assertTrue(b2h5py.is_dataset_class_patched())
            self.maybe_raise()
        self.assertFalse(b2h5py.is_dataset_class_patched())

    def test_exception(self):
        """Exceptions are propagated"""
        # This test always raises, do not use `self.patching_cmgr()`.
        with self.assertRaises(CMTestError):
            with b2h5py.patching_dataset_class():
                raise CMTestError

    def test_already_patched(self):
        """Not unpatching if already patched before entry"""
        b2h5py.patch_dataset_class()
        self.assertTrue(b2h5py.is_dataset_class_patched())
        with self.patching_cmgr():
            self.assertTrue(b2h5py.is_dataset_class_patched())
            self.maybe_raise()
        self.assertTrue(b2h5py.is_dataset_class_patched())

    def test_nested(self):
        """Nesting patching context managers"""
        self.assertFalse(b2h5py.is_dataset_class_patched())
        with self.patching_cmgr():
            self.assertTrue(b2h5py.is_dataset_class_patched())
            with self.patching_cmgr():
                self.assertTrue(b2h5py.is_dataset_class_patched())
                self.maybe_raise()
            self.assertTrue(b2h5py.is_dataset_class_patched())
            self.maybe_raise()
        self.assertFalse(b2h5py.is_dataset_class_patched())


class ErrorContextManagerTestCase(ContextManagerTestCase):
    """Blosc2 patching context manager (exception raised)"""

    shall_raise = True
