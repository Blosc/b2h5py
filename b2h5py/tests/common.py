"""Common code for b2h5py tests."""

import functools

import b2h5py
import h5py
import hdf5plugin as h5p


class Blosc2OptNotUsedError(Exception):
    """Blosc2 optimization was not used by unit test"""
    pass


class StoreArrayMixin:
    # Requires: self.f (read/write), self.arr, self.chunks
    # Provides: self.f (read-only), self.dset
    def setUp(self):
        comp = h5p.Blosc2(cname='lz4', clevel=5, filters=h5p.Blosc2.SHUFFLE)
        self.f.create_dataset('x', data=self.arr, chunks=self.chunks, **comp)

        # Reopen the test file read-only to ensure
        # that no HDF5/h5py caching takes place.
        fn = self.f.filename
        self.f.close()
        self.f = h5py.File(fn, 'r')
        self.dset = self.f['x']


def check_opt_slicing(test):
    """Decorate `test` to fail if slicing did not use expected optimization"""
    @functools.wraps(test)
    def checked_test(self):
        if not self.should_enable_opt():
            return test(self)
        # If the dataset class is not patched,
        # the exception set below is never raised anyway.
        self.assertTrue(b2h5py.is_fast_slicing_enabled())
        # Force an exception if the optimization is not used.
        orig_exc = b2h5py.blosc2._no_opt_error
        b2h5py.blosc2._no_opt_error = Blosc2OptNotUsedError
        try:
            return test(self)
        finally:
            b2h5py.blosc2._no_opt_error = orig_exc
    return checked_test
