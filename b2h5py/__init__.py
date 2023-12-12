"""Transparent optimized reading of n-dimensional Blosc2 slices for h5py.

Optimizations are applied to slices of the form ``dataset[...]`` or
``dataset.__getitem__(...)`` with step 1 on Blosc2-compressed datasets using
the native byte order.

They are enabled automatically on module import, by monkey-patching the
``h5py.Dataset`` class.  You may explicitly undo this patching and deactivate
optimization globally with `disable_fast_slicing()` and redo it and activate
it again with `enable_fast_slicing()`.  You may also patch the class
temporarily using `patching_dataset_class()` to get a context manager.

**Note:** For testing and debugging purposes, you may force-disable the
optimization at any time by setting ``BLOSC2_FILTER=1`` in the environment.
"""

from .blosc2 import (disable_fast_slicing,
                     enable_fast_slicing,
                     is_dataset_class_patched,
                     patching_dataset_class)


__all__ = ['disable_fast_slicing',
           'enable_fast_slicing',
           'is_dataset_class_patched',
           'patching_dataset_class']


enable_fast_slicing()
