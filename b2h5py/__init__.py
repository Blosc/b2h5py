"""Transparent optimized reading of n-dimensional Blosc2 slices for h5py.

Optimizations are applied to slices of the form ``dataset[...]`` or
``dataset.__getitem__(...)`` with step 1 on Blosc2-compressed datasets using
the native byte order.

They are enabled automatically on module import, by monkey-patching the
``h5py.Dataset`` class.  You may explicitly undo this patching with
`unpatch_dataset_class()` and redo it with `patch_dataset_class()`.  You may
also patch the class temporarily using `patching_dataset_class()` to get a
context manager.

You may force-disable the optimization at any time by setting
``BLOSC2_FILTER=1`` in the environment.
"""

from .blosc2 import (is_dataset_class_patched,
                     patch_dataset_class,
                     patching_dataset_class,
                     unpatch_dataset_class)


__all__ = ['is_dataset_class_patched',
           'patch_dataset_class',
           'patching_dataset_class',
           'unpatch_dataset_class']


patch_dataset_class()
