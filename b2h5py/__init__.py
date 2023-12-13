"""Transparent optimized reading of n-dimensional Blosc2 slices for h5py.

Optimizations are applied to slices of the form ``dataset[...]`` or
``dataset.__getitem__(...)`` with step 1 on Blosc2-compressed datasets using
the native byte order.  They are implemented by monkey-patching the
``h5py.Dataset`` class.

Optimizations need to be enabled explicitly.  One option is to call
`enable_fast_slicing()` to enable them globally (by performing the patching).
Then `disable_fast_slicing()` may be called to disable them again (by undoing
the patching).  As an alternative, you may also activate optimizations
temporarily using `fast_slicing()` to get a context manager.

**Note:** For testing and debugging purposes, you may force-disable
optimizations at any time by setting ``BLOSC2_FILTER=1`` in the environment.
"""

from .patch import (disable_fast_slicing,
                    enable_fast_slicing,
                    fast_slicing,
                    is_fast_slicing_enabled)


__all__ = ['disable_fast_slicing',
           'enable_fast_slicing',
           'fast_slicing',
           'is_fast_slicing_enabled']
