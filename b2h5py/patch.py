"""Support for patching the ``h5py.Dataset`` class.

Use `enable_fast_slicing()` to patch the class globally,
`disable_fast_slicing()` to unpatch it, and `is_fast_slicing_enabled()` to
tell whether the class is patched or not.  Call `fast_slicing()` to get a
context manager that patches the class temporarily.
"""

import contextlib
import functools

import h5py

from h5py._hl.base import cached_property as h5cached_property, phil as h5phil

from . import blosc2 as b2


# ``B2Dataset_*`` functions will be monkey-patched
# into the ``h5py.Dataset`` class.

@h5cached_property
def B2Dataset_opt_dataset_ok(self):
    """Is this dataset suitable for Blosc2 optimized slicing"""
    return b2.opt_slicing_dataset_ok(self)
# Fixing observable function name as it is used to cache the result.
B2Dataset_opt_dataset_ok.func.__name__ = b2.opt_dataset_ok_prop


def B2Dataset___getitem__(self, args, new_dtype=None):
    args = args if isinstance(args, tuple) else (args,)

    with h5phil:
        try:
            return b2.opt_slice_read(self, args, new_dtype)
        except b2.NoOptSlicingError:
            pass  # No Blosc2 optimized slicing, try other approaches

    # ``__wrapped__`` is set by ``functools.update_wrapper()`` below.
    return B2Dataset___getitem__.__wrapped__(self, args, new_dtype)


def is_fast_slicing_enabled():
    """Return whether global support for Blosc2 optimized slicing is
    activated.

    This means checking whether``h5py.Dataset`` is already patched for Blosc2
    optimizations.
    """
    return hasattr(h5py.Dataset, b2.opt_dataset_ok_prop)


def enable_fast_slicing():
    """Globally activate support for Blosc2 optimized slicing.

    This means patching ``h5py.Dataset`` to support Blosc2 optimizations.  It
    has no effect if the class has already been patched for this purpose.

    This supports patching the class if it has already been patched by other
    code for other purposes.
    """
    if is_fast_slicing_enabled():
        return  # already patched

    setattr(h5py.Dataset, b2.opt_dataset_ok_prop, B2Dataset_opt_dataset_ok)
    # Update the wrapper in the last moment,
    # to work correctly in case the function was already monkey-patched
    # by someone else after importing this module.
    functools.update_wrapper(B2Dataset___getitem__, h5py.Dataset.__getitem__)
    h5py.Dataset.__getitem__ = B2Dataset___getitem__


def disable_fast_slicing():
    """Globally deactivate support for Blosc2 optimized slicing.

    This means undoing the patching of ``h5py.Dataset`` to remove support for
    Blosc2 optimizations.  It has no effect if the class has not been patched
    for this purpose.

    Raises `ValueError` if the operations patched by this code were already
    patched over by some other code.  In this case, the latter patch must be
    removed first (if the other code supports it).
    """
    if not is_fast_slicing_enabled():
        return  # not patched

    if h5py.Dataset.__getitem__ is not B2Dataset___getitem__:
        # To support this, we would need to
        # go down the chain of ``__wrapped__`` attributes
        # and alter them in place, which feels quite dangerous.
        raise ValueError("dataset class was patched over by someone else")
    h5py.Dataset.__getitem__ = h5py.Dataset.__getitem__.__wrapped__
    delattr(h5py.Dataset, b2.opt_dataset_ok_prop)


@contextlib.contextmanager
def fast_slicing():
    """Get a context manager to temporarily activate support for Blosc2
    optimized slicing.

    This means patching ``h5py.Dataset`` temporarily.  If the class was
    already patched when the context manager is entered, it remains patched on
    exit.  Otherwise, it is unpatched.

    Note: this change is applied globally while the context manager is active.
    """
    already_patched = is_fast_slicing_enabled()

    if already_patched:  # do nothing
        yield None
        return

    enable_fast_slicing()
    try:
        yield None
    finally:
        disable_fast_slicing()
