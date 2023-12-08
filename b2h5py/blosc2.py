"""Implements support for Blosc2 optimized slicing.

Please note that for a selection over a dataset to be suitable for Blosc2
optimized slicing, besides being amenable to fast reading, (i) such slicing
must be enabled globally (`opt_slicing_enabled()`), (ii) the dataset must be
amenable to it (`opt_slicing_dataset_ok()`), and (iii) the selection must be
amenable to it (`opt_slicing_selection_ok()`).

If these conditions have already been checked for a given dataset,
`opt_selection_read()` may be used.

If a dataset is adapted for Blosc2 optimized slicing (e.g. by having its class
monkey-patched), `opt_slice_read()` shoud suffice, as it takes care of the
checks.
"""

import contextlib
import functools
import os
import platform

import h5py
import hdf5plugin
import numpy

from blosc2.schunk import open as b2schunk_open
from h5py._hl import selections as h5sel
from h5py._hl.base import cached_property as h5cached_property, phil as h5phil


class NoOptSlicingError(TypeError):
    """Blosc2 optimized slicing is not possible."""
    pass


# For testing whether optimizations are used or not,
# replace this with a class not derived from `NoOptSlicingError`;
# it will be raised by slicing operations if optimizations are not used.
# Remember to restore the original value when you are done.
_no_opt_error = NoOptSlicingError


def opt_slicing_selection_ok(selection):
    """Is the given selection suitable for Blosc2 optimized slicing?"""
    return (
        isinstance(selection, h5sel.SimpleSelection)
        and numpy.prod(selection._sel[2]) == 1  # all steps equal 1
    )


def opt_slicing_dataset_ok(dataset):
    """Is the given dataset suitable for Blosc2 optimized slicing?

    It is assumed that the dataset is also ok for fast reading.  The result
    may be cached.
    """
    return (
        dataset.chunks is not None
        # '.compression' and '.compression_opts' don't work with plugins:
        # <https://forum.hdfgroup.org/t/registering-custom-filter-issues/9239>
        and '32026' in dataset._filters  # Blosc2's ID
        and dataset.dtype.isnative
        and (dataset.file.mode == 'r'
             or platform.system().lower() != 'windows')
    )


def opt_slicing_enabled():
    """Is Blosc2 optimized slicing not disabled via the environment?

    This returns false if the BLOSC2_FILTER environment variable is set to a
    non-zero integer (which forces the use of the HDF5 filter pipeline).
    """
    try:
        force_filter = int(os.environ.get('BLOSC2_FILTER', '0'), 10)
    except ValueError:
        force_filter = 0
    return force_filter == 0


def _read_chunk_slice(path, offset, slice_, dtype):
    schunk = b2schunk_open(path, mode='r', offset=offset)
    s = schunk[slice_]
    if s.dtype.kind != 'V':
        return s
    # hdf5-blosc2 always uses an opaque dtype, convert the array
    # (the wrapping below does not copy the data anyway).
    return numpy.ndarray(s.shape, dtype=dtype, buffer=s.data)


def opt_selection_read(dataset, selection, new_dtype=None):
    """Read the specified selection from the given dataset.

    Blosc2 optimized slice reading is used, but the caller must make sure
    beforehand that both the dataset and the selection are suitable for such
    operation.

    A NumPy array is returned with the desired slice.  The array will have the
    given new dtype if specified.
    """
    slice_start = selection._sel[0]
    slice_shape = selection.mshape
    slice_ = tuple(slice(st, st + sh)
                   for (st, sh) in zip(slice_start, slice_shape))
    slice_arr = numpy.empty(dtype=new_dtype or dataset.dtype,
                            shape=slice_shape)
    if 0 in slice_shape:  # empty slice
        return slice_arr.reshape(selection.array_shape)

    # TODO: consider using 'dataset.id.get_chunk_info' for performance
    get_chunk_info = dataset.id.get_chunk_info_by_coord
    for chunk_slice in dataset.iter_chunks(slice_):
        # TODO: Remove when h5py#2341 is fixed.
        if any(s.stop <= s.start for s in chunk_slice):
            continue  # bogus iter_chunks item, see h5py#2341

        # Compute different parameters for the slice/chunk combination.
        (
            slice_as_chunk_slice,
            chunk_as_slice_slice,
            chunk_slice_start,
        ) = tuple(zip(*(
            (  # nth value below gets added to nth tuple above
                slice(csl.start % csh, ((csl.start % csh)
                                        + (csl.stop - csl.start))),
                slice(csl.start - sst, csl.stop - sst),
                csl.start,
            )
            for (csl, csh, sst)
            in zip(chunk_slice, dataset.chunks, slice_start)
        )))

        # Get the part of the slice that overlaps the current chunk.
        chunk_info = get_chunk_info(chunk_slice_start)
        chunk_slice_arr = _read_chunk_slice(
            dataset.file.filename, chunk_info.byte_offset,
            slice_as_chunk_slice, dataset.dtype)
        if (
                chunk_slice_arr.dtype != dataset.dtype
                or len(chunk_slice_arr.shape) != len(slice_shape)
                or chunk_slice_arr.shape > slice_shape
        ):
            # The data in the Blosc2 super-chunk is bogus.
            raise RuntimeError(
                f"Invalid shape/dtype of "
                f"chunk covering coordinate {chunk_slice_start} "
                f"(offset {chunk_info.byte_offset}): "
                f"expected <= {slice_shape}/{dataset.dtype}, "
                f"got {chunk_slice_arr.shape}/{chunk_slice_arr.dtype}")

        # Place the part in the final slice.
        slice_arr[chunk_as_slice_slice] = chunk_slice_arr

    # Adjust result dimensions to those dictated by the input selection.
    ret_shape = selection.array_shape
    if ret_shape == ():  # scalar result
        return slice_arr[()]
    return slice_arr.reshape(ret_shape)


def opt_slice_read(dataset, slice_, new_dtype=None):
    """Read the specified slice from the given dataset.

    The dataset must support a ``_blosc2_opt_slicing_ok`` property that calls
    `opt_slicing_dataset_ok()`.

    Blosc2 optimized slice reading is used if available and suitable,
    otherwise a `NoOptSlicingError` is raised.

    A NumPy array is returned with the desired slice.  The array will have the
    given new dtype if specified.
    """
    # In the following checks,
    # if `_no_opt_error` is not derived from `NoOptSlicingError`
    # the get item operation shall not be able to catch the exception.

    if not dataset._blosc2_opt_slicing_ok:
        raise _no_opt_error(
            "Dataset is not suitable for Blosc2 optimized slicing")

    if not opt_slicing_enabled():
        raise _no_opt_error(
            "Blosc2 optimized slicing is unavailable or disabled")

    selection = h5sel.select(dataset.shape, slice_, dataset=dataset)
    if not opt_slicing_selection_ok(selection):
        raise _no_opt_error(
            "Selection is not suitable for Blosc2 optimized slicing")

    return opt_selection_read(dataset, selection, new_dtype)


# ``B2Dataset_*`` functions will be monkey-patched
# into the ``h5py.Dataset`` class.

@h5cached_property
def B2Dataset__blosc2_opt_slicing_ok(self):
    """Is this dataset suitable for Blosc2 optimized slicing"""
    return (
        self._extent_type == h5py.h5s.SIMPLE
        and opt_slicing_dataset_ok(self)
    )


def B2Dataset___getitem__(self, args, new_dtype=None):
    args = args if isinstance(args, tuple) else (args,)

    with h5phil:
        try:
            return opt_slice_read(self, args, new_dtype)
        except NoOptSlicingError:
            pass  # No Blosc2 optimized slicing, try other approaches

    # ``__wrapped__`` is set by ``functools.update_wrapper()`` below.
    return B2Dataset___getitem__.__wrapped__(self, args, new_dtype)


def is_dataset_class_patched():
    """Return whether ``h5py.Dataset`` is already patched for Blosc2
    optimizations.
    """
    return hasattr(h5py.Dataset, '_blosc2_opt_slicing_ok')


def patch_dataset_class():
    """Patch ``h5py.Dataset`` to support Blosc2 optimizations.

    This has no effect if the class has already been patched for this purpose.

    This supports patching the class if it has already been patched by other
    code for other purposes.
    """
    if is_dataset_class_patched():
        return  # already patched

    h5py.Dataset._blosc2_opt_slicing_ok = B2Dataset__blosc2_opt_slicing_ok
    # Update the wrapper in the last moment,
    # to work correctly in case the function was already monkey-patched
    # by someone else after importing this module.
    functools.update_wrapper(B2Dataset___getitem__, h5py.Dataset.__getitem__)
    h5py.Dataset.__getitem__ = B2Dataset___getitem__


def unpatch_dataset_class():
    """Undo the patching of ``h5py.Dataset`` to remove support for Blosc2
    optimizations.

    This has no effect if the class has not been patched for this purpose.

    Raises `ValueError` if the operations patched by this code were already
    patched over by some other code.  In this case, the latter patch must be
    removed first (if the other code supports it).
    """
    if not is_dataset_class_patched():
        return  # not patched

    if h5py.Dataset.__getitem__ is not B2Dataset___getitem__:
        # To support this, we would need to
        # go down the chain of ``__wrapped__`` attributes
        # and alter them in place, which feels quite dangerous.
        raise ValueError("dataset class was patched over by someone else")
    h5py.Dataset.__getitem__ = h5py.Dataset.__getitem__.__wrapped__
    del h5py.Dataset._blosc2_opt_slicing_ok


@contextlib.contextmanager
def patching_dataset_class():
    """Get a context manager to patch ``h5py.Dataset`` temporarily.

    If the class was already patched when the context manager is entered, it
    remains patched on exit.  Otherwise, it is unpatched.

    Note: this change is applied globally while the context manager is active.
    """
    already_patched = is_dataset_class_patched()

    if already_patched:  # do nothing
        yield None
        return

    patch_dataset_class()
    try:
        yield None
    finally:
        unpatch_dataset_class()
