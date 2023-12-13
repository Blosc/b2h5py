"""Automatic activation of Blosc2 optimized slicing for h5py.

Importing this module enables the optimization globally, just use::

    import b2h5py.auto

After that, all slicing operations on Blosc2-compressed datasets will be
transparently optimized when possible.
"""

from .patch import enable_fast_slicing


enable_fast_slicing()
