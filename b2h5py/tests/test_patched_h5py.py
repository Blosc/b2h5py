"""Run h5py unit tests with patched dataset class.

This module does not provide test cases, but it calls the h5py tests directly.
You may run it with ``python -m b2h5py.tests.test_patched_h5py``.

This module has additional dependencies.  You may want to install this
package's ``h5py-test`` extra.
"""

import os
import unittest

import b2h5py
import h5py
import h5py.tests


def run_h5py_tests():
    """Run h5py unit tests with patched dataset class"""
    test_suite = unittest.defaultTestLoader.discover(
        os.path.dirname(h5py.tests.__file__),
        top_level_dir=os.path.dirname(os.path.dirname(h5py.__file__)))
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)


if __name__ == '__main__':
    run_h5py_tests()
