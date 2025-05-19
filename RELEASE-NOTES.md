# Release notes

## Changes from 0.5.0 to 0.5.1

* Fixes a bug in 1-dim arrays whose typesize is larger than 1.

## Changes from 0.4.1 to 0.5.0

* Workaround for recognizing 1-dim arrays in HDF5. For some reason,
  HDF5-Blosc2 does not add the dim info in cd_values when ndim == 1.
  This adds the dim info manually.

* Upload/download artifact in CI bumped to v4.

* Development status has been bumped to 4 - Beta.
