.. currentmodule:: ahlive

Changelog
==========

v0.1.0 (?)
----------

new features
~~~~~~~~~~~~

- Added two new merge functions: `slide` and `stagger`. Also, implemented merge methods and refactored the merge functions internally (:pull:`56`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `ascending` configuration to `race` preset (:pull:`58`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `tiles`, `zoom`, and allow passing cartopy.crs + cartopy.feature instances to geographic params (:pull:`62`).
  By `Andrew Huang <https://github.com/ahuang11>`_

documentation
~~~~~~~~~~~~~

- Tidy up documentation and fix a bug in Overview regarding `s=0` (:pull:`46`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix documentation bugs by adding `pooch` for `xr.tutorial` and add missing documentation from PR 56 (:pull:`57`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix documentation bugs by unpinning `cartopy=0.18.0` and adding a default for `tiles_kwds.pop("zoom")` (:pull:`63`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix documentation crashes (:pull:`65`).
  By `Andrew Huang <https://github.com/ahuang11>`_

enhancements
~~~~~~~~~~~~

- `invert` is now more polished and documented, accepting label, group, and state_labels. (:pull:`44`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Cast labels input to str. (:pull:`60`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Default `coastline` only if `crs` or `projection` is set, but not any other geo features (:pull:`67`).
  By `Andrew Huang <https://github.com/ahuang11>`_

bug fixes
~~~~~~~~~

- Fixed error when rendering `ah.Dataset` with `ah.DataFrame` and non-numeric columns in tutorial (:pull:`54`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fixed bug with `ah.Array2D` when input arrays only had two dimensions (:pull:`58`).

- Loosen restrictions on `inline_locs` in `reference` method when both `x0s` and `y0s` are passed (:pull:`66`).
  By `Andrew Huang <https://github.com/ahuang11>`_

internal changes
~~~~~~~~~~~~~~~~

- `label` now accepts integer and float values. (:pull:`45`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Suppress warning from `util.fillna` (:pull:`47`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Clean up and simplify internals (:pull:`64`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- R-stripped limit data variable (xlim0s now xlim0) for consistency (:pull:`66`).
  By `Andrew Huang <https://github.com/ahuang11>`_

v0.0.3 (15 February 2021)
-------------------------

documentation
~~~~~~~~~~~~~

- Make version shown and copyright year on docs dynamic (:pull:`34`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Updated untriggered remark in intro and added new example (:pull:`40`).
  By `Andrew Huang <https://github.com/ahuang11>`_


bug fixes
~~~~~~~~~

- Fixed datetime / timedelta formatting (:pull:`37`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fixed `show` dynamic default so `pygifsicle` works (:pull:`39`).
  By `Andrew Huang <https://github.com/ahuang11>`_

v0.0.2 (14 February 2021)
-------------------------

new features
~~~~~~~~~~~~

- `progress` is added to allow ability to toggle showing of progress bar. (:pull:`30`)
  By `Andrew Huang <https://github.com/ahuang11>`_

enhancements
~~~~~~~~~~~~

- `save` now accepts pathlib.Path. (:pull:`29`).
  By `Andrew Huang <https://github.com/ahuang11>`_
- Note that `pygifsicle` warning can be disabled by setting `pygifsicle=False` (:pull:`31`).
  By `Andrew Huang <https://github.com/ahuang11>`_

internal changes
~~~~~~~~~~~~~~~~

- Default of `show` replaced from `True` to `None` for non-IPython users. (:pull:`28`).
  By `Andrew Huang <https://github.com/ahuang11>`_

bug fixes
~~~~~~~~~

- Replaced internal `util.is_subdtype` with `pandas.api.types` to make type checking more robust across numpy versions (:pull:`25`).
  By `Andrew Huang <https://github.com/ahuang11>`_

documentation
~~~~~~~~~~~~~

- Added `xskillscore` to acknowledgements and added CHANGELOG.rst plus HOWTORELEASE.rst to documentation (:pull:`26`).
  By `Andrew Huang <https://github.com/ahuang11>`_.
- Fixed external links to GitHub (:pull:`27`).
  By `Andrew Huang <https://github.com/ahuang11>`_.
- Added group labels documentation (:pull:`32`).
  By `Andrew Huang <https://github.com/ahuang11>`_.


v0.0.1 (10 February 2021)
-------------------------

- Initial release!
