.. currentmodule:: ahlive

Changelog
==========

v1.0.4
------

bug fixes
~~~~~~~~~

- Fix deprecated `fps`
  By `Andrew Huang <https://github.com/ahuang11>`_

documentation
~~~~~~~~~~~~~

- Fix errors in docs
  By `Andrew Huang <https://github.com/ahuang11>`_

v1.0.3
------

bug fixes
~~~~~~~~~

- Set `tqdm` to None if not installed (:pull:`149`)
- Import `Iterable` from `collections.abc` (:pull:`149`)
  By `Andrew Huang <https://github.com/ahuang11>`_

v1.0.1 and v1.0.2
-----------------

documentation
~~~~~~~~~~~~~

- Fix code snippet on index.rst (:pull:`137`).
  By `Andrew Huang <https://github.com/ahuang11>`_

v1.0.0
-------

new features
~~~~~~~~~~~~

- Added two new merge functions: `slide` and `stagger`. Also, implemented merge methods and refactored the merge functions internally (:pull:`56`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `ascending` configuration to `race` preset (:pull:`58`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `tiles`, `zoom`, and allow passing cartopy.crs + cartopy.feature instances to geographic params (:pull:`62`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `morph` preset (:pull:`76`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Refactored `bar` and `barh` charts, removing `series` as a preset, but using it as the default, and added `stacked`, `morph`, and `morph_stacked` presets for bar charts. (:pull:`80`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Added `morph_trail` and updated internals (:pull:`86`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- `state_labels` can now be appended/prepended to title labels and documented prefix/suffix/units/replacements (:pull:`90`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Added support for `pie` charts (:pull:`97`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Added new charts including errorbar, area, annotation, hexbin, quiver, streamplot (:pull:`99`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Expanded preset support for various charts (:pull:`100`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Capability to set defaults (:pull:`101`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Added OWID datasets for use (:pull:`107`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Upgrade `open_dataset` and add `list_datasets` (:pull:`109`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Added `last` for `reference`, and `labels` and `**other_vars`for `remark` (:pull:`112`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Support `adjust_text` keyword that attempts to prevent text overlaps (:pull:`126`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `windbarb` chart and cleanup internals (:pull:`131`).
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

- Re-order params by setting precedences (:pull:`70`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Documented presets and defaults (:pull:`91`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix documentation and rename `set_defaults` to `config_defaults` (:pull:`101`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add missing docs and polish defaults (:pull:`113`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `why_animate` docs (:pull:`130`).
  By `Andrew Huang <https://github.com/ahuang11>`_

enhancements
~~~~~~~~~~~~

- `invert` is now more polished and documented, accepting label, group, and state_labels. (:pull:`44`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Cast labels input to str. (:pull:`60`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Default `coastline` only if `crs` or `projection` is set, but not any other geo features (:pull:`67`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Improve `print` of objects by showing modified configurations (:pull:`92`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Remove temp file (:pull:`96`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Stylize plots for less chart junk (:pull:`102`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Improve setting margins and limits (:pull:110`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add `pattern` and `sample` as keywords to `list_datasets` and fix docs (:pull:`115`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add dummy frame so Twitter doesn't automatically restart GIF at the end (:pull:`132`).
  By `Andrew Huang <https://github.com/ahuang11>`_

bug fixes
~~~~~~~~~

- Fixed error when rendering `ah.Dataset` with `ah.DataFrame` and non-numeric columns in tutorial (:pull:`54`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fixed bug with `ah.Array2D` when input arrays only had two dimensions (:pull:`58`).

- Loosen restrictions on `inline_locs` in `reference` method when both `x0s` and `y0s` are passed (:pull:`66`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Various bug fixes, primarily with simultaneous usage of `c` and `color` plus `preset=trail` (:pull:`75`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Move `_config_chart` before `_precompute_base` to fix automated preset text format (:pull:`76`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix various issues: formatting, ease of use, use of `group` instead of `label` for `morph` preset, negative bar values, compressing vars, bar width, refactor of remarks, and interpolation of bar labels. (:pull:`77`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix `remark` bugs and lower memory consumption in docs. (:pull:`78`)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix use of `reference` and `remark` with bar `morph`. (:pull:`79`)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Do not require cartopy to be installed to use. (:pull:`93`)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix bar charts for 1 item. (:pull:`94`)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix grid charts, removing warnings and updating to valid keys. (:pull:`95`)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Add missing charting notebook to docs and fix various bugs in presetting. (:pull:`98`)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix poor logic in tqdm missing warning (:pull:`123`:)
  By `Andrew Huang <https://github.com/ahuang11>`_

- Revise method for popping invalid keys for a given chart method (:pull:`129`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fix geographic remarks to show up (:pull:`135`).
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

- Refactor class inheritance (:pull:`71`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Replace dask with concurrent.futures and have tqdm required (:pull:`120`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Improved maintainability of code and remove ability to set multiple `ease` and `interp` for a single subplot (:pull:`128`).
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
