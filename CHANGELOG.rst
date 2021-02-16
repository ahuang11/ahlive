.. currentmodule:: ahlive

Changelog
==========

v0.0.3 (15 February 2021)
-------------------------

documentation
~~~~~~~~~~~~~

- Make version shown and copyright year on docs dynamic (:pull:`34`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Updated untriggered remark in intro and added new example (:pull:`39`).
  By `Andrew Huang <https://github.com/ahuang11>`_


bug fixes
~~~~~~~~~

- Fixed datetime / timedelta formatting (:pull:`37`).
  By `Andrew Huang <https://github.com/ahuang11>`_

- Fixed `show` dynamic default so `pygifsicle` works (:pull:`38`).
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
