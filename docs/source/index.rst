ahlive - animate your data to life!
====================================

ahlive is an open-source Python package that makes animating data simple, clean, and enjoyable!

It can be as easy as:

.. code-block:: python

   import ahlive as ah
   df = ah.open_dataset(
       "owid_co2_concentrations_over_the_long_term_scripps",
       names=["entity", "year", "co2"]
   )
   ah.DataFrame(df, xs="year", ys="co2").render()

Install the package:
   ``pip install ahlive``

View the repository:
   https://github.com/ahuang11/ahlive/

Ask a question:
   https://github.com/ahuang11/ahlive/discussions

Report a bug or request a feature:
   https://github.com/ahuang11/ahlive/issues

Help me develop:
   https://ahlive.readthedocs.io/en/latest/contributing.html

.. toctree::
   :maxdepth: 1
   :caption: INTRODUCTIONS

   introductions/quick_start.ipynb
   introductions/cheat_sheet.ipynb
   introductions/why_animate.ipynb
   introductions/about.ipynb

.. toctree::
   :maxdepth: 1
   :caption: ESSENTIALS

   essentials/fetching.ipynb
   essentials/serializing.ipynb
   essentials/charting.ipynb
   essentials/merging.ipynb
   essentials/labeling.ipynb
   essentials/remarking.ipynb
   essentials/referencing.ipynb
   essentials/mapping.ipynb
   essentials/exporting.ipynb

.. toctree::
   :maxdepth: 1
   :caption: CUSTOMIZATIONS

   customizations/animating.ipynb
   customizations/bounding.ipynb
   customizations/interpolating.ipynb
   customizations/configuring.ipynb
   customizations/presetting.ipynb
   customizations/transforming.ipynb

.. toctree::
   :maxdepth: 1
   :caption: EXAMPLES

   examples/gapminder.ipynb
   examples/rain_simulation.ipynb
   examples/hurricane_tracks.ipynb
   examples/hurricane_intensification.ipynb
   examples/nuclear_weapons.ipynb
   examples/gridded_weather.ipynb

.. toctree::
   :maxdepth: 1
   :caption: REFERENCES

   contributing
   release
   changelog
