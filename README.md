ahlive - animate your data to life!
====================================

Install the package: ``pip install ahlive``

Full Documentation: http://ahlive.readthedocs.io/

ahlive is an open-source Python package that makes animating data simple, clean, and enjoyable.

It can be as easy as:

```python
import ahlive as ah
df = ah.open_dataset(
    "owid_co2_concentrations_over_the_long_term_scripps",
    names=["entity", "year", "co2"]
)
ah.DataFrame(df, xs="year", ys="co2").render()
```

Here are some features that make ahlive stand out!

- inline labels that follow the data
- dynamic axes limits that expand as necessary
- remarks that pause the animation when a threshold is met
- moving average reference line
- straightforward usage; just set keywords!

![CO2 Concentrations](https://raw.githubusercontent.com/ahuang11/ahlive/main/docs/source/_static/co2_concentrations.gif)
The code to generate this example can be found [here](https://ahlive.readthedocs.io/en/latest/introductions/quick_start.html).

Need support? Join the community and ask a question at the [discussions](https://github.com/ahuang11/ahlive/discussions) page. Don't be shy--it would make my day to see others use my package, seriously! (And I personally would love to )

And if you like the project, don't forget to star the project!
