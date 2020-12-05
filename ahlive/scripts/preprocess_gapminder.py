import os
import pandas as pd
from bokeh.sampledata import gapminder

from ahlive.tutorial import DATA_DIR


fertility = gapminder.fertility.reset_index().melt(
    id_vars="Country", var_name="Year", value_name="Fertility"
)
population = gapminder.population.reset_index().melt(
    id_vars="Country", var_name="Year", value_name="Population"
)
life_expectancy = gapminder.life_expectancy.reset_index().melt(
    id_vars="Country", var_name="Year", value_name="Life Expectancy"
)
df = pd.merge(
    pd.merge(pd.merge(fertility, population), life_expectancy),
    gapminder.regions,
    on="Country",
)
df["Year"] = df["Year"].astype(int)
df.columns = df.columns.str.lower()
df.index = range(len(df))
df.to_pickle(os.path.join(DATA_DIR, "gapminder.pkl"))
