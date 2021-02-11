import pandas as pd
import param

from .configuration import ITEMS


class TutorialData(param.Parameterized):

    label = param.ObjectSelector(objects=ITEMS["datasets"])

    _source = None
    _base_url = None
    _data_url = None

    def _load_annual_co2(self, raw, **kwds):
        self._source = "NOAA ESRL"
        self._base_url = "https://www.esrl.noaa.gov/"
        self._data_url = (
            "https://www.esrl.noaa.gov/"
            "gmd/webdata/ccgg/trends/co2/co2_annmean_mlo.txt"
        )
        df = pd.read_csv(
            self._data_url,
            header=None,
            comment="#",
            sep="\s+",  # noqa
            names=["year", "co2_ppm", "uncertainty"],
            **kwds,
        )
        return df

    def _load_tc_tracks(self, raw, **kwds):
        self._source = "IBTrACS v04 - USA"
        self._base_url = "https://www.ncdc.noaa.gov/ibtracs/"
        self._data_url = (
            "https://www.ncei.noaa.gov/data/"
            "international-best-track-archive-for-climate-stewardship-ibtracs/"
            "v04r00/access/csv/ibtracs.last3years.list.v04r00.csv"
        )
        if raw:
            return pd.read_csv(self._data_url, keep_default_na=False, **kwds)
        cols = [
            "BASIN",
            "NAME",
            "LAT",
            "LON",
            "ISO_TIME",
            "USA_WIND",
            "USA_PRES",
            "USA_SSHS",
            "USA_RMW",
        ]
        df = pd.read_csv(self._data_url, keep_default_na=False, usecols=cols, **kwds)
        df.columns = df.columns.str.lower()
        df = df.iloc[1:]
        df = df.set_index("iso_time")
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors="ignore")
        return df

    def _load_covid19_us_cases(self, raw, **kwds):
        self._source = "JHU CSSE COVID-19"
        self._base_url = "https://github.com/CSSEGISandData/COVID-19"
        self._data_url = (
            "https://github.com/CSSEGISandData/COVID-19/raw/master/"
            "csse_covid_19_data/csse_covid_19_time_series/"
            "time_series_covid19_confirmed_US.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df = df.drop(
            ["UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Country_Region"],
            axis=1,
        )
        df.columns = df.columns.str.lower().str.rstrip("_")
        df = df.melt(
            id_vars=["lat", "long", "combined_key", "province_state"],
            var_name="date",
            value_name="cases",
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_covid19_global_cases(self, raw, **kwds):
        self._source = "JHU CSSE COVID-19"
        self._base_url = "https://github.com/CSSEGISandData/COVID-19"
        self._data_url = (
            "https://github.com/CSSEGISandData/COVID-19/raw/master/"
            "csse_covid_19_data/csse_covid_19_time_series/"
            "time_series_covid19_confirmed_global.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df.columns = df.columns.str.lower().str.rstrip("_")
        df = df.melt(
            id_vars=["province/state", "country/region", "lat", "long"],
            var_name="date",
            value_name="cases",
        )
        df.columns = df.columns.str.replace("/", "_")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_covid19_population(self, raw, **kwds):
        self._source = "JHU CSSE COVID-19"
        self._base_url = "https://github.com/CSSEGISandData/COVID-19"
        self._data_url = (
            "https://raw.githubusercontent.com/"
            "CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df.columns = df.columns.str.lower().str.rstrip("_")
        return df

    def _load_gapminder_life_expectancy(self, raw, **kwds):
        self._source = "World Bank Gapminder"
        self._base_url = (
            "https://github.com/open-numbers/ddf--gapminder--systema_globalis"
        )
        self._data_url = (
            "https://raw.githubusercontent.com/open-numbers/"
            "ddf--gapminder--systema_globalis/master/"
            "countries-etc-datapoints/ddf--datapoints--"
            "life_expectancy_years--by--geo--time.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df = df.rename(columns={"life_expectancy_years": "life_expectancy"})
        return df

    def _load_gapminder_income(self, raw, **kwds):
        self._source = "World Bank Gapminder"
        self._base_url = (
            "https://github.com/open-numbers/ddf--gapminder--systema_globalis"
        )
        self._data_url = (
            "https://raw.githubusercontent.com/open-numbers/"
            "ddf--gapminder--systema_globalis/master/"
            "countries-etc-datapoints/ddf--datapoints--"
            "income_per_person_gdppercapita_ppp_inflation_adjusted"
            "--by--geo--time.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df = df.rename(
            columns={
                "income_per_person_gdppercapita_ppp_inflation_adjusted": "income"  # noqa
            }
        )
        return df

    def _load_gapminder_population(self, raw, **kwds):
        self._source = "World Bank Gapminder"
        self._base_url = (
            "https://github.com/open-numbers/ddf--gapminder--systema_globalis"
        )
        self._data_url = (
            "https://raw.githubusercontent.com/open-numbers/"
            "ddf--gapminder--systema_globalis/master/"
            "countries-etc-datapoints/ddf--datapoints--"
            "population_total--by--geo--time.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df = df.rename(columns={"population_total": "population"})
        return df

    def _load_gapminder_country(self, raw, **kwds):
        self._source = "World Bank Gapminder"
        self._base_url = (
            "https://github.com/open-numbers/ddf--gapminder--systema_globalis"
        )
        self._data_url = (
            "https://raw.githubusercontent.com/open-numbers/"
            "ddf--gapminder--systema_globalis/master/"
            "ddf--entities--geo--country.csv"
        )
        df = pd.read_csv(self._data_url, **kwds)
        if raw:
            return df
        df = df[["country", "name", "world_6region"]].rename(
            columns={"world_6region": "region"}
        )
        df["region"] = df["region"].str.replace("_", " ").str.title()
        return df

    def open_dataset(self, raw, verbose, **kwds):
        data = getattr(self, f"_load_{self.label}")(raw=raw, **kwds)
        label = self.label.replace("_", " ").upper()
        attr = f"{label} | Source: {self._source} | {self._base_url}"
        if verbose:
            attr = f"{attr}\nData: {self._data_url}"
        print(attr)
        return data


def open_dataset(label=None, raw=False, verbose=False, **kwds):
    return TutorialData(label=label).open_dataset(raw, verbose, **kwds)
