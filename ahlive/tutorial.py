import json
import os
import pickle
import re
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import param

from .configuration import DEFAULTS, OPTIONS


class TutorialData(param.Parameterized):

    label = param.String(allow_None=True)

    _source = None
    _base_url = None
    _data_url = None

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._cache_dir = DEFAULTS["cache_kwds"]["directory"]
        os.makedirs(self._cache_dir, exist_ok=True)
        self._init_owid()

    def _cache_dataset(self, df, cached_path):
        df.to_pickle(cached_path)

    def _read_cached(self, cached_path):
        try:
            if os.path.exists(cached_path):
                with open(cached_path, "wb") as f:
                    return pickle.load(f)
        except Exception:
            os.remove(cached_path)
        return None

    @staticmethod
    def _snake_urlify(s):
        # Replace all hyphens with underscore
        s = s.replace(" - ", "_").replace("-", "_")
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", "", s)
        # Replace all runs of whitespace with a underscore
        s = re.sub(r"\s+", "_", s)
        return s.lower()

    def _init_owid(self):
        owid_labels_path = os.path.join(self._cache_dir, "owid_labels.pkl")

        self._owid_labels = self._read_cached(owid_labels_path)
        if self._owid_labels is not None:
            return

        owid_api_url = (
            "https://api.github.com/"
            "repos/owid/owid-datasets/"
            "git/trees/master?recursive=1"
        )
        with urlopen(owid_api_url) as f:
            sources = json.loads(f.read().decode("utf-8"))

        self._owid_labels = {}
        owid_raw_url = "https://raw.githubusercontent.com/" "owid/owid-datasets/master/"
        for source_tree in sources["tree"]:
            path = source_tree["path"]
            if ".csv" not in path and ".json" not in path:
                continue

            label = "owid_" + self._snake_urlify(path.split("/")[-2].strip())
            if label not in self._owid_labels:
                self._owid_labels[label] = {}

            url = f"{owid_raw_url}/{quote(path)}"
            if ".csv" in path:
                self._owid_labels[label]["data"] = url
            elif ".json" in path:
                self._owid_labels[label]["meta"] = url

        with open(owid_labels_path, "wb") as f:
            pickle.dump(self._owid_labels, f)

    def _load_owid(self, raw, **kwds):
        self._data_url = self._owid_labels[self.label]["data"]
        meta_url = self._owid_labels[self.label]["meta"]
        df = pd.read_csv(self._data_url, **kwds)
        with urlopen(meta_url) as response:
            meta = json.loads(response.read().decode())
        self._source = (
            " & ".join(source["dataPublishedBy"] for source in meta["sources"])
            + " curated by Our World in Data (OWID)"
        )
        self._base_url = (
            " & ".join(source["link"] for source in meta["sources"])
            + " through https://github.com/owid/owid-datasets"
        )
        if raw:
            return df

        df.columns = [self._snake_urlify(col) for col in df.columns]
        return df

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
        numeric_cols = ["lat", "lon", "usa_rmw", "usa_pres", "usa_sshs", "usa_rmw"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
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

    def _load_iem_asos(
        self,
        raw,
        ini="2020-01-01",
        end="2020-12-31",
        stn="CMI",
        tz="utc",
        data="all",
        latlon="no",
        elev="no",
        **kwds,
    ):
        stn = stn.upper()

        if isinstance(data, list):
            data = ",".join(data)

        tzs = {
            "utc": "Etc/UTC",
            "akst": "America/Anchorage",
            "wst": "America/Los_Angeles",
            "mst": "America/Denver",
            "cst": "America/Chicago",
            "est": "America/New_York",
        }
        tz = tzs.get(tz, tz)
        if tz not in tzs.values():
            raise ValueError(f"tz must be one of the following: {tzs}")

        ini_dt = pd.to_datetime(ini)
        end_dt = pd.to_datetime(end)

        self._source = "Iowa Environment Mesonet ASOS"
        self._base_url = "https://mesonet.agron.iastate.edu/ASOS/"
        self._data_url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
            f"station={stn}&data={data}&latlon={latlon}&elev={elev}&"
            f"year1={ini_dt:%Y}&month1={ini_dt:%m}&day1={ini_dt:%d}&"
            f"year2={end_dt:%Y}&month2={end_dt:%m}&day2={end_dt:%d}&"
            f"tz={tz}&format=onlycomma&"
            f"missing=empty&trace=empty&"
            f"direct=no&report_type=1&report_type=2"
        )
        df = pd.read_csv(self._data_url, **kwds)

        if raw:
            return df

        df["valid"] = pd.to_datetime(df["valid"])
        df = df.set_index("valid")
        return df

    def open_dataset(self, raw, verbose, **kwds):
        options = "\n".join(OPTIONS["datasets"] + list(self._owid_labels.keys()))
        if self.label is None or self.label not in options:
            raise ValueError(f"Select from one of the following:\n{options}")

        if self.label.startswith("owid_"):
            data = getattr(self, f"_load_owid")(raw, **kwds)
        else:
            data = getattr(self, f"_load_{self.label}")(raw, **kwds)
        label = self.label.replace("_", " ").upper()
        attr = f"{label}\nSource: {self._source}\n{self._base_url}"
        if verbose:
            attr = f"{attr}\nData: {self._data_url}"
        print(attr)
        return data


def open_dataset(label=None, raw=False, verbose=False, **kwds):
    return TutorialData(label=label).open_dataset(raw, verbose, **kwds)
