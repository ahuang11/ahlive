import fnmatch
import inspect
import json
import os
import random
import re
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import param

from .configuration import DEFAULTS, OPTIONS


class TutorialData(param.Parameterized):

    label = param.String(allow_None=True)
    raw = param.Boolean()
    verbose = param.Boolean()
    return_meta = param.Boolean()
    use_cache = param.Boolean()

    _source = None
    _base_url = None
    _data_url = None
    _description = None

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._cache_dir = DEFAULTS["cache_kwds"]["directory"]
        self._remove_href = re.compile(r"<(a|/a).*?>")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._init_owid()

    @property
    def _cache_path(self):
        cache_file = f"{self.label}.pkl"
        return os.path.join(self._cache_dir, cache_file)

    @property
    def _dataset_options(self):
        options = set([])
        for method in dir(self):
            if method.startswith("_load_") and "owid" not in method:
                options.add(method.replace("_load_", ""))
        return list(options) + list(self._owid_labels_df.columns)

    @staticmethod
    def _specify_cache(cache_path, **kwds):
        if kwds:
            cache_ext = (
                "_".join(
                    f"{key}={val}".replace(os.sep, "") for key, val in kwds.items()
                )
                .replace(" ", "_")
                .replace(",", "_")
                .replace("'", "")
            )
            cache_path = f"{os.path.splitext(cache_path)[0]}_{cache_ext}.pkl"
        return cache_path

    def _cache_dataset(self, df, cache_path=None, **kwds):
        if cache_path is None:
            cache_path = self._cache_path
        cache_path = self._specify_cache(cache_path, **kwds)

        df.to_pickle(cache_path)

    def _read_cache(self, cache_path=None, **kwds):
        if not self.use_cache:
            return None

        if cache_path is None:
            cache_path = self._cache_path
        cache_path = self._specify_cache(cache_path, **kwds)

        try:
            return pd.read_pickle(cache_path)
        except Exception:
            if os.path.exists(cache_path):
                os.remove(cache_path)
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
        cache_path = os.path.join(self._cache_dir, "owid_labels.pkl")
        self._owid_labels_df = self._read_cache(cache_path=cache_path)
        if self._owid_labels_df is not None:
            return

        owid_api_url = (
            "https://api.github.com/"
            "repos/owid/owid-datasets/"
            "git/trees/master?recursive=1"
        )
        with urlopen(owid_api_url) as f:
            sources = json.loads(f.read().decode("utf-8"))

        owid_labels = {}
        owid_raw_url = "https://raw.githubusercontent.com/owid/owid-datasets/master/"
        for source_tree in sources["tree"]:
            path = source_tree["path"]
            if ".csv" not in path and ".json" not in path:
                continue

            label = "owid_" + self._snake_urlify(path.split("/")[-2].strip())
            if label not in owid_labels:
                owid_labels[label] = {}

            url = f"{owid_raw_url}/{quote(path)}"
            if ".csv" in path:
                owid_labels[label]["data"] = url
            elif ".json" in path:
                owid_labels[label]["meta"] = url

        self._owid_labels_df = pd.DataFrame(owid_labels)
        self._cache_dataset(self._owid_labels_df, cache_path=cache_path)

    def _load_owid(self, **kwds):
        self._data_url = self._owid_labels_df[self.label]["data"]
        meta_url = self._owid_labels_df[self.label]["meta"]
        with urlopen(meta_url) as response:
            meta = json.loads(response.read().decode())
        self.label = meta["title"]
        self._source = (
            " & ".join(source.get("dataPublishedBy", "") for source in meta["sources"])
            + " curated by Our World in Data (OWID)"
        )
        self._base_url = (
            " & ".join(source["link"] for source in meta["sources"])
            + " through https://github.com/owid/owid-datasets"
        )
        self._description = re.sub(self._remove_href, "", meta["description"])

        df = self._read_cache(**kwds)
        if df is None:
            if "names" in kwds.keys() and "header" not in kwds.keys():
                kwds["header"] = 0
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

        df.columns = [self._snake_urlify(col) for col in df.columns]
        return df

    def _load_annual_co2(self, **kwds):
        self._source = "NOAA ESRL"
        self._base_url = "https://www.esrl.noaa.gov/"
        self._data_url = (
            "https://www.esrl.noaa.gov/"
            "gmd/webdata/ccgg/trends/co2/co2_annmean_mlo.txt"
        )
        self._description = (
            "The carbon dioxide data on Mauna Loa constitute the longest record "
            "of direct measurements of CO2 in the atmosphere. They were started "
            "by C. David Keeling of the Scripps Institution of Oceanography in "
            "March of 1958 at a facility of the National Oceanic and Atmospheric "
            "Administration [Keeling, 1976]. NOAA started its own CO2 measurements "
            "in May of 1974, and they have run in parallel with those made by "
            "Scripps since then [Thoning, 1989]."
        )

        df = self._read_cache(**kwds)
        if df is None:
            base_kwds = dict(
                header=None,
                comment="#",
                sep="\s+",  # noqa
                names=["year", "co2_ppm", "uncertainty"],
            )
            base_kwds.update(kwds)
            df = pd.read_csv(self._data_url, **base_kwds)
            self._cache_dataset(df, **kwds)

        return df

    def _load_tc_tracks(self, **kwds):
        self._source = "IBTrACS v04 - USA"
        self._base_url = "https://www.ncdc.noaa.gov/ibtracs/"
        self._data_url = (
            "https://www.ncei.noaa.gov/data/"
            "international-best-track-archive-for-climate-stewardship-ibtracs/"
            "v04r00/access/csv/ibtracs.last3years.list.v04r00.csv"
        )
        self._description = (
            "The intent of the IBTrACS project is to overcome data availability "
            "issues. This was achieved by working directly with all the Regional "
            "Specialized Meteorological Centers and other international centers "
            "and individuals to create a global best track dataset, merging storm "
            "information from multiple centers into one product and archiving "
            "the data for public use."
        )

        df = self._read_cache(**kwds)
        if df is None:
            base_kwds = dict(keep_default_na=False)
            base_kwds.update(kwds)
            df = pd.read_csv(self._data_url, **base_kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

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
            "USA_STATUS",
            "USA_RECORD",
            "LANDFALL",
        ]
        df = df[cols]
        df.columns = df.columns.str.lower()
        df = df.iloc[1:]
        df = df.set_index("iso_time")
        df.index = pd.to_datetime(df.index)
        numeric_cols = ["lat", "lon", "usa_wind", "usa_pres", "usa_sshs", "usa_rmw"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _load_covid19_us_cases(self, **kwds):
        self._source = "JHU CSSE COVID-19"
        self._base_url = "https://github.com/CSSEGISandData/COVID-19"
        self._data_url = (
            "https://github.com/CSSEGISandData/COVID-19/raw/master/"
            "csse_covid_19_data/csse_covid_19_time_series/"
            "time_series_covid19_confirmed_US.csv"
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
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

    def _load_covid19_global_cases(self, **kwds):
        self._source = "JHU CSSE COVID-19"
        self._base_url = "https://github.com/CSSEGISandData/COVID-19"
        self._data_url = (
            "https://github.com/CSSEGISandData/COVID-19/raw/master/"
            "csse_covid_19_data/csse_covid_19_time_series/"
            "time_series_covid19_confirmed_global.csv"
        )
        self._description = (
            "This is the data repository for the 2019 Novel Coronavirus "
            "Visual Dashboard operated by the Johns Hopkins University Center "
            "for Systems Science and Engineering (JHU CSSE). Also, Supported "
            "by ESRI Living Atlas Team and the Johns Hopkins University "
            "Applied Physics Lab (JHU APL)."
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
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

    def _load_covid19_population(self, **kwds):
        self._source = "JHU CSSE COVID-19"
        self._base_url = "https://github.com/CSSEGISandData/COVID-19"
        self._data_url = (
            "https://raw.githubusercontent.com/"
            "CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"
        )
        self._description = (
            "This is the data repository for the 2019 Novel Coronavirus "
            "Visual Dashboard operated by the Johns Hopkins University Center "
            "for Systems Science and Engineering (JHU CSSE). Also, Supported "
            "by ESRI Living Atlas Team and the Johns Hopkins University "
            "Applied Physics Lab (JHU APL)."
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

        df.columns = df.columns.str.lower().str.rstrip("_")
        return df

    def _load_gapminder_life_expectancy(self, **kwds):
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
        self._description = (
            "This is the main dataset used in tools on the official Gapminder "
            "website. It contains local & global statistics combined from "
            "hundreds of sources."
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

        df = df.rename(columns={"life_expectancy_years": "life_expectancy"})
        return df

    def _load_gapminder_income(self, **kwds):
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
        self._description = (
            "This is the main dataset used in tools on the official Gapminder "
            "website. It contains local & global statistics combined from "
            "hundreds of sources."
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

        df = df.rename(
            columns={
                "income_per_person_gdppercapita_ppp_inflation_adjusted": "income"  # noqa
            }
        )
        return df

    def _load_gapminder_population(self, **kwds):
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
        self._description = (
            "This is the main dataset used in tools on the official Gapminder "
            "website. It contains local & global statistics combined from "
            "hundreds of sources."
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

        df = df.rename(columns={"population_total": "population"})
        return df

    def _load_gapminder_country(self, **kwds):
        self._source = "World Bank Gapminder"
        self._base_url = (
            "https://github.com/open-numbers/ddf--gapminder--systema_globalis"
        )
        self._data_url = (
            "https://raw.githubusercontent.com/open-numbers/"
            "ddf--gapminder--systema_globalis/master/"
            "ddf--entities--geo--country.csv"
        )
        self._description = (
            "This is the main dataset used in tools on the official Gapminder "
            "website. It contains local & global statistics combined from "
            "hundreds of sources."
        )

        df = self._read_cache(**kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **kwds)

        if self.raw:
            return df

        df = df[["country", "name", "world_6region"]].rename(
            columns={"world_6region": "region"}
        )
        df["region"] = df["region"].str.replace("_", " ").str.title()
        return df

    def _load_iem_asos(
        self,
        ini="2020-01-01",
        end="2020-01-03",
        stn="CMI",
        tz="utc",
        data="all",
        latlon="no",
        elev="no",
        **kwds,
    ):
        if isinstance(stn, str):
            stn = [stn]
        stn = "&station=".join(stn)

        if isinstance(data, str):
            data = [data]

        valid_tzs = OPTIONS["iem_tz"]
        tz = valid_tzs.get(tz, tz)
        if tz not in valid_tzs.values():
            raise ValueError(f"tz must be one of the following: {valid_tzs}; got {tz}")

        valid_data = OPTIONS["iem_data"]
        cols = []
        for col in data:
            col = col.strip()
            if col not in valid_data and col != "all":
                raise ValueError(f"data must be a subset of: {valid_data}; got {col}")
            cols.append(col)
        data = "&data=".join(cols)

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
        self._description = (
            "The IEM maintains an ever growing archive of automated airport "
            "weather observations from around the world! These observations "
            "are typically called 'ASOS' or sometimes 'AWOS' sensors. "
            "A more generic term may be METAR data, which is a term that "
            "describes the format the data is transmitted as. If you don't "
            "get data for a request, please feel free to contact us for help. "
            "The IEM also has a one minute interval dataset for US ASOS (2000-) "
            "and Iowa AWOS (1995-2011) sites. This archive simply provides the "
            "as-is collection of historical observations, very little "
            "quality control is done."
        )

        cache_kwds = kwds.copy()
        cache_kwds.update(
            ini=ini, end=end, stn=stn, tz=tz, data=data, latlon=latlon, elev=elev
        )
        df = self._read_cache(**cache_kwds)
        if df is None:
            df = pd.read_csv(self._data_url, **kwds)
            self._cache_dataset(df, **cache_kwds)

        if self.raw:
            return df

        df["valid"] = pd.to_datetime(df["valid"])
        df = df.set_index("valid")
        return df

    def open_dataset(self, **kwds):
        if self.label is None or self.label not in self._dataset_options:
            self.list_datasets()
            raise ValueError("Select a valid dataset listed above")

        if self.label.startswith("owid_"):
            data = getattr(self, "_load_owid")(**kwds)
        else:
            data = getattr(self, f"_load_{self.label}")(**kwds)

        label = self.label.replace("_", " ").upper()
        attr = f"{label}\n\nSource: {self._source}\n{self._base_url}\n"
        if self.verbose:
            attr = (
                f"{attr}\nDescription: {self._description}\n\nData: {self._data_url}\n"
            )

        if self.return_meta:
            meta = {}
            meta["label"] = self.label
            meta["source"] = self._source
            meta["base_url"] = self._base_url
            meta["description"] = self._description
            meta["data_url"] = self._data_url
            return data, meta
        else:
            print(attr)
            return data

    def list_datasets(self, pattern=None, sample=None):
        signatures = {}
        for option in self._dataset_options:
            if "owid" in option:
                signatures[option] = {}
                continue
            signature = inspect.signature(getattr(self, f"_load_{option}"))
            signatures[option] = {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }

        keys = signatures.keys()
        if pattern is not None:
            if "*" not in pattern:
                pattern = f"*{pattern}*"
            keys = [key for key in keys if fnmatch.fnmatch(key, pattern)]

        if sample is not None:
            num_keys = len(keys)
            if num_keys < sample:
                sample = num_keys
            keys = random.sample(keys, sample)

        for key in keys:
            val = signatures[key]
            print(f"- {key}")
            if val:
                print("    adjustable keywords")
                for k, v in val.items():
                    print(f"    {k}: {v}")


def open_dataset(
    label=None, raw=False, verbose=False, return_meta=False, use_cache=True, **kwds
):
    return TutorialData(
        label=label,
        raw=raw,
        verbose=verbose,
        return_meta=return_meta,
        use_cache=use_cache,
    ).open_dataset(**kwds)


def list_datasets(pattern=None, sample=None):
    return TutorialData().list_datasets(pattern=pattern, sample=sample)
