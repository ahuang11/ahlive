from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)


def to_1d(value, unique=False, flat=True):
    # pd.Series converts datetime to Timestamps
    if isinstance(value, xr.DataArray):
        value = value.values

    array = np.atleast_1d(value)
    if is_datetime(value):
        array = pd.to_datetime(array).values
    elif is_timedelta(value):
        array = pd.to_timedelta(array).values

    if unique:
        array = pd.unique(array)
    if flat:
        array = array.flat
    return array


def to_pydt(*values):
    if values is None:
        return
    array = to_1d(
        values,
        flat=False,
    )
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype("M8[ms]").astype("O")
    if np.size(array) == 1:
        return array.flat[0]
    else:
        return array


def to_list(value):
    if not isinstance(value, list):
        value = [value]
    return value


def to_set(iterable):
    return set(np.array(iterable))


def to_str(string):
    if not isinstance(string, str):
        if len(string) > 0:
            string = string[0]
    return string


def to_num(num):
    try:
        return float(num)
    except ValueError:
        return num


def is_scalar(value):
    return np.size(value) == 1


def to_scalar(value, get=-1):
    value = to_1d(value)[get]
    return value


def is_datetime(value):
    if isinstance(value, (list, tuple)):
        value = pd.Series(value)
    return is_datetime64_any_dtype(value) or isinstance(value, datetime)


def is_timedelta(value):
    if isinstance(value, (list, tuple)):
        value = pd.Series(value)
    return is_timedelta64_dtype(value) or isinstance(value, timedelta)


def is_numeric(value):
    if isinstance(value, (list, tuple)):
        value = pd.Series(value)
    return is_numeric_dtype(value) or isinstance(value, (int, float))


def is_str(value):
    if isinstance(value, (list, tuple)):
        value = pd.Series(value)
    return is_string_dtype(value) or isinstance(value, str)


def pop(ds, key, dflt=None, get=None, squeeze=False, to_numpy=True):
    try:
        array = ds[key]
        if to_numpy:
            array = to_1d(array, flat=False)
        del ds[key]
    except KeyError:
        array = dflt

    if array is None:
        return array

    if get is not None:
        array = to_scalar(array, get=get)

    if squeeze:
        array = array.squeeze()
        if is_scalar(array):
            array = array.item()
    return array


def srange(length, start=1, stride=1):
    if isinstance(length, xr.DataArray):
        length = np.size(length)
    return np.arange(start, length + start, stride)


def transpose(da, dims=None):
    if dims is None:
        item = "item" if "item" in da.dims else "ref_item"
        dims = (item, "state")
    return da.transpose(*dims)


def _fillna(da, how, dim="state"):
    kwds = {}
    if dim in da:
        kwds["axis"] = dim

    if how == "both":
        da = da.bfill(**kwds).ffill(**kwds)
    elif how == "ffill":
        da = da.ffill(**kwds)
    elif how == "bfill":
        da = da.bfill(**kwds)
    return da


def fillna(da, how="ffill", dim="state"):
    """ds.ffill does not handle datetimes"""
    if "state" not in da.dims:
        return da
    try:
        da = _fillna(da, how, dim=dim)
    except (TypeError, ImportError):
        if "item" not in da.dims:
            da = _fillna(da.to_series(), how, dim=dim).to_xarray()
        else:
            da = xr.concat(
                (
                    _fillna(da.sel(item=item).to_series(), how, dim=dim).to_xarray()
                    for item in da["item"]
                ),
                "item",
            )
    return da
