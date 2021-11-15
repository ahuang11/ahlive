from collections import Iterable
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


def to_1d(value, unique=False, flat=True, get=None):
    # pd.Series converts datetime to Timestamps
    if isinstance(value, xr.DataArray):
        value = value.values

    array = np.atleast_1d(value)
    if is_datetime(value):
        array = pd.to_datetime(array).values
    elif is_timedelta(value):
        array = pd.to_timedelta(array).values

    if array.ndim > 1 and get is not None:
        array = array[get]
    if unique:
        try:
            array = pd.unique(array)
        except ValueError:  # TODO: figure out ordered
            array = np.unique(array)
    if flat:
        array = array.ravel()
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


def to_scalar(value, get=-1, standard=False):
    value = to_1d(value)
    for _ in np.arange(len(value)):
        scalar = value[get]
        if pd.isnull(scalar):
            get -= 1
        else:
            break

    if standard and hasattr(scalar, "item"):
        scalar = scalar.item()

    return scalar


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


def pop(ds, key, dflt=None, get=None, squeeze=False, to_numpy=True, standard=False):
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
        array = to_scalar(array, get=get, standard=standard)

    if squeeze and hasattr(array, "squeeze"):
        array = array.squeeze()
        if is_scalar(array):
            array = array.item()
    return array


def srange(length, start=1, stride=1):
    if isinstance(length, xr.DataArray):
        length = np.size(length)
    return np.arange(start, length + start, stride)


def length(value):
    return len(to_1d(value))


def transpose(da, dims=None):
    if dims is None:
        item = "item" if "item" in da.dims else "ref_item"
        dims = (item, "state")
    return da.transpose(*dims)


def _fillna(da, how, dim="state"):
    kwds = {}
    transposed = False
    if isinstance(da, xr.DataArray) and dim in da.dims:
        kwds["dim"] = dim
    elif isinstance(da, pd.DataFrame) and dim == "state":
        da = da.transpose()
        transposed = True

    if how == "both":
        da = da.bfill(**kwds).ffill(**kwds)
    elif how == "ffill":
        da = da.ffill(**kwds)
    elif how == "bfill":
        da = da.bfill(**kwds)

    if transposed:
        da = da.transpose()
    return da


def fillna(da, how="ffill", dim="state", item_dim="item"):
    """ds.ffill does not handle datetimes"""
    if "state" not in da.dims:
        return da
    try:
        da = _fillna(da, how, dim=dim)
    except (TypeError, ImportError):
        if item_dim not in da.dims:
            da = _fillna(da.to_series(), how, dim=dim).to_xarray()
        else:
            da = xr.DataArray(
                _fillna(pd.DataFrame(da.values), how, dim=dim).values, dims=da.dims
            )
    return da


def remap(da, mapping):
    return np.array([mapping[k] for k in da.ravel()]).reshape(da.shape)


def traverse(obj):
    if isinstance(obj, Iterable):
        return traverse(obj[0])
    return obj


def get(ds, value, to_str=False):
    if isinstance(value, str):
        if value in ds:
            value = ds[value]
            if to_str:
                value = value.astype(str)
    return value
