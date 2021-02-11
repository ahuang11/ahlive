from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr


def to_1d(value, unique=False, flat=True):
    array = np.atleast_1d(value)
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


def is_subdtype(value, subdtype):
    if isinstance(subdtype, tuple):
        return any(is_subdtype(value, st) for st in subdtype)

    value = np.array(to_1d(value)).ravel()
    return np.issubdtype(value.dtype, subdtype)


def is_datetime(value):
    return is_subdtype(value, (np.datetime64, datetime)) and not is_str(value)


def is_timedelta(value):
    return is_subdtype(value, (np.timedelta64, timedelta))


def is_numeric(value):
    date_time_delta = is_datetime(value) or is_timedelta(value)
    if not date_time_delta:
        if is_str(value):
            return np.char.isnumeric(value.astype(str)).all()
        else:
            return is_subdtype(value, np.number)
    return False


def is_str(value):
    is_obj = is_subdtype(value, (np.string_, np.unicode, np.object))
    return is_obj and not is_subdtype(value, np.number)


def to_scalar(value, get=-1):
    value = to_1d(value)[get]
    return value


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


def _fillna(da, how):
    if how == "both":
        da = da.bfill().ffill()
    elif how == "ffill":
        da = da.ffill()
    elif how == "bfill":
        da = da.bfill()
    return da


def fillna(da, how="ffill"):
    """ds.ffill does not handle datetimes"""
    if "state" not in da.dims:
        return da
    try:
        da = da.ffill("state")
    except (TypeError, ImportError):
        if "item" not in da.dims:
            da = _fillna(da.to_series(), how).to_xarray()
        else:
            da = xr.concat(
                (
                    _fillna(da.sel(item=item).to_series(), how=how).to_xarray()
                    for item in da["item"]
                ),
                "item",
            )
    return da
