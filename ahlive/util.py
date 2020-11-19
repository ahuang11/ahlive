import numpy as np
import xarray as xr

from .config import sizes, defaults


def to_pydt(*values):
    if values is None:
        return
    array = np.array(values)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype('M8[ms]').astype('O')
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


def is_datetime(value):
    value = np.array(value)
    return np.issubdtype(value.dtype, np.datetime64)


def to_scalar(value, get=-1):
    value = np.atleast_1d(value).flat[get]
    return value


def pop(ds, key, dflt=None, get=None, squeeze=False, to_numpy=True):
    try:
        array = ds[key]
        if to_numpy:
            array = np.atleast_1d(array)
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


def srange(length, start=1):
    if isinstance(length, xr.DataArray):
        length = np.size(length)
    return np.arange(start, length + start)


def transpose(da, dims=None):
    if dims is None:
        item = 'item' if 'item' in da.dims else 'ref_item'
        dims = (item, 'state')
    return da.transpose(*dims)


def ffill(da):
    if 'state' not in da.dims:
        return da
    try:
        da = da.ffill('state')
    except TypeError:
        if 'item' not in da.dims:
            da = da.to_series().ffill().to_xarray()
        else:
            da = xr.concat((
                da.sel(item=item).to_series().ffill().to_xarray()
                for item in da['item']
            ), 'item')
    return da
