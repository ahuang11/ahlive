import numpy as np
import xarray as xr

from .config import sizes, defaults


def to_pydt(*values):
    if values is None:
        return
    array = np.array(values)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype('M8[ms]').astype('O')
    if len(array.flat) == 1:
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
    return len(np.atleast_1d(value).flat) == 1


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
        array = np.atleast_1d(array)[get]

    if squeeze:
        array = array.squeeze()
        if is_scalar(array):
            array = array.item()
    return array


def srange(length, start=1):
    return np.arange(start, length + start)


def ffill(arr):
    # https://stackoverflow.com/questions/41190852/
    mask = np.isnan(arr)
    indices = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    out = arr[np.arange(indices.shape[0])[:, None], indices]
    return out


def transpose(da, dims=None):
    if dims is None:
        dims = ('item', 'state')
    return da.transpose(*dims)
