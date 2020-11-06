import numpy as np
import xarray as xr

from .config import sizes, defaults


def try_to_pydatetime(*values):
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
    return len(np.atleast_1d(value)) == 1


def pop(ds, key, dflt=None, squeeze=True, to_numpy=True):
    try:
        array = ds[key]
        if to_numpy:
            array = np.atleast_1d(array)
        del ds[key]
    except KeyError:
        array = dflt
    return array


def ffill(arr):
    # https://stackoverflow.com/questions/41190852/
    mask = np.isnan(arr)
    indices = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    out = arr[np.arange(indices.shape[0])[:, None], indices]
    return out


def cascade(arrays):
    array0 = arrays[0]
    for array in arrays[1:]:
        array0 -= array
    return array0


def overlay(arrays):
    array0 = arrays[0]
    for array in arrays[1:]:
        array0 *= array
    return array0


def layout(arrays, cols=None):
    array0 = arrays[0]
    for array in arrays[1:]:
        array0 += array
    if cols is not None:
        array0 = array0.cols(cols)
    return array0


def scale_sizes(scale, keys=None):
    if keys is None:
        keys = sizes.keys()

    for key in keys:
        sizes[key] = sizes[key] * scale


def load_defaults(default_key, input_kwds=None, **other_kwds):
    updated_kwds = defaults.get(default_key, {}).copy()
    if default_key in ['chart_kwds', 'ref_plot_kwds']:
        updated_kwds = updated_kwds.get(
            other_kwds.pop('base_chart', None), updated_kwds
        ).copy()
    if isinstance(input_kwds, xr.Dataset):
        input_kwds = input_kwds.attrs[default_key]
    updated_kwds.update(
        {key: val for key, val in other_kwds.items()
        if val is not None
    })
    if input_kwds is not None:
        updated_kwds.update(input_kwds)
    updated_kwds.pop('base_chart', None)
    return updated_kwds


def update_defaults(default_key, **kwds):
    defaults[default_key].update(**kwds)


def transpose(da, dims=None):
    if dims is None:
        dims = ('item', 'state')
    return da.transpose(*dims)
