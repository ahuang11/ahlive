import numpy as np


def try_to_pydatetime(*values):
    array = np.array(values)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype('M8[ms]').astype('O')
    if len(array.flat) == 1:
        return array.flat[0]
    else:
        return array


def is_scalar(value):
    return len(np.atleast_1d(value)) == 1


def pop(ds, key, dflt=None):
    try:
        array = ds[key].values
        del ds[key]
    except KeyError:
        array = dflt
    return array
