import xarray as xr

from .util import srange, ffill


QUICK_CONCAT_KWDS = dict(
    combine_attrs='override'
)


def _get_rowcols(objs):
    rowcols = set([])
    for array in objs:
        rowcols |= set(array.data)
    return rowcols


def cascade(objs, quick=False):
    if len(objs) == 1:
        return objs[0]

    obj = objs[0]
    if quick:
        rowcols = _get_rowcols(objs)
        obj.data = {
            rowcol: xr.concat((
                array.data[rowcol].assign_coords(item=[i])
                for i, array in enumerate(objs)), 'state', **QUICK_CONCAT_KWDS
            ).pipe(
                lambda ds: ds.assign(state=srange(ds['state']))
            ).map(ffill, keep_attrs=True)
            for rowcol in rowcols
        }
    else:
        for array in objs[1:]:
            obj -= array
    return obj


def overlay(objs, quick=False):
    if len(objs) == 1:
        return objs[0]

    obj = objs[0]
    if quick:
        rowcols = _get_rowcols(objs)
        obj.data = {
            rowcol: xr.concat((
                array.data[rowcol] for array in objs
                if rowcol in array.data), 'item', **QUICK_CONCAT_KWDS
            ).pipe(lambda ds: ds.assign(item=srange(ds['item'])))
            for rowcol in rowcols
        }
    else:
        for array in objs[1:]:
            obj *= array
    return obj


def layout(objs, cols=None, quick=False):
    if len(objs) == 1:
        return objs[0]

    obj = objs[0]
    if quick:
        rowcol = list(obj.data.keys())[0]
        obj.data = {
            (row, 1): array.data[rowcol]
            for row, array in enumerate(objs, 1)}
    else:
        for array in objs[1:]:
            obj += array

    if cols is not None:
        obj = obj.cols(cols)
    return obj
