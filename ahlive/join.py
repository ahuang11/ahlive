import xarray as xr

from .util import srange, ffill


def _get_rowcols(objs):
    rowcols = set([])
    for array in objs:
        rowcols |= set(array.data)
    return rowcols


def _combine(objs, method='concat', concat_dim='state', **kwds):
    combined_attrs = {}
    for obj in objs:
        for key, val in obj.attrs.items():
            if key not in combined_attrs:
                combined_attrs[key] = val
    if method == 'concat':
        kwds['dim'] = concat_dim
    elif method == 'combine_nested':
        kwds['concat_dim'] = concat_dim
    kwds['combine_attrs'] = 'drop'
    return getattr(xr, method)(objs, **kwds).assign_attrs(**combined_attrs)


def cascade(objs, quick=False):
    if len(objs) == 1:
        return objs[0]

    obj = objs[0]
    if quick:
        rowcols = _get_rowcols(objs)
        obj.data = {
            rowcol: _combine([
                array.data[rowcol].assign_coords(item=[i])
                for i, array in enumerate(objs)],
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
            rowcol: _combine([
                array.data[rowcol] for array in objs
                if rowcol in array.data], concat_dim='item'
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
