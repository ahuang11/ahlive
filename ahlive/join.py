import xarray as xr

from .configuration import ITEMS
from .util import fillna, srange


def _get_rowcols(objs):
    rowcols = set([])
    for array in objs:
        rowcols |= set(array.data)
    return rowcols


def _get_item_dim(ds):
    for item_dim in ds.dims:
        if item_dim.endswith("item"):
            return item_dim


def _combine(ds_list, method="concat", concat_dim="state", **kwds):
    combined_attrs = {}
    for ds in ds_list:
        item_dim = _get_item_dim(ds)
        for (key, val) in ds.attrs.items():
            if key not in combined_attrs:
                combined_attrs[key] = val
    if method == "concat":
        kwds["dim"] = concat_dim
    elif method == "combine_nested":
        kwds["concat_dim"] = concat_dim
    kwds["combine_attrs"] = "drop"
    ds = getattr(xr, method)(ds_list, **kwds).assign_attrs(**combined_attrs)
    ds[item_dim] = srange(ds[item_dim])
    ds["state"] = srange(ds["state"])
    ds = ds.transpose(item_dim, "state", ...)
    return ds


def cascade(objs, quick=False):
    if len(objs) == 1:
        return objs[0]

    obj = objs[0]
    if quick:
        rowcols = _get_rowcols(objs)
        obj.data = {
            rowcol: _combine(
                [
                    array.data[rowcol].assign_coords(
                        **{_get_item_dim(array.data[rowcol]): [i]}
                    )
                    for i, array in enumerate(objs)
                    if rowcol in array.data
                ],
            ).map(
                fillna,
                keep_attrs=True,
            )
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
            rowcol: _combine(
                [array.data[rowcol] for array in objs if rowcol in array.data],
                concat_dim=_get_item_dim(objs[0].data[rowcol]),
            )
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
        obj.data = {(row, 1): array.data[rowcol] for row, array in enumerate(objs, 1)}
    else:
        for array in objs[1:]:
            obj += array

    if cols is not None:
        obj = obj.cols(cols)
    return obj


def merge(objs, join="overlay", quick=False):
    if join == "overlay":
        obj = overlay(objs, quick=quick)
    elif join == "layout":
        obj = layout(objs, quick=quick)
    elif join == "cascade":
        obj = cascade(objs, quick=quick)
    else:
        raise NotImplementedError(
            f'Only {ITEMS["join"]} are implemented for merge; got {join}'
        )
    return obj
