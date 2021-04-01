import warnings
from operator import itemgetter

import xarray as xr
import numpy as np

from .configuration import ITEMS, VARS
from .util import fillna, srange


def _get_rowcols(objs):
    rowcols = set([])
    for obj in objs:
        rowcols |= set(obj.data)
    return rowcols


def _get_item_dim(ds):
    for item_dim in ds.dims:
        if item_dim.endswith("item"):
            return item_dim


def _shift_items(ds, ds2):
    for item in VARS["item"]:
        if not (item in ds.dims and item in ds2.dims):
            continue
        has_same_items = (
            len(set(ds[item].values) | set(ds2[item].values)) > 0
        )
        if has_same_items:
            ds2[item] = ds2[item].copy()
            ds2[item] = ds2[item] + ds[item].max()
    return ds2


def _match_states(ds, ds2, method="ffill"):
    self_num_states = len(ds["state"])
    other_num_states = len(ds2["state"])

    if method == "ffill":
        if self_num_states > other_num_states:
            ds2 = ds2.reindex(state=ds["state"]).map(fillna, keep_attrs=True)
        else:
            ds = ds.reindex(state=ds2["state"]).map(fillna, keep_attrs=True)
    elif method == "shift":
        if self_num_states > other_num_states:
            shift_states = self_num_states - other_num_states
            ds2['state'] = ds2['state'] + shift_states
        else:
            shift_states = other_num_states - self_num_states
            ds['state'] = ds['state'] + shift_states
    return ds, ds2


def _drop_state(joined_ds):
    for var in VARS["stateless"]:
        if var in joined_ds:
            if "state" in joined_ds[var].dims:
                joined_ds[var] = joined_ds[var].isel(state=-1)
    return joined_ds


def _combine_ds_list(ds_list, method="concat", concat_dim="state", **kwds):
    joined_attrs = {}
    for ds in ds_list:
        item_dim = _get_item_dim(ds)
        for (key, val) in ds.attrs.items():
            if key not in joined_attrs:
                joined_attrs[key] = val
    if method == "concat":
        kwds["dim"] = concat_dim
    elif method == "combine_nested":
        kwds["concat_dim"] = concat_dim
    kwds["combine_attrs"] = "drop"
    ds = getattr(xr, method)(ds_list, **kwds).assign_attrs(**joined_attrs)
    ds[item_dim] = srange(ds[item_dim])
    ds = ds.transpose(item_dim, "state", ...)
    return ds


def _stack_data(data_list, join, rowcol):
    """Helper for cascade, overlay, stagger, slide"""
    num_data = len(data_list)
    if num_data == 1:
        return data_list[0][rowcol]

    ds_list = []
    max_items = 1
    for i, data in enumerate(data_list):
        if rowcol not in data:
            continue

        ds = data[rowcol]
        item_dim = _get_item_dim(ds)
        num_item = len(ds[item_dim])
        if item_dim not in ds.coords:
            ds[item_dim] = srange(ds[item_dim])

        max_item = max(ds[item_dim].values)
        max_items = max_item if max_item > max_items else max_items

        if num_item == 1:
            item = [i + max_items]
        else:
            item = srange(max_items)

        if join == 'stagger':
            # interweave the states
            # converts ds['state'] = [1, 2, 3] and ds2['state'] = [1, 2, 3]
            # to [[1, 3, 5], [2, 4, 6]]
            states = np.arange(
                ds['state'].max() * num_data
            ).reshape(-1, num_data).T + 1
            ds = ds.assign_coords(**{item_dim: item, 'state': states[i]})
        elif join == 'slide':
            ds = ds.assign_coords(**{item_dim: item, 'state': ds['state'] + i})
        else:
            ds = ds.assign_coords(**{item_dim: item})

        ds_list.append(ds)

    if join == 'cascade':
        joined_ds = _combine_ds_list(ds_list)
        joined_ds['state'] = srange(joined_ds['state'])
        joined_ds = joined_ds.map(fillna, keep_attrs=True)
    elif join == 'overlay':
        joined_ds = _combine_ds_list(ds_list, concat_dim=item_dim)
        joined_ds['state'] = srange(joined_ds['state'])
    else:
        joined_ds = _combine_ds_list(ds_list, method='merge')
        joined_ds = joined_ds.sortby('state')
        joined_ds = joined_ds.map(fillna, how='both', keep_attrs=True)

    joined_ds =_drop_state(joined_ds).map(fillna, keep_attrs=True)
    joined_ds[item_dim] = srange(joined_ds[item_dim])
    return joined_ds


def _wrap_stack(objs, join):
    objs = [obj.copy() for obj in objs if obj is not None]

    obj = objs[0]
    rowcols = _get_rowcols(objs)
    for rowcol in rowcols:
        data_list = [obj.data for obj in objs if obj.data[rowcol] is not None]
        obj.data[rowcol] = _stack_data(data_list, join, rowcol)
    return obj


def cascade(objs):
    return _wrap_stack(objs, 'cascade')


def overlay(objs):
    return _wrap_stack(objs, 'overlay')


def stagger(objs):
    return _wrap_stack(objs, 'stagger')


def slide(objs):
    return _wrap_stack(objs, 'slide')


def cols(obj, num_cols):
    obj = obj.copy()
    if num_cols == 0:
        raise ValueError("Number of columns must be > 1!")
    data = {}
    for iplot, rowcol in enumerate(list(obj.data)):
        row = (iplot) // num_cols + 1
        col = (iplot) % num_cols + 1
        data[(row, col)] = obj.data.pop(rowcol)
    obj.data = data
    return obj


def _layout_objs(objs, by):
    num_objs = len(objs)
    obj = objs[0]
    if num_objs == 1:
        return obj
    rowcol = list(obj.data.keys())[0]
    flip = slice(None, None, -1) if by == 'col' else slice(None, None, 1)
    obj.data = {(extent, 1)[flip]: array.data[rowcol] for extent, array in enumerate(objs, 1)}
    return obj


def layout(objs, by='col', num_cols=None):
    if by not in ['row', 'col']:
        raise ValueError('Only by=row or by=col available!')

    objs = [obj.copy() for obj in objs if obj is not None]
    obj = objs[0]
    obj = _layout_objs(objs, by)

    if num_cols is not None:
        obj = obj.cols(num_cols)
    return obj


def merge(objs, join="overlay"):
    if join == "overlay":
        obj = overlay(objs)
    elif join == "layout":
        obj = layout(objs)
    elif join == "cascade":
        obj = cascade(objs)
    elif join == "stagger":
        obj = stagger(objs)
    elif join == "slide":
        obj = slide(objs)
    else:
        raise NotImplementedError(
            f'Only {ITEMS["join"]} are implemented for merge; got {join}'
        )
    return obj
