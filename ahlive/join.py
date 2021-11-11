import numpy as np
import pandas as pd
import xarray as xr

from .configuration import OPTIONS, VARS
from .util import fillna, is_str, srange


def _get_rowcols(objs):
    rowcols = set([])
    for obj in objs:
        rowcols |= set(obj.data)
    return rowcols


def _get_item_dim(ds):
    for item_dim in VARS["item"]:
        if item_dim in ds.dims:
            return item_dim


def _shift_items(ds, ds2):
    for item in VARS["item"]:
        if not (item in ds.dims and item in ds2.dims):
            continue
        has_same_items = len(set(ds[item].values) | set(ds2[item].values)) > 0
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
            ds2["state"] = ds2["state"] + shift_states
        else:
            shift_states = other_num_states - self_num_states
            ds["state"] = ds["state"] + shift_states
    return ds, ds2


def _match_layout_states(obj):
    max_states = np.max([ds["state"].values.max() for ds in obj.values()])
    states = srange(max_states)

    data = {}
    for i, rowcol in enumerate(obj.data.keys()):
        ds = obj.data[rowcol]
        num_states = len(ds["state"])
        if num_states != max_states:
            ds = ds.reindex(state=states).map(fillna, keep_attrs=True)
        data[rowcol] = ds
    return data


def _drop_state(joined_ds):
    for var in VARS["stateless"]:
        if var in joined_ds:
            if "state" in joined_ds[var].dims:
                value = joined_ds[var].isel(state=-1)
                if pd.isnull(value).all():
                    value = joined_ds[var].isel(state=0)
                joined_ds[var] = value
    return joined_ds


def _combine_ds_list(ds_list, method="concat", concat_dim="state", **kwds):
    joined_attrs = {}
    for i, ds in enumerate(ds_list):
        item_dim = _get_item_dim(ds)
        if item_dim is None:
            item_dim = "item"
            ds["item"] = len(ds_list) + i

        for key, val in ds.attrs.items():
            if key not in joined_attrs:
                joined_attrs[key] = val

    if method == "concat":
        kwds["dim"] = concat_dim
    elif method == "combine_nested":
        kwds["concat_dim"] = concat_dim

    if "combine_attrs" not in kwds:
        kwds["combine_attrs"] = "drop"

    ds_list = ds_list[::-1]  # override with last item
    ds = getattr(xr, method)(ds_list, **kwds).assign_attrs(**joined_attrs)

    if "state" not in ds.dims:
        ds = ds.drop("state", errors="ignore").expand_dims("state")
    ds[item_dim] = srange(ds[item_dim])
    ds = ds.transpose(item_dim, "state", ...)

    for var in ds.data_vars:
        if len(ds.dims) > 2:
            break
        elif is_str(ds[var]):
            if "label" in var or "x" in var:
                continue
            ds[var] = fillna(ds[var], dim="item", how="both")

    return ds


def _stack_data(data_list, join, rowcol):
    """Helper for cascade, overlay, stagger, slide"""
    num_data = len(data_list)
    if num_data == 1:
        return data_list[0][rowcol]

    ds_list = []
    offset = 0
    for i, data in enumerate(data_list):
        if rowcol not in data:
            continue

        ds = data[rowcol]
        item_dim = _get_item_dim(ds)
        num_items = len(ds[item_dim])
        if item_dim not in ds.coords:
            ds[item_dim] = srange(ds[item_dim])

        max_item = np.max(ds[item_dim].values)
        if num_items == 1:
            item = [max_item + offset]
        else:
            item = srange(max_item) + offset * i

        if join == "stagger":
            # interweave the states
            # converts ds['state'] = [1, 2, 3] and ds2['state'] = [1, 2, 3]
            # to [[1, 3, 5], [2, 4, 6]]
            states = np.arange(ds["state"].max() * num_data).reshape(-1, num_data).T + 1
            ds = ds.assign_coords(**{item_dim: item, "state": states[i]})
        elif join == "slide":
            ds = ds.assign_coords(**{item_dim: item, "state": ds["state"] + i})
        elif join == "cascade":
            if i == 0:
                max_state = ds["state"].max()
            ds = ds.assign_coords(**{item_dim: item, "state": ds["state"] + max_state})
            max_state = ds["state"].max()
        else:
            ds = ds.assign_coords(**{item_dim: item})

        ds_list.append(ds)
        offset += num_items

    if join == "cascade":
        joined_ds = _combine_ds_list(ds_list, method="combine_by_coords")
        fillna_how = "ffill"
    elif join == "overlay":
        try:
            joined_ds = _combine_ds_list(ds_list, method="combine_by_coords")
        except Exception:
            joined_ds = _combine_ds_list(
                ds_list, concat_dim=item_dim, method="combine_by_coords"
            )
        fillna_how = "ffill"
    else:
        joined_ds = _combine_ds_list(ds_list, method="merge")
        joined_ds = joined_ds.sortby("state")
        fillna_how = "both"

    joined_ds["state"] = srange(joined_ds["state"])
    joined_ds = _drop_state(joined_ds.map(fillna, how=fillna_how, keep_attrs=True))
    joined_ds[item_dim] = srange(joined_ds[item_dim])
    joined_ds = joined_ds.transpose(..., "state")
    return joined_ds


def _wrap_stack(objs, join):
    objs = [obj.copy() for obj in objs if obj is not None]

    obj0 = objs[0]

    data = {}
    rowcols = _get_rowcols(objs)
    for rowcol in rowcols:
        data_list = [obj.data for obj in objs if obj.data.get(rowcol) is not None]
        if len(data_list) == 0:
            continue
        ds = _stack_data(data_list, join, rowcol)
        data[rowcol] = ds

    obj0.data = data
    for obj in objs[1:]:
        obj0 = obj0._propagate_params(obj0, obj)
    return obj0


def cascade(objs):
    return _wrap_stack(objs, "cascade")


def overlay(objs):
    return _wrap_stack(objs, "overlay")


def stagger(objs):
    return _wrap_stack(objs, "stagger")


def slide(objs):
    return _wrap_stack(objs, "slide")


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

    data = {}
    obj0 = objs[0]
    for obj in objs:
        for rowcol, ds in obj.data.items():
            for _ in np.arange(0, 1000):
                if rowcol in data.keys():
                    if by == "row":
                        rowcol = rowcol[0], rowcol[1] + 1
                    elif by == "col":
                        rowcol = rowcol[0] + 1, rowcol[1]
                else:
                    break
            else:
                raise ValueError("Could not find an open subplot row or col")
            data[rowcol] = ds
    obj0.data = data

    for obj in objs[1:]:
        obj0 = obj0._propagate_params(obj0, obj)
    return obj0


def layout(objs, by="row", num_cols=None):
    if by not in ["row", "col"]:
        raise ValueError("Only by=row or by=col available!")

    objs = [obj.copy() for obj in objs if obj is not None]
    obj = _layout_objs(objs, by)

    if num_cols is not None:
        obj = obj.cols(num_cols)
    obj.data = _match_layout_states(obj)
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
            f'Only {OPTIONS["join"]} are implemented for merge; got {join}'
        )
    return obj
