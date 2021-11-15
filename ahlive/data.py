import warnings
from collections.abc import Iterable
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import param
import xarray as xr
from matplotlib.colors import BoundaryNorm

from .animation import Animation
from .configuration import (
    CANVAS,
    CHARTS,
    CONFIGURABLES,
    DEFAULTS,
    DIMS,
    ITEMS,
    NULL_VALS,
    OPTIONS,
    PARAMS,
    PRECEDENCES,
    PRESETS,
    VARS,
    CartopyCRS,
    CartopyFeature,
    CartopyTiles,
    Configuration,
    load_defaults,
)
from .easing import Easing
from .join import (
    _combine_ds_list,
    _drop_state,
    _get_item_dim,
    _wrap_stack,
    cols,
    layout,
    merge,
)
from .util import (
    fillna,
    get,
    is_datetime,
    is_numeric,
    is_scalar,
    is_str,
    is_timedelta,
    length,
    pop,
    remap,
    srange,
    to_1d,
    to_scalar,
)


class Data(Easing, Animation, Configuration):

    chart = param.ClassSelector(
        class_=Iterable,
        doc=f"Type of plot; {CHARTS['all']}",
        precedence=PRECEDENCES["common"],
    )
    preset = param.ObjectSelector(
        objects=list(PRESETS.keys()),
        allow_None=True,
        doc=f"Chart preset; {PRESETS}",
        precedence=PRECEDENCES["common"],
    )
    state_labels = param.ClassSelector(
        class_=(Iterable,),
        doc="Dynamic label per state (bottom right)",
        precedence=PRECEDENCES["common"],
    )
    inline_labels = param.ClassSelector(
        class_=(Iterable,),
        doc="Dynamic label per item per state (item location)",
        precedence=PRECEDENCES["common"],
    )
    label = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Legend label for each item",
        precedence=PRECEDENCES["common"],
    )
    group = param.ClassSelector(
        class_=(int, float, str),
        default=None,
        doc="Group label for multiple items",
        precedence=PRECEDENCES["misc"],
    )

    title = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Title label (outer top left)",
        precedence=PRECEDENCES["label"],
    )
    xlabel = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="X-axis label (bottom center)",
        precedence=PRECEDENCES["label"],
    )
    ylabel = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Y-axis label (left center",
        precedence=PRECEDENCES["label"],
    )
    subtitle = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Subtitle label (outer top right)",
        precedence=PRECEDENCES["sub_label"],
    )
    note = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Note label (bottom left)",
        precedence=PRECEDENCES["sub_label"],
    )
    caption = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Caption label (outer left)",
        precedence=PRECEDENCES["sub_label"],
    )
    xlims = param.ClassSelector(
        class_=Iterable, doc="Limits for the x-axis", precedence=PRECEDENCES["limit"]
    )
    ylims = param.ClassSelector(
        class_=Iterable, doc="Limits for the y-axis", precedence=PRECEDENCES["limit"]
    )
    xlim0s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the left bounds of the x-axis",
        precedence=PRECEDENCES["limit"],
    )
    xlim1s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the right bounds of the x-axis",
        precedence=PRECEDENCES["limit"],
    )
    ylim0s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the bottom bounds of the y-axis",
        precedence=PRECEDENCES["limit"],
    )
    ylim1s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the top bounds of the y-axis",
        precedence=PRECEDENCES["limit"],
    )
    xmargins = param.ClassSelector(
        class_=(tuple, int, float),
        doc="Margins on the x-axis; ranges from 0-1",
        precedence=PRECEDENCES["limit"],
    )
    ymargins = param.ClassSelector(
        class_=(tuple, int, float),
        doc="Margins on the y-axis; ranges from 0-1",
        precedence=PRECEDENCES["limit"],
    )

    xticks = param.ClassSelector(
        class_=(Iterable,), doc="X-axis tick locations", precedence=PRECEDENCES["style"]
    )
    yticks = param.ClassSelector(
        class_=(Iterable,), doc="Y-axis tick locations", precedence=PRECEDENCES["style"]
    )

    legend = param.ObjectSelector(
        objects=OPTIONS["legend"],
        doc="Legend location",
        precedence=PRECEDENCES["style"],
    )
    grid = param.ObjectSelector(
        objects=OPTIONS["grid"],
        doc="Grid type",
        precedence=PRECEDENCES["style"],
    )
    style = param.ObjectSelector(
        objects=OPTIONS["style"],
        doc=f"Chart style; {OPTIONS['style']}",
        precedence=PRECEDENCES["style"],
    )
    adjust_text = param.Boolean(
        default=None,
        doc="Whether to use adjustText to adjust "
        "inline labels' location minimize overlap",
        precedence=PRECEDENCES["style"],
    )

    hooks = param.HookList(
        doc="List of customization functions to apply; "
        "function must contain fig and ax as arguments",
        precedence=PRECEDENCES["misc"],
    )

    rowcol = param.NumericTuple(
        default=(1, 1),
        length=2,
        doc="Subplot location as (row, column)",
        precedence=PRECEDENCES["misc"],
    )

    num_rows = param.Integer(doc="Number of rows", **DEFAULTS["num_kwds"])
    num_cols = param.Integer(doc="Number of cols", **DEFAULTS["num_kwds"])

    configurables = param.Dict(
        default={},
        doc="Possible configuration keys",
        constant=True,
        precedence=PRECEDENCES["attr"],
    )
    _attrs = param.Dict(
        doc="Attributes of the first dataset",
        constant=True,
        precedence=PRECEDENCES["attr"],
    )

    _crs_names = {}
    _tiles_names = {}

    def __init__(self, **kwds):
        self._ds = None
        self._parameters = set(
            key
            for key in dir(self)
            if not key.startswith("_")
            and key not in ["xs", "ys", "cs", "x0s", "y0s", "x1s", "y1s"]
        )
        kwds = self._set_input_vars(**kwds)
        super().__init__(**kwds)
        self._init_num_states()
        self._load_dataset()
        self.configurables.update(
            {
                "canvas": CONFIGURABLES["canvas"],
                "subplot": CONFIGURABLES["subplot"],
                "label": CONFIGURABLES["label"],
            }
        )

    def _init_num_states(self):
        cs = self._input_vars.get("cs", self._input_vars.get("us"))
        if cs is not None:
            cs = np.array(cs)
            shape = cs.shape[-2:]
            if cs.ndim > 2:
                num_states = len(cs)
                if shape[0] != len(self._input_vars["ys"]):
                    cs = np.swapaxes(cs, -1, -2)
                    shape = shape[::-1]  # TODO: auto figure out time dimension
            elif cs.ndim == 2:
                num_states = 1
            else:
                num_states = len(cs)
        else:
            num_states = max(
                max(np.shape(val)) if np.shape(val) else 1
                for val in self._input_vars.values()
            )

        with param.edit_constant(self):
            self.num_states = num_states

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = self._config_data(data)
        with param.edit_constant(self):
            ds = list(self._data.values())[0]

            for var in VARS["itemless"]:
                if var in ds:
                    if "item" in ds[var].dims:
                        ds[var] = fillna(ds[var], dim="item").isel(item=-1)

            self._ds = ds
            self.attrs = ds.attrs
            self.num_states = len(ds["state"])
            self.num_rows, self.num_cols = [
                max(rowcol) for rowcol in zip(*self.data.keys())
            ]

    def cols(self, num_cols):
        self_copy = cols(self, num_cols)
        self_copy._cols = num_cols
        return self_copy

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        strings = []
        for rowcol, ds in self.data.items():
            dims = ", ".join(f"{key}: {val}" for key, val in ds.dims.items())
            data = repr(ds.data_vars)

            attrs = "\n"
            counts = 0
            for key, val in ds.attrs.items():
                if key in ["configured", "finalized"] or len(val) == 0:
                    continue

                key_str = str(key)
                if len(key) > 12:
                    key_str = key[:9] + "..."
                val_str = str(val)
                if len(val_str) > 105:
                    val_str = val_str[:102] + "..."
                attrs += f'{" ":4}{key_str:13}{val_str}\n'
                counts += 1

            attrs_ratio = f"{counts}/{len(ds.attrs)}"
            strings.append(
                f'Subplot:{" ":9}{rowcol}\n'
                f'Dimensions:{" ":6}({dims})\n'
                f"{data}\n"
                f"Attributes ({attrs_ratio}):{attrs}\n"
            )
        return "<ahlive.Data>\n" + "".join(strings)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return self.overlay(other)

    def __rmul__(self, other):
        return other.overlay(self)

    def __floordiv__(self, other):
        return self.slide(other)

    def __pow__(self, other):
        return self.stagger(other)

    def __truediv__(self, other):
        return self.layout(other, by="col")

    def __add__(self, other):
        return self.layout(other, by="row")

    def __radd__(self, other):
        return other.layout(self, by="row")

    def __sub__(self, other):
        return self.cascade(other)

    def __rsub__(self, other):
        return other.cascade(self)

    def __iter__(self):
        return self.data.__iter__()

    def __eq__(self, other):
        if not isinstance(other, Data):
            raise TypeError("Other object is not an ah.Data object!")
        return self.equals(other)

    def copy(self):
        return deepcopy(self)

    def _stack(self, other, how):
        self_copy = _wrap_stack([self, other], how)
        self_copy = self_copy._propagate_params(self_copy, other)
        return self_copy

    def cascade(self, other):
        return self._stack(other, how="cascade")

    def overlay(self, other):
        return self._stack(other, how="overlay")

    def stagger(self, other):
        return self._stack(other, how="stagger")

    def slide(self, other):
        return self._stack(other, how="slide")

    def layout(self, other, by="row", num_cols=None):
        self_copy = layout([self, other], by)
        self_copy = self_copy._propagate_params(self_copy, other, layout=True)
        return self_copy

    def equals(self, other):
        for self_items, other_items in zip(self.data.items(), other.data.items()):
            self_rowcol, self_ds = self_items
            other_rowcol, other_ds = other_items
            return self_items == other_items and self_ds.equals(other_ds)

    @staticmethod
    def _get_cumulation(data, **kwargs):
        # https://stackoverflow.com/a/38900035/9324652
        cum = data.clip(**kwargs)
        cum = np.cumsum(cum, axis=0)
        d = np.zeros(np.shape(data))
        d[1:] = cum[:-1]
        return d

    def _config_bar_chart(self, ds, chart, preset):
        num_items = len(ds["item"])
        one_bar = ds["chart"].str.startswith("bar").sum() == 1

        width_key = "width" if chart == "bar" else "height"
        ds["tick_label"] = ds["x"]
        if not one_bar:
            if "morph" not in preset and (not preset or preset == "stacked"):
                warnings.warn(
                    "Multiple items found, you may want to use the 'morph' preset"
                )
            if is_str(ds["x"]):
                mapping = {x: i for i, x in enumerate(np.unique(ds["x"].values))}
                ds["x"] = xr.apply_ufunc(remap, ds["x"], kwargs=dict(mapping=mapping))
                ds.attrs["xticks_kwds"]["mapping"] = {
                    float(val): key for key, val in mapping.items()
                }

            if "stacked" in preset:
                cumulation = self._get_cumulation(ds["y"].values, min=0)
                negative_cumulation = self._get_cumulation(ds["y"].values, max=0)
                negative_mask = ds["y"].values < 0
                cumulation[negative_mask] = negative_cumulation[negative_mask]
                lim_key = "xlim1" if chart == "barh" else "ylim1"
                ds["bar_offset"] = ds["y"].dims, cumulation
                if lim_key not in ds.data_vars:
                    ds[lim_key] = ds["y"].sum("item")
            elif preset not in ["race", "delta"]:
                # for side by side bars
                width = 1 / num_items / 1.5
                offsets = (width * (1 - num_items) / num_items) + np.arange(
                    num_items
                ) * width
                offsets += offsets.mean() / 2
                shape = (-1, 1, 1) if "batch" in ds else (-1, 1)
                ds["x"] = ds["x"] + offsets.reshape(shape)
                if width_key not in ds.attrs["plot_kwds"]:
                    ds.attrs["plot_kwds"][width_key] = width

        if not preset or preset == "stacked":
            if not one_bar:
                ds["x"].attrs["is_bar"] = True
            else:
                ds["tick_label"].attrs["is_bar"] = True
            return ds

        preset_kwds = load_defaults("preset_kwds", ds, base_chart=preset)
        ascending = preset_kwds.pop("ascending", False)
        bar_label = preset_kwds.get("bar_label", "morph" not in preset)
        if bar_label:
            ds["bar_label"] = ds["tick_label"]

        if preset == "race":
            preset_kwds = load_defaults("preset_kwds", ds, base_chart=preset)
            num_labels = len(np.unique(ds["label"]))
            limit = preset_kwds.get("limit", None)
            if limit > num_labels:
                limit = num_labels - 1

            if ascending:
                ds["y"] *= -1

            # rank; first increments value
            ds["x"] = xr.DataArray(
                pd.DataFrame(ds["y"].values)
                .rank(method="first", axis=0, ascending=True, na_option="top")
                .values,
                dims=ds["x"].dims,
            )

            # optimize runtime by only keeping the labels that show up
            limit_labels = pd.unique(
                ds["label"]
                .where(ds["x"] >= (ds["x"].max() - limit), drop=True)
                .values.ravel()
            )
            ds = ds.sel(item=ds["label"].isin(limit_labels)["item"])

            # fill back in NaNs
            ds["y"] = ds["y"].where(np.isfinite(ds["y"]))
            if ascending:
                ds["y"] *= -1

            x_max = ds["x"].max()
            if chart == "bar":
                ds["xlim0"] = x_max - limit - 0.5
                ds["xlim1"] = x_max + 0.5
            else:
                ds["ylim0"] = x_max - limit - 0.5
                ds["ylim1"] = x_max + 0.5

        elif "morph" not in preset:
            if not is_str(ds["x"]):
                ds["x"] = ds["x"].rank("item")
            else:
                xs = np.repeat(
                    np.arange(num_items).reshape(1, -1), self.num_states
                ).reshape(num_items, self.num_states)
                ds["x"] = ds["x"].dims, xs

            if preset == "delta":
                x_delta = ds["x"].diff("item").mean() / 2
                ds["x_center"] = ds["x"] - x_delta
                ds["delta_label"] = ds["y"].diff("item")
                ds["y_center"] = ds["y"].shift(item=1) + ds["delta_label"] / 2
                ds["delta_label"] = ds["delta_label"].isel(item=slice(1, None))
                ds["delta"] = ds["delta_label"] / 2

        ds = _drop_state(ds)
        return ds

    @staticmethod
    def _config_trail_chart(ds, preset):
        preset_kwds = load_defaults("preset_kwds", ds, base_chart=preset)
        trail_chart = preset_kwds["chart"]
        if trail_chart in ["line", "both"]:
            ds["x_trail"] = ds["x"].copy()
            ds["y_trail"] = ds["y"].copy()
            if "morph" in preset:
                ds = ds.rename({"x_trail": "x_morph_trail", "y_trail": "y_morph_trail"})

        if trail_chart in ["scatter", "both"]:
            ds["x_discrete_trail"] = ds["x"].copy()
            ds["y_discrete_trail"] = ds["y"].copy()

        return ds

    @staticmethod
    def _config_morph_chart(ds, chart):
        group_ds_list = []
        ref_vars = [var for var in ds.data_vars if var.startswith("ref_")]
        ds["chart"] = chart
        if len(ref_vars) > 0:
            ref_ds = ds[ref_vars]
            ds = ds.drop_vars(ref_vars)
            ds = ds.drop("ref_item")
        else:
            ref_ds = None

        for group, group_ds in ds.groupby("group"):
            group_ds = group_ds.rename({"state": "batch", "item": "state"})
            group_ds["state"] = srange(group_ds["state"])
            if len(group_ds["state"]) == 1:
                group_ds = group_ds.squeeze("state", drop=True)
            group_ds_list.append(group_ds)

        ds = _combine_ds_list(group_ds_list, concat_dim="item")
        if "state" not in ds.dims:
            ds = ds.drop("state", errors="ignore").expand_dims("state")
        ds["item"] = srange(ds["item"])
        ds = ds.transpose("item", "batch", "state")

        if to_scalar(ds["group"]) == "":
            ds["group"] = "_morph_group"

        if ref_ds is not None:
            ref_ds = ref_ds.isel(state=0, drop=True)
            ds = xr.merge([ds, ref_ds])

        if chart == "errorbar":
            for key in ["xerr", "yerr"]:
                if key in ds.data_vars:
                    ds[key].attrs["is_errorbar_morph"] = True

        ds = _drop_state(ds)
        return ds

    def _config_pie_chart(self, ds):
        num_items = len(ds["item"])
        ds = ds.rename({"label": "labels"})
        ds["group"] = ("item", np.repeat("_pie_group", num_items))

        if "inline_label" not in ds.data_vars:
            ds["inline_label"] = ds["y"]

        need_norm = (ds["y"].sum("item") > 1).any()
        if need_norm:
            ds["y"] = (ds["y"] / ds["y"].sum("item")).fillna(0)
        return ds

    def _config_grid_axes(self, ds, chart):
        if self.style == "bare":
            ds.attrs["grid_kwds"]["b"] = ds.attrs["grid_kwds"].get("b", False)
        elif chart == "barh":
            ds.attrs["grid_kwds"]["axis"] = ds.attrs["grid_kwds"].get("axis", "x")
        elif chart == "bar":
            ds.attrs["grid_kwds"]["axis"] = ds.attrs["grid_kwds"].get("axis", "y")
        else:
            ds.attrs["grid_kwds"]["axis"] = ds.attrs["grid_kwds"].get("axis", "both")
        return ds

    @staticmethod
    def _config_legend(ds):
        legend_kwds = load_defaults("legend_kwds", ds)
        legend_sortby = legend_kwds.pop("sortby", None)
        if legend_sortby and "label" in ds:
            items = ds.max("state").sortby(legend_sortby, ascending=False)["item"]
            ds = ds.sel(item=items)
            ds["item"] = srange(ds["item"])

        show = legend_kwds.pop("show", None)
        if show is None:
            item_dim = _get_item_dim(ds)
            num_items = len(ds[item_dim])
            if num_items > 10:
                warnings.warn(
                    "More than 10 items in legend; setting legend=False; "
                    "set legend=True to show legend."
                )
                ds.attrs["legend_kwds"]["show"] = False
            elif num_items == 1:
                ds.attrs["legend_kwds"]["show"] = False
            else:
                ds.attrs["legend_kwds"]["show"] = True
        return ds

    def _config_chart(self, ds, chart):
        preset = ds.attrs["preset_kwds"].get("preset", "")
        if preset != "" and chart not in PRESETS[preset]:
            raise ValueError(f"{preset} preset is not supported for {chart} charts")

        if "morph" in preset:
            ds = self._config_morph_chart(ds, chart)

        if chart.startswith("bar"):
            ds = self._config_bar_chart(ds, chart, preset)
        elif chart == "pie":
            ds = self._config_pie_chart(ds)
        elif "trail" in preset:
            ds = self._config_trail_chart(ds, preset)

        ds = self._config_grid_axes(ds, chart)
        ds = self._config_legend(ds)
        return ds

    def _add_figsize(self, ds):
        figure_kwds = ds.attrs["figure_kwds"]
        if figure_kwds.get("figsize") is None:
            width = 7.5 + 7.5 * (self.num_cols - 1)
            height = 5 + 5 * (self.num_rows - 1)
            figsize = (width, height)
            ds.attrs["figure_kwds"]["figsize"] = figsize
        return ds

    @staticmethod
    def _fill_null(ds):
        for var in ds.data_vars:
            if ds[var].dtype == "O":
                try:
                    ds[var] = ds[var].astype(float)
                except ValueError:
                    pass
        return ds

    @staticmethod
    def _get_median_diff(array):
        array = np.atleast_1d(array)
        if len(array) == 1:
            return array
        nan_indices = np.where(np.isnan(array))
        array[nan_indices] = array.ravel()[0]
        array = np.unique(array)
        if array.ndim > 1 and array.shape[-1] > 1:
            base_diff = np.nanmedian(np.diff(array, axis=1))
        else:
            base_diff = np.nanmedian(np.diff(array.ravel()))
        base_diff = np.abs(base_diff)
        return base_diff

    def _add_xy01_limits(self, ds, chart):
        if chart == "pie":
            return ds

        # TODO: breakdown function
        limits = {key: ds.attrs["limits_kwds"].pop(key, None) for key in ITEMS["limit"]}

        for axis in ["x", "y"]:
            axis_lim = limits.pop(f"{axis}lims", None)
            if axis_lim is None:
                continue
            elif not isinstance(axis_lim, str) and len(axis_lim) != 2:
                raise ValueError(
                    f"{axis_lim} must be a string or tuple, got "
                    f"{axis_lim}; for moving limits, set {axis}lim0s "
                    f"and {axis}lim1s instead!"
                )

            axis_lim0 = f"{axis}lim0s"
            axis_lim1 = f"{axis}lim1s"

            has_axis_lim0 = limits[axis_lim0] is not None
            has_axis_lim1 = limits[axis_lim1] is not None
            if has_axis_lim0 or has_axis_lim1:
                warnings.warn(
                    "Overwriting {axis_lim0} and {axis_lim1} "
                    "with set {axis_lim} {axis_lim}!"
                )
            if isinstance(axis_lim, str):
                limits[axis_lim0] = axis_lim
                limits[axis_lim1] = axis_lim
            else:
                limits[axis_lim0] = axis_lim[0]
                limits[axis_lim1] = axis_lim[1]

        if ds.attrs["limits_kwds"].get("worldwide") is None:
            if any(limit is not None for limit in limits.values()):
                ds.attrs["limits_kwds"]["worldwide"] = False

        if ds.attrs["limits_kwds"].get("worldwide"):
            return ds  # ax.set_global() will be called in animation.py

        if chart == "barh":
            swap_letter = (
                lambda key: f"y{key[1:]}" if key.startswith("x") else f"x{key[1:]}"
            )
            limits = {swap_letter(key): limit for key, limit in limits.items()}

        item_dim = _get_item_dim(ds)
        axes_kwds = load_defaults("axes_kwds", ds)
        for key, limit in limits.items():
            # example: xlim0s
            key = key.rstrip("s")
            axis = key[0]  # x
            num = int(key[-1])  # 0
            is_lower_limit = num == 0

            axis_limit_key = f"{axis}lim"
            if axes_kwds is not None:
                in_axes_kwds = axis_limit_key in axes_kwds
            else:
                in_axes_kwds = False
            unset_limit = limit is None and not in_axes_kwds
            auto_zero = False
            if in_axes_kwds:
                limit = axes_kwds[axis_limit_key][num]
            elif unset_limit:
                has_other_limit = limits[f"{key[:-1]}{1 - num}s"] is not None
                is_scatter = chart == "scatter"
                is_line_y = chart in ITEMS["continual_charts"] and axis == "y"
                is_bar_x = chart.startswith("bar") and axis == "x"
                is_bar_y = chart.startswith("bar") and axis == "y"
                is_fixed = any([is_scatter, is_line_y, is_bar_y, has_other_limit])
                if is_bar_y and is_lower_limit:
                    limit = "zero"
                    auto_zero = True
                elif is_bar_x:
                    limit = "fixed"
                elif item_dim == "grid_item" or is_fixed:
                    limit = "fixed"
                elif num == 1:
                    limit = "explore"
                else:
                    limit = "explore"

            if isinstance(limit, str):
                if "_" in limit:
                    limit, padding = limit.split("_")
                    try:
                        padding = float(padding)
                    except ValueError:
                        padding = pd.to_timedelta(padding)
                else:
                    padding = 0
                if limit not in OPTIONS["limit"]:
                    raise ValueError(
                        f"Got {limit} for {key}; must be either "
                        f"from {OPTIONS['limit']} or numeric values!"
                    )

                var = item_dim.replace("item", axis)
                if item_dim == "ref_item":
                    for n in [num, 0, 1]:
                        ref_var = f"{var}{n}"
                        if ref_var in ds.data_vars:
                            var = ref_var
                            break
                    else:
                        continue

                da = ds[var]
                if is_datetime(da) or is_timedelta(da):
                    da = fillna(da, how="both")

                if limit == "zero":
                    min_val = da.min().values
                    if auto_zero and min_val < 0:
                        limit = min_val
                        padding = 0.05
                    else:
                        limit = 0
                elif limit == "fixed":
                    # fixed bounds, da.min()
                    stat = "min" if is_lower_limit else "max"
                    limit = getattr(da, stat)().values
                elif limit == "explore":
                    stat = "min" if is_lower_limit else "max"
                    # explore bounds, pd.Series(da.min('item')).cummin()
                    limit = getattr(
                        pd.Series(
                            getattr(fillna(ds[var], dim=item_dim), stat)(
                                item_dim
                            ).values
                        ),
                        f"cum{stat}",
                    )().values
                elif limit == "follow":
                    # follow latest state, da.min('item')
                    stat = "min" if is_lower_limit else "max"
                    limit = getattr(da, stat)(item_dim).values

                if limit is not None and padding != 0:
                    if num == 0:
                        # if I try limit -= offset, UFuncTypeError
                        limit = limit - padding
                    else:
                        limit = limit + padding

            if limit is not None:

                # pad bar charts
                if chart.startswith("bar") and axis == "x" and not is_str(limit):
                    if is_lower_limit:
                        limit -= 0.5
                    else:
                        limit += 0.5

                if chart == "barh":
                    axis = "x" if axis == "y" else "y"
                    key = axis + key[1:]

                if is_scalar(limit) == 1:
                    limit = np.repeat(limit, len(ds["state"]))
                ds[key] = ("state", limit)

        return ds

    def _compress_vars(self, da):
        if isinstance(da, xr.Dataset):
            attrs = da.attrs  # keep_attrs raises an error
            da = da.map(self._compress_vars)
            da.attrs = attrs
            if "state" not in da.dims:
                da = da.expand_dims("state").transpose(..., "state")
            return da
        elif da.name in ["x", "y"]:
            return da

        item_dim = _get_item_dim(da)
        if item_dim and len(da[item_dim]) > 2:
            return da

        vals = da.values.ravel()
        try:
            null_vals = pd.isnull(vals)
            unique_vals = np.unique(vals[~null_vals])

            items = da[item_dim].values
            if (
                len(unique_vals) == 1
                and len(vals) > 1
                and len(items) == 1
                and len(null_vals) == 0
            ):
                dim = da.dims[0]
                if dim != "state":
                    return xr.DataArray(
                        unique_vals[0], dims=(item_dim,), coords={dim: [items[0]]}
                    )
                else:
                    return unique_vals[0]
            else:
                return da
        except Exception:
            return da

    @staticmethod
    def _add_color_kwds(ds, chart):
        if chart is not None:
            if chart.startswith("bar"):
                if set(np.unique(ds["label"].astype(str))) == set(np.unique(ds["x"])):
                    ds.attrs["legend_kwds"]["show"] = False

        if "c" in ds and np.issubdtype(ds["c"], np.number):
            c_var = "c"
            plot_key = "plot_kwds"
        else:
            c_var = "grid_c"
            plot_key = "grid_plot_kwds"

        cticks_kwds = load_defaults("cticks_kwds", ds)
        colorbar_kwds = load_defaults("colorbar_kwds", ds)
        if c_var in ds:
            cticks = cticks_kwds.get("ticks")
            if cticks is None:
                num_colors = DEFAULTS["cticks_kwds"]["num_colors"]
            else:
                num_colors = len(cticks) - 1
            if "num_colors" in cticks_kwds and chart == "contourf":
                warnings.warn("num_colors is ignored for contourf!")
            num_colors = cticks_kwds.pop("num_colors", num_colors)
            if num_colors < 2:
                raise ValueError("There must be at least 2 colors!")

            if "cmap" in ds.data_vars:
                cmap = pop(ds, "cmap", get=-1)
            elif "ref_cmap" in ds.data_vars:
                cmap = pop(ds, "ref_cmap", get=-1)
            elif "grid_cmap" in ds.data_vars:
                cmap = pop(ds, "grid_cmap", get=-1)
            else:
                cmap = ds.attrs[plot_key].get("cmap", "RdYlBu_r")
            ds.attrs[plot_key]["cmap"] = plt.get_cmap(cmap, num_colors)

            vmin = pop(ds, "vmin", get=-1)
            vmax = pop(ds, "vmax", get=-1)
            if cticks is None:
                num_ticks = cticks_kwds.get("num_ticks", num_colors + 1)
                if vmin is None:
                    vmin = np.nanmin(ds[c_var].values)
                if vmax is None:
                    vmax = np.nanmax(ds[c_var].values)
                indices = np.round(np.linspace(0, num_ticks - 1, num_ticks)).astype(
                    int
                )  # select 10 values equally
                cticks = np.linspace(vmin, vmax, num_ticks)[indices]
                ds.attrs["cticks_kwds"]["ticks"] = cticks
                ds.attrs[plot_key]["vmin"] = vmin
                ds.attrs[plot_key]["vmax"] = vmax
            else:
                ds.attrs[plot_key]["norm"] = ds.attrs[plot_key].get(
                    "norm", BoundaryNorm(cticks, num_colors)
                )

            ds.attrs["colorbar_kwds"]["show"] = colorbar_kwds.get("show", True)
        elif "colorbar_kwds" in ds.attrs:
            ds.attrs["colorbar_kwds"]["show"] = False
        return ds

    @staticmethod
    def _precompute_base_ticks(ds, base_kwds):
        # for x y c
        for xyc in ITEMS["axes"]:
            if xyc in ds:
                if is_str(ds[xyc]):
                    ds.attrs[f"{xyc}ticks_kwds"]["is_str"] = True
                    continue
                elif "c" not in xyc:
                    ds.attrs[f"{xyc}ticks_kwds"]["is_str"] = False

                if "is_bar" in ds[xyc].attrs:
                    base_kwds[f"{xyc}ticks"] = np.nanmedian(ds[xyc])
                else:
                    try:
                        base_kwds[f"{xyc}ticks"] = np.nanmedian(ds[xyc]) / 10
                    except TypeError:
                        base_kwds[f"{xyc}ticks"] = np.nanmin(ds[xyc])

                if "c" in xyc:
                    continue

                ds.attrs[f"{xyc}ticks_kwds"]["is_datetime"] = is_datetime(ds[xyc])

        return ds, base_kwds

    def _precompute_base_labels(self, ds, chart, base_kwds):
        # for inline_label, state_label, etc
        for key in ITEMS["base"]:
            key_label = f"{key}_label"
            base = None
            if key_label in ds:
                try:
                    if is_scalar(ds[key_label]):
                        base = np.nanmin(ds[key_label]) / 10
                    else:
                        key_values = ds[key_label].values
                        if is_timedelta(key_values):
                            base = key_values[0]
                        elif not is_str(key_values):
                            base_diff = self._get_median_diff(key_values)
                            if is_datetime(base):
                                base = np.nanmin(base_diff) / 5
                            else:
                                base = np.nanquantile(base_diff, 0.25)
                    if not pd.isnull(base):
                        base_kwds[key] = base
                except Exception as e:
                    if self.debug:
                        warnings.warn(str(e))
        return ds, base_kwds

    def _precompute_base(self, ds, chart):
        base_kwds = {}
        ds, base_kwds = self._precompute_base_ticks(ds, base_kwds)
        ds, base_kwds = self._precompute_base_labels(ds, chart, base_kwds)
        if "s" in ds.data_vars:
            base_kwds["s"] = np.nanmedian(ds["s"])

        ds.attrs["base_kwds"] = base_kwds
        return ds

    @staticmethod
    def _compute_padding(lower, upper, padding=None, log=False):
        """
        Pads the range by a fraction of the interval

        Adapted from holoviews
        https://holoviews.org/_modules/holoviews/core/util.html
        """
        if padding is not None and not isinstance(padding, tuple):
            padding = (padding, padding)

        are_numeric = is_numeric(lower) and is_numeric(upper)
        are_datetime = is_datetime(lower) and is_datetime(upper)
        if (are_numeric or are_datetime) and padding is not None:
            if not is_datetime(lower) and log and lower > 0 and upper > 0:
                log_min = np.log(lower) / np.log(10)
                log_max = np.log(upper) / np.log(10)
                lspan = (log_max - log_min) * (1 + padding[0] * 2)
                uspan = (log_max - log_min) * (1 + padding[1] * 2)
                center = (log_min + log_max) / 2.0
                start, end = np.power(10, center - lspan / 2.0), np.power(
                    10, center + uspan / 2.0
                )
            else:
                if is_datetime(lower):
                    # Ensure timedelta can be safely divided
                    span = (upper - lower).astype(">m8[ns]")
                else:
                    span = upper - lower
                lpad = span * (padding[0])
                upad = span * (padding[1])
                start, end = lower - lpad, upper + upad
        else:
            start, end = lower, upper

        return start, end

    def _add_margins(self, ds, chart):
        if chart == "pie":
            return ds

        margins_kwds = load_defaults("margins_kwds", ds)

        for axis in ["x", "y"]:
            axis_margins = margins_kwds.pop(axis, None)
            if axis_margins is None:
                continue
            elif axis == "y" and chart.startswith("bar"):
                if axis_margins == DEFAULTS["margins_kwds"]["y"]:
                    axis_margins = 0

            axis_lim0 = f"{axis}lim0"
            axis_lim1 = f"{axis}lim1"
            try:
                ds[axis_lim0], ds[axis_lim1] = self._compute_padding(
                    ds[axis_lim0], ds[axis_lim1], axis_margins
                )
            except KeyError:
                pass

        return ds

    def _add_durations(self, ds):
        if "fps" in ds.attrs["animate_kwds"]:
            return ds

        num_states = len(ds["state"])

        durations_kwds = load_defaults("durations_kwds", ds)
        transition_frames = durations_kwds.pop("transition_frames")
        aggregate = durations_kwds.pop("aggregate")

        durations = durations_kwds.get(
            "durations", 0.5 if num_states < 8 else transition_frames
        )
        if isinstance(durations, (int, float)):
            durations = np.repeat(durations, num_states)

        if np.isnan(durations[-1]):
            durations[-1] = 0
        durations[-1] += durations_kwds["final_frame"]

        if "duration" in ds:
            try:
                ds["duration"] = ("state", durations + ds["duration"].values)
            except ValueError:  # incompatible
                warnings.warn("Setting duration is in incompatible with morph!")
                ds["duration"] = ("state", durations)
        else:
            ds["duration"] = ("state", durations)

        ds["duration"].attrs["transition_frames"] = transition_frames
        ds["duration"].attrs["aggregate"] = aggregate
        return ds

    def _interp_dataset(self, ds):
        ds = ds.map(self.interpolate, keep_attrs=True)

        if "s" in ds:
            ds["s"] = fillna(ds["s"].where(ds["s"] >= 0), how="both")

        if "state" not in ds.dims:
            ds = ds.expand_dims("state").transpose(..., "state")

        num_states = len(ds["state"])
        ds["state"] = srange(num_states)
        return ds

    def _get_crs(self, crs_obj, crs_kwds, central_longitude=None):
        if isinstance(crs_obj, bool):
            crs_obj = "PlateCarree"

        if isinstance(crs_obj, str):
            import cartopy.crs as ccrs

            if len(self._crs_names) == 0:
                self._crs_names.update(
                    {
                        name.lower(): name
                        for name, obj in vars(ccrs).items()
                        if isinstance(obj, type)
                        and issubclass(obj, ccrs.Projection)
                        and not name.startswith("_")
                        and name not in ["Projection"]
                        or name == "GOOGLE_MERCATOR"
                    }
                )

            if central_longitude is not None:
                crs_kwds["central_longitude"] = central_longitude

            crs_obj = getattr(ccrs, self._crs_names[crs_obj.lower()])
            if callable(crs_obj):  # else ccrs.GOOGLE_MERCATOR
                crs_obj = crs_obj(**crs_kwds)
        else:
            if central_longitude is not None:
                raise ValueError(
                    f"central_longitude only supported if "
                    f"projection is a string type; got {crs_obj}"
                )
        return crs_obj

    def _add_geo_transforms(self, ds, chart):
        crs_kwds = load_defaults("crs_kwds", ds)
        crs = crs_kwds.pop("crs", None)

        projection_kwds = load_defaults("projection_kwds", ds)
        projection = projection_kwds.pop("projection", None)

        tiles_kwds = load_defaults("tiles_kwds", ds)
        tiles = tiles_kwds.pop("tiles", None)
        geo_features = {
            geo for geo in CONFIGURABLES["geo"] if ds.attrs[f"{geo}_kwds"].get(geo)
        }
        if len(geo_features) > 0:
            projection = projection or ("GOOGLE_MERCATOR" if tiles else "PlateCarree")

        if "grid_item" not in ds.dims and "item" not in ds.dims:
            return ds
        elif crs or projection:
            if chart == "pie":
                raise ValueError(
                    "Geographic transforms are not supported for pie charts"
                )

            if len(geo_features - set(["crs", "projection"])) == 0:
                ds.attrs["coastline_kwds"]["coastline"] = True

            if "central_longitude" in ds:
                central_lon = pop(ds, "central_longitude")
            elif "central_longitude" in projection_kwds:
                central_lon = projection_kwds["central_longitude"]
            else:
                x_var = "grid_x" if "grid_x" in ds else "x"
                if x_var not in ds:
                    for x_var in VARS["ref"]:
                        if x_var in ds and "x" in x_var:
                            break
                central_lon = float(ds[x_var].mean())
                projection_kwds["central_longitude"] = central_lon

            crs = crs or "PlateCarree"
            crs_obj = self._get_crs(crs, crs_kwds)
            for key in ITEMS["transformables"]:
                if key not in ds.attrs:
                    ds.attrs[key] = {}
                ds.attrs[key]["transform"] = crs_obj

            if is_scalar(central_lon):
                projection_obj = self._get_crs(projection, projection_kwds)
                ds["projection"] = projection_obj
            else:
                projection_obj = [
                    self._get_crs(projection, projection_kwds, central_longitude=cl)
                    for cl in central_lon
                ]

                if len(central_lon) != len(ds["state"]):
                    raise ValueError(
                        f"Length of central_longitude must be scalar or "
                        f"have {len(ds['state'])} num_states!"
                    )
                ds["projection"] = "state", projection_obj
        return ds

    def _add_geo_features(self, ds):
        try:
            import cartopy.feature as cfeature
        except ImportError:
            return ds

        for feature in CONFIGURABLES["geo"]:
            if feature in ["projection", "crs", "tiles"]:
                continue
            feature_key = f"{feature}_kwds"
            feature_kwds = load_defaults(feature_key, ds)
            feature_obj = feature_kwds.pop(feature, False)
            if feature_obj:
                if isinstance(feature_obj, bool):
                    feature_obj = getattr(cfeature, feature.upper())
                ds.attrs[feature_key][feature] = feature_obj
        return ds

    def _get_zoom(self, bounds, width, height):
        """
        Compute zoom level given bounds and the plot size.
        https://github.com/holoviz/geoviews/blob/master/geoviews/util.py#L111-L136
        """
        w, e, s, n = bounds  # changed from w, s, e, n
        max_width, max_height = 256, 256
        num_states = self.num_states
        ZOOM_MAX = np.repeat(21, num_states)
        ln2 = np.log(2)
        pi = np.repeat(np.pi, num_states)

        def latRad(lat):
            sin = np.sin(lat * pi / 180)
            radX2 = np.log((1 + sin) / (1 - sin)) / 2
            return np.max([np.min([radX2, pi], axis=0), -pi], axis=0) / 2

        def zoom(mapPx, worldPx, fraction):
            return np.floor(np.log(mapPx / worldPx / fraction) / ln2)

        latFraction = (latRad(n) - latRad(s)) / pi

        lngDiff = e - w
        lngFraction = np.where(lngDiff < 0, lngDiff + 360, lngDiff / 360)

        latZoom = zoom(height, max_height, latFraction)
        lngZoom = zoom(width, max_width, lngFraction)
        zoom = np.min([latZoom, lngZoom, ZOOM_MAX], axis=0)
        return np.where(np.isfinite(zoom), zoom, 0).astype(int)

    def _add_geo_tiles(self, ds):
        figure_kwds = load_defaults("figure_kwds", ds)
        tiles_kwds = load_defaults("tiles_kwds", ds)
        tiles_obj = tiles_kwds.pop("tiles", False)
        style = tiles_kwds.pop("style", None)
        if tiles_obj:
            import cartopy
            import cartopy.io.img_tiles as ctiles

            cartopy_version = cartopy.__version__
            if cartopy_version < "0.19.0":
                raise ValueError(
                    f"To use tiles, ensure cartopy>=0.19.0; got {cartopy_version}"
                )

            if len(self._tiles_names) == 0:
                self._tiles_names = {
                    name.lower(): name
                    for name, obj in vars(ctiles).items()
                    if isinstance(obj, type)
                    and issubclass(obj, ctiles.GoogleWTS)
                    and not name.startswith("_")
                }

            zoom = tiles_kwds.pop("zoom", None)
            if zoom is None:
                if ds.attrs["limits_kwds"].get("worldwide"):
                    xlim0s = -179
                    xlim1s = 179
                    ylim0s = -89
                    ylim1s = 89
                else:
                    xlim0s = ds["xlim0"].values
                    xlim1s = ds["xlim1"].values
                    ylim0s = ds["ylim0"].values
                    ylim1s = ds["ylim1"].values
                bounds = np.vstack(
                    [
                        self._adapt_input(xlim0s),
                        self._adapt_input(xlim1s),
                        self._adapt_input(ylim0s),
                        self._adapt_input(ylim1s),
                    ]
                )
                width, height = np.array(figure_kwds["figsize"]) * figure_kwds.get(
                    "dpi", 75
                )
                zoom = self._get_zoom(bounds, width, height)
            else:
                zoom = self._adapt_input(zoom, reshape=False)
            ds["zoom"] = ("state", zoom)

            if isinstance(tiles_obj, bool):
                tiles_obj = "OSM"

            if isinstance(tiles_obj, str):
                tiles_obj = self._tiles_names[tiles_obj.lower()]
                try:
                    tiles_obj = getattr(ctiles, tiles_obj)(style=style, cache=True)
                except TypeError:
                    tiles_obj = getattr(ctiles, tiles_obj)(cache=True)
                ds.attrs["tiles_kwds"]["tiles"] = tiles_obj
        return ds

    def _add_animate_kwds(self, ds):
        animate_kwds = {}

        # after interpolation, haven't updated self.num_states
        num_states = len(ds["state"])
        if isinstance(self.animate, str):
            if "_" in self.animate:
                animate, value = self.animate.split("_")
                value = int(value)
            else:
                animate = self.animate
                value = 11

            if num_states <= 10:
                animate_kwds["states"] = None
            elif animate in ["head", "ini", "start"]:
                animate_kwds["states"] = np.arange(1, value)
            elif animate in ["tail", "end", "value"]:
                animate_kwds["states"] = np.arange(-value, 0, 1)
            else:
                animate_kwds["states"] = np.linspace(1, num_states, value).astype(int)

            if "fps" not in ds.attrs["animate_kwds"]:
                animate_kwds["fps"] = 1
            animate_kwds["stitch"] = True
            animate_kwds["static"] = False
        elif isinstance(self.animate, slice):
            start = self.animate.start or 1
            stop = self.animate.stop
            step = self.animate.step or 1
            animate_kwds["states"] = np.arange(start, stop, step)
            animate_kwds["stitch"] = True
            animate_kwds["static"] = is_scalar(animate_kwds["states"])
        elif isinstance(self.animate, bool):
            animate_kwds["states"] = None
            animate_kwds["stitch"] = self.animate
            animate_kwds["static"] = False
        elif isinstance(self.animate, (Iterable, int)):
            animate_kwds["states"] = to_1d(self.animate, flat=False)
            animate_kwds["stitch"] = True
            negative_indices = animate_kwds["states"] < 0
            animate_kwds["states"][negative_indices] = (
                num_states - animate_kwds["states"][negative_indices]
            )
            if animate_kwds["states"][0] == 0:
                warnings.warn("State 0 detected in animate; shifting by 1.")
                animate_kwds["states"] += 1
            animate_kwds["static"] = True if isinstance(self.animate, int) else False
        ds.attrs["animate_kwds"].update(**animate_kwds)
        return ds

    def _get_chart(self, ds):
        for kind in ["", "ref_", "grid_"]:
            chart_key = f"{kind}chart"
            if chart_key in ds.data_vars:
                break
        else:
            return ""
        try:
            chart = to_scalar(ds[chart_key], get=-1)
        except IndexError:
            chart = "line"
        return chart

    def finalize(self):
        if all(ds.attrs.get("finalized", False) for ds in self.data.values()):
            return self

        data = {}
        self_copy = self.copy()
        for rowcol, ds in self_copy.data.items():
            chart = self._get_chart(ds)
            ds = self_copy._add_figsize(ds)
            ds = self_copy._fill_null(ds)
            ds = self_copy._add_xy01_limits(ds, chart)
            ds = self_copy._add_color_kwds(ds, chart)
            ds = self_copy._config_chart(ds, chart)
            ds = self_copy._add_margins(ds, chart)  # must be after config chart
            ds = self_copy._add_durations(ds)
            ds = self_copy._precompute_base(ds, chart)  # must be after config chart
            ds = self_copy._add_geo_tiles(ds)  # before interp
            ds = self_copy._interp_dataset(ds)
            ds = self_copy._add_geo_transforms(ds, chart)  # after interp
            ds = self_copy._add_geo_features(ds)
            ds = self_copy._add_animate_kwds(ds)
            ds = self_copy._compress_vars(ds)
            ds.attrs["finalized"] = True
            data[rowcol] = ds

        self_copy.data = data
        return self_copy

    def _set_input_vars(self, **kwds):
        # TODO: add test
        self._input_vars = {}
        for key in list(kwds):
            if key in self._parameters:
                continue
            val = kwds.pop(key)
            if not isinstance(val, tuple):
                val = np.array(val)
            self._input_vars[key] = val

        for key in list(self._input_vars.keys()):
            if key == "c":
                # won't show up if
                continue
            key_and_s = key + "s"
            key_strip = key.rstrip("s")
            if key_and_s in self._parameters and key_and_s != key:
                raise KeyError(
                    f"Invalid param: '{key}', replace with expected '{key_and_s}'"
                )
            elif key_strip in self._parameters and key_strip != key:
                raise KeyError(
                    f"Invalid param: '{key}', replace with expected '{key_strip}'"
                )

        for lim in ITEMS["limit"]:
            if lim in self._input_vars:
                raise KeyError(
                    f"Invalid param: '{key}', replace with expected '{key_strip}'"
                )

        return kwds

    def _adapt_input(self, val, num_items=None, reshape=True, shape=None):
        # TODO: add test
        num_states = self.num_states
        val = np.array(val)
        ndim = val.ndim

        if is_scalar(val):
            val = np.repeat(val, num_states)
        elif ndim > 1:
            other_lengths = np.sum(val.shape[1:])
            if other_lengths == 1:
                val = val.squeeze()
                ndim = val.ndim

        if not is_datetime(val):
            # make string '1' into 1
            try:
                val = val.astype(float)
            except (ValueError, TypeError):
                pass

        if reshape:
            try:
                if ndim == 3:
                    val = val.reshape(-1, *val.shape)
                elif ndim == 2 and val.shape[0] > 1:
                    val = val.reshape(-1, num_states, *val.shape)
                elif shape is not None:
                    val = val.reshape(-1, num_states, *shape)
                else:
                    val = val.reshape(-1, num_states)
            except ValueError:
                pass

        if num_items is not None and val.shape[0] != num_items:
            val = np.tile(val, (num_items, 1))
        return val

    def _load_dataset(self, **kwds):
        # TODO: add test
        input_vars = self._input_vars
        num_states = self.num_states
        label = self.label or ""
        group = self.group or ""

        if self.chart is None:
            if num_states < 8 or "s" in input_vars:
                chart = "scatter"
            else:
                chart = "line"
        else:
            chart = self.chart

        try:
            if np.array(input_vars.get("ys", None)).item() is None:
                input_vars["ys"] = input_vars["xs"]
                input_vars["xs"] = np.arange(num_states)
        except (ValueError, KeyError):
            # ValueError no xs for ref
            pass  # ref or grid

        attrs = {"plot_kwds": {}, "grid_plot_kwds": {}, "ref_plot_kwds": {}}
        coords = {}
        data_vars = {}
        dims = DIMS[self._dim_type]
        for key, val in input_vars.items():
            if len(key) > 1 and key.endswith("s"):  # exclude s / size
                key = key[:-1]
            data_vars[key] = val

        num_items = 1
        for var in data_vars:
            try:
                if data_vars[var].ndim > 1 and var.startswith(("x", "y")):
                    shape = data_vars[var].shape
                    num_items = max(shape[0], num_items)
            except Exception:
                pass

        if self._dim_type == "grid":
            num_ys = len(input_vars["ys"])
            num_xs = len(input_vars["xs"])

        plot_key = f"{self._dim_type}_plot_kwds".replace("basic_", "")
        for var in list(data_vars.keys()):
            val = data_vars.pop(var)
            if isinstance(val, tuple) and var not in ["x", "y"]:
                attrs[plot_key][var] = val
            elif self._dim_type == "grid" and var in ["x", "y"]:
                coords[var] = val
            elif self._dim_type == "grid" and is_scalar(val):
                shape = (num_items, self.num_states, num_ys, num_xs)
                data_vars[var] = dims, np.full(shape, val)
            else:
                data_vars[var] = dims, self._adapt_input(val, num_items=num_items)

        dims = DIMS["basic"] if self._dim_type == "grid" else dims

        if self.state_labels is not None:
            state_labels = self._adapt_input(self.state_labels, reshape=False)
            data_vars["state_label"] = ("state", state_labels)

        if self.inline_labels is not None:
            inline_labels = self._adapt_input(self.inline_labels, num_items=num_items)
            data_vars["inline_label"] = dims, inline_labels

        # pass unique for dataframe
        data_vars["chart"] = (
            dims,
            self._adapt_input(chart, num_items=num_items),
        )
        data_vars["label"] = (
            dims,
            self._adapt_input(label, num_items=num_items),
        )
        data_vars["group"] = (
            dims,
            self._adapt_input(group, num_items=num_items),
        )

        coords.update({dims[0]: srange(num_items), "state": srange(num_states)})
        ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        ds = _drop_state(ds)
        self._ds = ds

    @staticmethod
    def _propagate_params(self_copy, other, layout=False):
        self_copy.configurables.update(**other.configurables)
        for param_ in self_copy._parameters:
            not_canvas_param = param_ not in CANVAS
            if callable(param_) or (layout and not_canvas_param):
                continue

            try:
                self_value = getattr(self_copy, param_)
                other_value = getattr(other, param_)
            except AttributeError:
                continue

            try:
                self_null = self_value in NULL_VALS
            except ValueError:
                self_null = False

            try:
                other_null = other_value in NULL_VALS
            except ValueError:
                other_null = False

            if param_ in PARAMS:
                if self_null and not other_null:
                    setattr(self_copy, param_, other_value)
                    self_copy = self_copy.config(PARAMS[param_])
                elif not self_null and other_null:
                    self_copy = self_copy.config(PARAMS[param_])
        return self_copy


class GeographicData(Data):

    crs = CartopyCRS(
        doc="The coordinate reference system to project from",
        precedence=PRECEDENCES["geo"],
    )
    projection = CartopyCRS(
        doc="The coordinate reference system to project to",
        precedence=PRECEDENCES["geo"],
    )
    central_lon = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Longitude to center the map on",
        precedence=PRECEDENCES["geo"],
    )

    borders = CartopyFeature(
        doc="Whether to show borders", precedence=PRECEDENCES["geo"]
    )
    coastline = CartopyFeature(
        doc="Whether to show coastlines", precedence=PRECEDENCES["geo"]
    )
    land = CartopyFeature(
        doc="Whether to show land surfaces", precedence=PRECEDENCES["geo"]
    )
    ocean = CartopyFeature(
        doc="Whether to show ocean surfaces", precedence=PRECEDENCES["geo"]
    )
    lakes = CartopyFeature(doc="Whether to show lakes", precedence=PRECEDENCES["geo"])
    rivers = CartopyFeature(doc="Whether to show rivers", precedence=PRECEDENCES["geo"])
    states = CartopyFeature(doc="Whether to show states", precedence=PRECEDENCES["geo"])
    worldwide = param.Boolean(
        default=None, doc="Whether to view globally", precedence=PRECEDENCES["geo"]
    )

    tiles = CartopyTiles(doc="Whether to show web tiles", precedence=PRECEDENCES["geo"])
    zoom = param.Integer(
        default=None,
        bounds=(0, 25),
        doc="Zoom level of tiles",
        precedence=PRECEDENCES["geo"],
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.configurables["geo"] = CONFIGURABLES["geo"]

    def _config_rotate_chart(self, ds):
        num_states = self.num_states
        x_dim = "grid_x" if "grid_x" in ds else "x"
        central_lon = ds.attrs["projection_kwds"].get(
            "central_longitude", ds[x_dim].min()
        )
        if is_scalar(central_lon):
            central_lon_end = ds[x_dim].max()
            central_lons = np.linspace(central_lon, central_lon_end, num_states)
        elif length(central_lon) != num_states:
            central_lons = np.linspace(
                np.min(central_lon), np.max(central_lon), num_states
            )
        else:
            central_lons = central_lon
        ds["central_longitude"] = ("state", central_lons)
        if "projection" not in ds.attrs["projection_kwds"]:
            ds.attrs["projection_kwds"]["projection"] = "Orthographic"
        return ds

    def _config_chart(self, ds, chart):
        ds = super()._config_chart(ds, chart)
        preset = ds.attrs["preset_kwds"].get("preset", "")
        if "rotate" in preset:
            ds = self._config_rotate_chart(ds)
        return ds


class ReferenceArray(param.Parameterized):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.configurables["ref"] = CONFIGURABLES["ref"]

    def reference(
        self,
        x0s=None,
        x1s=None,
        y0s=None,
        y1s=None,
        label=None,
        inline_labels=None,
        inline_locs=None,
        rowcols=None,
        last=False,
        **kwds,
    ):
        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = self.copy()
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            kwds.update(
                {
                    "x0s": x0s,
                    "x1s": x1s,
                    "y0s": y0s,
                    "y1s": y1s,
                    "label": label,
                    "inline_labels": inline_labels,
                    "inline_locs": inline_locs,
                }
            )

            for key in list(kwds):
                kwds[key] = get(ds, kwds[key])

            if inline_locs is None:
                has_x0s = kwds["x0s"] is not None
                has_y0s = kwds["y0s"] is not None
                if has_x0s and not has_y0s:
                    inline_locs = to_scalar(ds["y"].values)
                elif has_y0s and not has_x0s:
                    inline_locs = to_scalar(ds["x"].values)
                if not isinstance(inline_locs, str):
                    kwds["inline_locs"] = inline_locs

            self_copy *= Reference(**kwds)

        data = {}
        for rowcol, ds in self_copy.items():
            ds.attrs["ref_plot_kwds"]["last"] = last
            for var in ["ref_x0", "ref_x1", "ref_y0", "ref_y1"]:
                if var in ds.data_vars:
                    last_item = ~np.isnan(ds[var])
                    last_item = last_item.sum("ref_item").values
                    ds["ref_last_item"] = ("state", last_item)
                    break
            data[rowcol] = ds
        self_copy.data = data

        return self_copy


class ColorArray(param.Parameterized):

    cs = param.ClassSelector(
        class_=(Iterable,),
        doc="Array to be mapped to the colorbar",
        precedence=PRECEDENCES["xyc"],
    )

    cticks = param.ClassSelector(
        class_=(Iterable,),
        doc="Colorbar tick locations",
        precedence=PRECEDENCES["limit"],
    )
    ctick_labels = param.ClassSelector(
        class_=(Iterable,),
        doc="Colorbar tick labels",
        precedence=PRECEDENCES["sub_label"],
    )
    colorbar = param.Boolean(
        default=None, doc="Whether to show colorbar", precedence=PRECEDENCES["style"]
    )
    clabel = param.ClassSelector(
        class_=(int, float, str),
        allow_None=True,
        doc="Colorbar label",
        precedence=PRECEDENCES["label"],
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.configurables["color"] = CONFIGURABLES["color"]


class RemarkArray(param.Parameterized):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.configurables["remark"] = CONFIGURABLES["remark"]

    def _match_values(self, da, values, first, rtol, atol, condition):
        if is_datetime(da) or is_timedelta(da):
            da = da.astype(float)
            if hasattr(values, "values"):
                values = values.values
            elif isinstance(values, (str, list)):
                values = pd.to_datetime(to_1d(values)).values
            values = values.astype(float)
        values = np.array(to_1d(values))  # np.array required or else crash

        try:
            diff = np.abs(da.expand_dims("match").transpose(..., "match") - values)
            ctol = atol + rtol * np.abs(values)  # combined tol
            new_condition = (diff <= ctol).any("match")

            if first:
                # + 1 because state starts counting at 1
                da_masked = da.where(new_condition)
                new_condition = (
                    xr.concat(
                        (
                            da_masked.where(
                                da_masked["state"]
                                == (da_masked >= value).argmax("state") + 1
                            )
                            for value in values
                        ),
                        "match",
                    )
                    .sum("match")
                    .astype(bool)
                )
        except TypeError as e:
            if self.debug:
                warnings.warn(e)
            new_condition = da.isin(values)

        if hasattr(new_condition, "values"):
            new_condition = new_condition.values

        condition = condition & new_condition
        return condition

    def remark(
        self,
        remarks=None,
        durations=None,
        condition=None,
        xs=None,
        ys=None,
        cs=None,
        labels=None,
        state_labels=None,
        inline_labels=None,
        first=False,
        rtol=1e-8,
        atol=1e-5,
        rowcols=None,
        persist_plot=None,
        persist_inline=None,
        **other_vars,
    ):
        args = (
            xs,
            ys,
            cs,
            labels,
            state_labels,
            inline_labels,
            condition,
            *list(other_vars.values()),
        )
        args_none = sum([1 for arg in args if arg is None])
        if args_none == len(args):
            raise ValueError(
                "Must supply either xs, ys, cs, state_labels, "
                "inline_labels, condition, or other_vars!"
            )

        if durations is None and remarks is None:
            raise ValueError("Must supply at least remarks or durations!")
        elif durations is None:
            durations = 2

        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = self.copy()
        data = {}
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            if condition is None:
                condition = xr.full_like(ds["y"], True, dtype=bool).values
            else:
                condition = self_copy._adapt_input(condition)

            if xs is not None:
                condition = self_copy._match_values(
                    ds["x"], xs, first, rtol, atol, condition
                )
            if ys is not None:
                condition = self_copy._match_values(
                    ds["y"], ys, first, rtol, atol, condition
                )
            if cs is not None:
                condition = self_copy._match_values(
                    ds["c"], cs, first, rtol, atol, condition
                )
            if labels is not None:
                condition = self_copy._match_values(
                    ds["label"], labels, first, rtol, atol, condition
                )
            if state_labels is not None:
                condition = self_copy._match_values(
                    ds["state_label"], state_labels, first, rtol, atol, condition
                )
            if inline_labels is not None:
                condition = self_copy._match_values(
                    ds["inline_label"], inline_labels, first, rtol, atol, condition
                )

            # condition = condition.broadcast_like(ds)  # TODO: investigate grid
            if remarks is not None:
                if "remark" not in ds:
                    ds["remark"] = (
                        DIMS["basic"],
                        np.full((len(ds["item"].values), len(ds["state"])), ""),
                    )

                remarks = get(ds, remarks, to_str=True)
                ds["remark"] = xr.where(condition, remarks, ds["remark"])

            condition = np.array(condition)
            if durations is not None:
                if "duration" not in ds:
                    ds["duration"] = (
                        "state",
                        self_copy._adapt_input(
                            np.zeros_like(ds["state"]),
                            len(ds["state"]),
                            reshape=False,
                        ),
                    )
                ds["duration"] = xr.where(
                    condition.sum(axis=0), durations, ds["duration"]
                )
                if "item" in ds["duration"].dims:
                    ds["duration"] = ds["duration"].max("item")

            ds.attrs["remark_plot_kwds"]["persist"] = persist_plot
            ds.attrs["remark_inline_kwds"]["persist"] = persist_inline

            data[rowcol] = ds.transpose(..., "state")
        self_copy.data = data
        return self_copy


class Array(GeographicData, ReferenceArray, ColorArray, RemarkArray):

    xs = param.ClassSelector(
        class_=(Iterable, int, float, str),
        doc="Array to be mapped to the x-axis",
        precedence=PRECEDENCES["xyc"],
    )
    ys = param.ClassSelector(
        class_=(Iterable, int, float, str),
        doc="Array to be mapped to the y-axis",
        precedence=PRECEDENCES["xyc"],
    )

    _dim_type = "basic"

    def __init__(self, xs, ys=None, **kwds):
        for xys, xys_arr in {"xs": xs, "ys": ys}.items():
            if isinstance(xys_arr, str):
                raise ValueError(
                    f"{xys} must be an Iterable, but cannot be str; got {xys_arr}!"
                )

        super().__init__(xs=xs, ys=ys, **kwds)
        self.data = {self.rowcol: self._ds}

    def invert(self, label=None, group=None, state_labels=None):
        data = {}
        self_copy = self.copy()
        for rowcol, ds in self_copy.data.items():
            for item_dim in ["ref_item", "grid_item"]:
                if item_dim in ds.dims:
                    raise ValueError(
                        "Cannot invert reference / grid objects; first "
                        "invert then overlay!"
                    )

            attrs = ds.attrs
            inv_ds = (
                ds.to_dataframe()
                .rename_axis(DIMS["basic"][::-1])
                .to_xarray()
                .assign_attrs(attrs)
                .transpose(*DIMS["basic"])
            )
            num_items = len(inv_ds["item"])

            if state_labels is None:
                inv_ds["state_label"] = inv_ds["label"].isel(item=0)
            elif not state_labels:
                inv_ds = inv_ds.drop("state_label", errors="ignore")
            else:
                inv_ds["state_label"] = "state", state_labels
            inv_ds = _drop_state(inv_ds)

            if label is None:
                inv_ds["label"] = "item", np.repeat("", num_items)
            else:
                inv_ds["label"] = "item", label

            if group is not None:
                if is_scalar(group):
                    group = np.repeat(group, num_items)
                inv_ds["group"] = "item", group

            data[rowcol] = inv_ds
        self_copy.data = data
        return self_copy


class Array2D(Array):

    chart = param.ObjectSelector(
        default=CHARTS["grid"][0],
        objects=CHARTS["grid"],
        doc=f"Type of chart; {CHARTS['grid']}",
        precedence=PRECEDENCES["common"],
    )

    inline_xs = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Inline label's x locations",
        precedence=PRECEDENCES["sub_label"],
    )
    inline_ys = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Inline label's y locations",
        precedence=PRECEDENCES["sub_label"],
    )

    _dim_type = "grid"

    def __init__(self, xs, ys, cs=None, **kwds):
        if cs is not None:
            kwds["cs"] = cs
        super().__init__(xs, ys, **kwds)
        self.configurables["grid"] = CONFIGURABLES["grid"]
        self.data = {self.rowcol: self._ds}

    def _load_dataset(self, **kwds):
        super()._load_dataset(**kwds)

        ds = self._ds
        if self.inline_labels is not None:
            inline_xs = self.inline_xs
            inline_ys = self.inline_ys
            if inline_xs is None or inline_ys is None:
                raise ValueError(
                    "Must provide an inline x and y if inline_labels is not None!"
                )
            else:
                ds["inline_x"] = (
                    DIMS["basic"],
                    self._adapt_input(inline_xs),
                )
                ds["inline_y"] = (
                    DIMS["basic"],
                    self._adapt_input(inline_ys),
                )

        grid_vars = list(ds.data_vars) + list(ds.coords)
        ds = ds.rename(
            {var: f"grid_{var}" for var in grid_vars if ds[var].dims != ("state",)}
        )
        self._ds = ds

    def _config_scan_chart(self, ds, preset):
        preset, axis = preset.split("_")
        grid_axis = f"grid_{axis}"
        grid_scan_axis = f"grid_scan_{axis}"
        if "state_label" in ds:
            state_labels = list(pop(ds, "state_label"))
            ds[f"grid_scan_{axis}_0_inline_label"] = ("state", state_labels)
            ds[f"grid_scan_{axis}_1_inline_label"] = (
                "state",
                np.roll(state_labels, 1),
            )
            ds[f"grid_scan_{axis}_diff_inline_label"] = (
                ds[f"grid_scan_{axis}_0_inline_label"]
                - ds[f"grid_scan_{axis}_1_inline_label"]
            )
            other_axis = "y" if axis == "x" else "x"
            ds.attrs["preset_kwds"]["inline_loc"] = ds[f"grid_{other_axis}"].median()

        scan_ds_list = []
        stateless_vars = [var for var in ds.data_vars if "state" not in ds[var].dims]
        grid_vars = [var for var in ds.data_vars if grid_axis in ds[var].dims]
        scan_stride = ds.attrs["preset_kwds"].pop("stride", 1)
        states = srange(ds["state"])[:-1]
        num_states = len(states)
        for state in states:
            curr_state_ds = ds.sel(state=state).drop_vars(stateless_vars)
            next_state_ds = ds.sel(state=state + 1).drop_vars(stateless_vars)
            for i in srange(curr_state_ds[grid_axis], stride=scan_stride):
                scan_ds = curr_state_ds.where(
                    ~curr_state_ds[grid_axis].isin(
                        next_state_ds.isel(**{grid_axis: slice(None, i)})[
                            grid_axis
                        ].values
                    ),
                    next_state_ds,
                )
                scan_ds_list.append(scan_ds)

        ds = _combine_ds_list(scan_ds_list[::-1]).assign(**ds[stateless_vars])
        for var in ds.data_vars:
            if var not in grid_vars and grid_axis in ds[var].dims:
                ds[var] = ds[var].isel(**{grid_axis: 0})
        ds[grid_scan_axis] = (
            "state",
            np.tile(ds[grid_axis][::scan_stride].values, num_states),
        )

        item_dim = _get_item_dim(ds)
        ds = ds.transpose(item_dim, "state", ...)
        return ds

    def _config_chart(self, ds, chart):
        ds = super()._config_chart(ds, chart)
        if chart not in ITEMS["uv_charts"] and "grid_c" not in ds.data_vars:
            raise ValueError("cs must be specified!")

        preset = ds.attrs["preset_kwds"].get("preset", "")
        if preset.startswith("scan"):
            ds = self._config_scan_chart(ds, preset)
        return ds


class DataStructure(Array):

    xs = param.ClassSelector(
        class_=(Iterable,),
        doc="Variable name to be mapped to the x-axis",
        precedence=PRECEDENCES["xyc"],
    )
    ys = param.ClassSelector(
        class_=(Iterable,),
        doc="Variable name to be mapped to the y-axis",
        precedence=PRECEDENCES["xyc"],
    )

    join = param.ObjectSelector(
        objects=OPTIONS["join"],
        doc=f"Method to join; {OPTIONS['join']}",
        precedence=PRECEDENCES["common"],
    )

    def __init__(self, dataset, xs, ys, join="overlay", **kwds):
        if hasattr(dataset, "reset_coords"):
            keys = dataset
        else:
            dataset = dataset.reset_index()
            keys = dataset.columns
        group_key, label_key = self._validate_keys(xs, ys, kwds, keys)

        arrays = []
        for group, group_dataset in self._groupby_key(dataset, group_key):
            for label, label_dataset in self._groupby_key(group_dataset, label_key):
                if len(label_dataset) == 0:
                    continue

                kwds_updated = self._update_kwds(
                    label_dataset,
                    keys,
                    kwds,
                    xs,
                    ys,
                    kwds.get("c", ""),
                    label,
                    join,
                )

                num_states = len(label_dataset)
                if num_states > 2000 and label_key is None:
                    warnings.warn(
                        f"Found more than {num_states} states "
                        f"which may take a considerable time to animate; "
                        f"set label to group a set of rows as separate items."
                    )

                super().__init__(
                    xs=label_dataset.get(xs),
                    ys=label_dataset.get(ys),
                    group=group,
                    label=label,
                    **kwds_updated,
                )
                arrays.append(self.copy())
        self.data = merge(arrays, join).data

    @staticmethod
    def _validate_keys(xs, ys, kwds, keys):
        group_key = kwds.pop("group", None)
        label_key = kwds.pop("label", None)

        for key in [xs, ys, group_key, label_key]:
            if key and key not in keys:
                raise ValueError(f"{key} not found in {keys}!")
        return group_key, label_key

    @staticmethod
    def _update_kwds(label_ds, keys, kwds, xs, ys, cs, label, join):
        kwds_updated = kwds.copy()

        # get labels from kwds
        if kwds_updated.get("style") != "bare":
            if "xlabel" not in kwds:
                kwds_updated["xlabel"] = str(xs).title()
            if "ylabel" not in kwds:
                kwds_updated["ylabel"] = str(ys).title()
            if "clabel" not in kwds:
                kwds_updated["clabel"] = str(cs).title()
            if "title" not in kwds and join == "layout":
                kwds_updated["title"] = label

        # swap xlabel and ylabel
        if kwds_updated.get("chart") == "barh":
            xlabel = kwds_updated.get("xlabel", "")
            ylabel = kwds_updated.get("ylabel", "")
            kwds_updated["xlabel"] = ylabel
            kwds_updated["ylabel"] = xlabel

        for key, val in kwds.items():
            if isinstance(val, dict):
                continue
            elif isinstance(val, str):
                if val in keys:
                    val = label_ds[val].values
            kwds_updated[key] = val

        return kwds_updated


class DataFrame(DataStructure):

    df = param.DataFrame(doc="Pandas DataFrame", precedence=PRECEDENCES["data"])

    _dim_type = "basic"

    def __init__(self, df, xs, ys=None, join="overlay", **kwds):
        super().__init__(df, xs, ys, join=join, **kwds)


class Dataset(DataStructure, Array2D):

    ds = param.DataFrame(doc="XArray Dataset", precedence=PRECEDENCES["data"])

    cs = param.ClassSelector(
        class_=(Iterable,),
        default=None,
        doc="Variable name to be mapped to the colorbar",
        precedence=PRECEDENCES["xyc"],
    )

    _dim_type = "grid"

    def __init__(self, ds, xs, ys, cs=None, join="overlay", **kwds):
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        super().__init__(ds, xs, ys, cs=cs, join=join, **kwds)


class Reference(GeographicData):

    chart = param.ObjectSelector(
        objects=CHARTS["ref"],
        doc=f"Type of chart; {CHARTS['ref']}",
        precedence=PRECEDENCES["common"],
    )

    x0s = param.ClassSelector(
        class_=(Iterable,),
        doc="Array to be mapped to lower x-axis",
        precedence=PRECEDENCES["xyc"],
    )
    x1s = param.ClassSelector(
        class_=(Iterable,),
        doc="Array to be mapped to upper x-axis",
        precedence=PRECEDENCES["xyc"],
    )
    y0s = param.ClassSelector(
        class_=(Iterable,),
        doc="Array to be mapped to lower y-axis",
        precedence=PRECEDENCES["xyc"],
    )
    y1s = param.ClassSelector(
        class_=(Iterable,),
        doc="Array to be mapped to lower y-axis",
        precedence=PRECEDENCES["xyc"],
    )
    inline_locs = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Inline label's other axis' location",
        precedence=PRECEDENCES["label"],
    )

    _dim_type = "ref"

    def __init__(self, x0s=None, x1s=None, y0s=None, y1s=None, **kwds):
        kwds = self._prep_kwds(x0s, x1s, y0s, y1s, **kwds)
        super().__init__(**kwds)
        self.configurables["ref"] = CONFIGURABLES["ref"]
        self.data = {self.rowcol: self._ds}

    def _prep_kwds(self, x0s, x1s, y0s, y1s, **kwds):
        if x0s is None and x1s is not None:
            x0s, x1s = x1s, x0s

        if y0s is None and y1s is not None:
            y0s, y1s = y1s, y0s

        kwds.update(
            {
                "x0": x0s,
                "x1": x1s,
                "y0": y0s,
                "y1": y1s,
            }
        )

        has_kwds = {key: val is not None for key, val in kwds.items()}
        if not any(has_kwds.values()):
            raise ValueError("Must provide either x0s, x1s, y0s, y1s!")

        has_xs = has_kwds["x0"] and has_kwds["x1"]
        has_ys = has_kwds["y0"] and has_kwds["y1"]

        if has_xs and has_ys:
            kwds["chart"] = "rectangle"
        elif has_kwds["x0"] and has_kwds["y0"]:
            kwds["chart"] = "scatter"
        elif has_xs:
            kwds["chart"] = "axvspan"
        elif has_ys:
            kwds["chart"] = "axhspan"
        elif has_kwds["x0"] or has_kwds["x1"]:
            kwds["chart"] = "axvline"
        elif has_kwds["y0"] or has_kwds["y1"]:
            kwds["chart"] = "axhline"
        else:
            raise ValueError(
                "One of the following combinations must be provided: "
                "x0s, x1s, y0s, y1s"
            )

        for key in list(kwds):
            val = kwds[key]
            if val is None:
                kwds.pop(key)

        return kwds

    def _load_dataset(self, **kwds):
        super()._load_dataset(**kwds)
        ds = self._ds
        num_items = len(ds["ref_item"])
        if self.inline_labels is not None:
            inline_locs = self.inline_locs
            if inline_locs is None and not ("x0" in ds and "y0" in ds):
                raise ValueError(
                    "Must provide inline_locs if inline_labels is not None!"
                )
            else:
                ds["inline_loc"] = (
                    DIMS["ref"],
                    self._adapt_input(inline_locs, num_items=num_items),
                )

        ds = ds.rename(
            {
                var: f"ref_{var}"
                for var in list(ds.data_vars)
                if ds[var].dims != ("state",)
            }
        )
        self._ds = ds
