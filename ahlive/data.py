import operator
import warnings
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import param
import xarray as xr
from matplotlib.colors import BoundaryNorm

from .animation import Animation
from .configuration import (
    CHARTS,
    CONFIGURABLES,
    DIMS,
    ITEMS,
    NULL_VALS,
    OPTIONS,
    PARAMS,
    PRESETS,
    VARS,
    Configuration,
    defaults,
    load_defaults,
)
from .easing import Easing
from .join import _combine, _get_rowcols, merge
from .util import (
    fillna,
    is_datetime,
    is_scalar,
    is_str,
    is_timedelta,
    pop,
    srange,
    to_1d,
    to_scalar,
)


class Data(Easing, Animation, Configuration):

    chart = param.ClassSelector(class_=Iterable)
    preset = param.ObjectSelector(
        objects=list(chain(*PRESETS.values())), doc=f"Chart preset; {PRESETS}"
    )
    style = param.ObjectSelector(
        objects=OPTIONS["style"], doc=f"Chart style; {OPTIONS['style']}"
    )
    label = param.String(allow_None=True, doc="Legend label for each item")
    group = param.String(doc="Group label for multiple items")

    state_labels = param.ClassSelector(
        class_=(Iterable,), doc="Dynamic label per state (bottom right)"
    )
    inline_labels = param.ClassSelector(
        class_=(Iterable,),
        doc="Dynamic label per item per state (item location)",
    )

    xmargins = param.Number(doc="Margins on the x-axis; ranges from 0-1")
    ymargins = param.Number(doc="Margins on the y-axis; ranges from 0-1")
    xlims = param.ClassSelector(class_=Iterable, doc="Limits for the x-axis")
    ylims = param.ClassSelector(class_=Iterable, doc="Limits for the y-axis")
    xlim0s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the left bounds of the x-axis",
    )
    xlim1s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the right bounds of the x-axis",
    )
    ylim0s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the bottom bounds of the y-axis",
    )
    ylim1s = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Limits for the top bounds of the y-axis",
    )
    hooks = param.HookList(
        doc="List of customization functions to apply; "
        "function must contain fig and ax as arguments"
    )

    title = param.String(allow_None=True, doc="Title label (outer top left)")
    subtitle = param.String(allow_None=True, doc="Subtitle label (outer top right)")
    xlabel = param.String(allow_None=True, doc="X-axis label (bottom center)")
    ylabel = param.String(allow_None=True, doc="Y-axis label (left center")
    note = param.String(allow_None=True, doc="Note label (bottom left)")
    caption = param.String(allow_None=True, doc="Caption label (outer left)")

    xticks = param.ClassSelector(class_=(Iterable,), doc="X-axis tick locations")
    yticks = param.ClassSelector(class_=(Iterable,), doc="Y-axis tick locations")

    legend = param.ObjectSelector(objects=OPTIONS["legend"], doc="Legend location")
    grid = param.ObjectSelector(default=True, objects=OPTIONS["grid"], doc="Grid type")

    rowcol = param.NumericTuple(
        default=(1, 1), length=2, doc="Subplot location as (row, column)"
    )

    _crs_names = None
    _parameters = None
    configurables = None
    data = None

    def __init__(self, num_states, **kwds):
        self.configurables = {
            "canvas": CONFIGURABLES["canvas"],
            "subplot": CONFIGURABLES["subplot"],
            "label": CONFIGURABLES["label"],
        }
        self._parameters = [key for key in dir(self) if not key.startswith("_")]
        input_vars = {
            key: kwds.pop(key) for key in list(kwds) if key not in self._parameters
        }
        super().__init__(**kwds)
        input_vars = self._amend_input_vars(input_vars)
        data_vars, num_items = self._load_data_vars(input_vars, num_states)
        coords = {"item": srange(num_items), "state": srange(num_states)}
        ds = xr.Dataset(coords=coords, data_vars=data_vars)
        ds = self._drop_state(ds)
        self.data = {self.rowcol: ds}

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

    @property
    def attrs(self):
        rowcol = list(self._data)[0]
        return self._data[rowcol].attrs

    def cols(self, num_cols):
        if num_cols == 0:
            raise ValueError("Number of columns must be > 1!")
        self_copy = deepcopy(self)
        data = {}
        for iplot, rowcol in enumerate(self_copy.data.copy()):
            row = (iplot) // num_cols + 1
            col = (iplot) % num_cols + 1
            data[(row, col)] = self_copy.data.pop(rowcol)
        self_copy.data = data
        return self_copy

    def _init_join(self, other):
        self_copy = deepcopy(self)
        other_copy = deepcopy(other)
        rowcols = _get_rowcols([self_copy, other_copy])
        return self_copy, other_copy, rowcols

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        strings = []
        for rowcol, ds in self.data.items():
            dims = ", ".join(f"{key}: {val}" for key, val in ds.dims.items())
            data = repr(ds.data_vars)
            strings.append(
                f'Subplot:{" ":9}{rowcol}\n'
                f'Dimensions:{" ":6}({dims})\n'
                f"{data}\n\n"
            )
        return "<ahlive.Data>\n" + "".join(strings)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        self_copy, other_copy, rowcols = self._init_join(other)

        data = {}
        for rowcol in rowcols:
            self_ds = self_copy.data.get(rowcol)
            other_ds = other_copy.data.get(rowcol)
            if other_ds is None:
                continue
            other_ds = self._match_states(self_ds, other_ds)

            if self_ds is None:
                data[rowcol] = other_ds
            elif other_ds is None:
                data[rowcol] = self_ds
            else:
                other_ds = self._shift_items(self_ds, other_ds)
                merged_ds = _combine([self_ds, other_ds], method="merge")
                merged_ds = self._drop_state(merged_ds)
                data[rowcol] = merged_ds
        self_copy.data = data
        self_copy = self._propagate_params(self_copy, other_copy)
        return self_copy

    def __rmul__(self, other):
        return other * self

    def __floordiv__(self, other):
        self_copy, other_copy, rowcols = self._init_join(other)
        self_rows = max(self_copy.data)[0]

        data = {}
        for rowcol in rowcols:
            self_ds = self_copy.data.get(rowcol)
            other_ds = other_copy.data.get(rowcol)
            if other_ds is None:
                continue
            other_ds = self._match_states(self_ds, other_ds)

            if rowcol[0] <= self_rows:
                rowcol_shifted = (rowcol[0] + self_rows, rowcol[1])
                data[rowcol_shifted] = other_ds
            else:
                data[rowcol] = other_ds

        self_copy.data.update(data)
        self_copy = self._propagate_params(self_copy, other_copy, layout=True)
        return self_copy

    def __truediv__(self, other):
        return self // other

    def __add__(self, other):
        self_copy, other_copy, rowcols = self._init_join(other)
        self_cols = max(self_copy.data, key=operator.itemgetter(1))[1]

        data = {}
        for rowcol in rowcols:
            self_ds = self_copy.data.get(rowcol)
            other_ds = other_copy.data.get(rowcol)
            if other_ds is None:
                continue
            other_ds = self._match_states(self_ds, other_ds)

            if rowcol[0] <= self_cols:
                rowcol_shifted = (rowcol[0], rowcol[1] + self_cols)
                data[rowcol_shifted] = other_ds
            else:
                data[rowcol] = other_ds

        self_copy.data.update(data)
        self_copy = self._propagate_params(self_copy, other_copy, layout=True)
        return self_copy

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        self_copy, other_copy, rowcols = self._init_join(other)

        data = {}
        for rowcol in rowcols:
            self_ds = self_copy.data.get(rowcol)
            other_ds = other_copy.data.get(rowcol)

            if self_ds is None:
                data[rowcol] = other_ds
            elif other_ds is None:
                data[rowcol] = self_ds
            else:
                other_ds = self._shift_items(self_ds, other_ds)
                other_ds["state"] = other_ds["state"] + self_ds["state"].max()
                merged_ds = _combine([self_ds, other_ds], method="merge")
                merged_ds = self._drop_state(merged_ds)
                merged_ds = merged_ds.map(fillna, keep_attrs=True)
                data[rowcol] = merged_ds
        self_copy.data = data
        self_copy = self._propagate_params(self_copy, other)
        return self_copy

    def __rsub__(self, other):
        return self - other

    def __iter__(self):
        return self.data.__iter__()

    @staticmethod
    def _config_bar_chart(ds, preset):
        if preset is None or preset == "series":
            if len(ds["item"]) == 1:
                ds.attrs["preset_kwds"]["preset"] = "series"
                return ds

        preset_kwds = load_defaults("preset_kwds", ds, base_chart=preset)
        bar_label = preset_kwds.get("bar_label", True)
        ds["tick_label"] = ds["x"]
        if bar_label:
            ds["bar_label"] = ds["x"]

        if preset == "race":
            preset_kwds = load_defaults("preset_kwds", ds, base_chart=preset)
            limit = preset_kwds.get("limit", None)
            # want to count NaNs so highest number is consistent
            ds["y"] = ds["y"].fillna(-np.inf)
            ranks = ds["y"].rank("item")
            # only keep items that are show at least once above the limit
            # to optimize and keep a smaller dataset
            ds = ds.sel(
                item=ds.where(ranks >= len(ds["item"]) - limit, drop=True)["item"]
            )
            ds["x"] = ds["y"].rank("item")
            # fill back in NaNs
            ds["y"] = ds["y"].where(np.isfinite(ds["y"]))
        else:
            ds["x"] = ds["x"].rank("item")
            if preset == "delta":
                x_delta = ds["x"].diff("item").mean() / 2
                ds["x_center"] = ds["x"] - x_delta
                ds["delta_label"] = ds["y"].diff("item")
                ds["y_center"] = ds["y"].shift(item=1) + ds["delta_label"] / 2
                ds["delta_label"] = ds["delta_label"].isel(item=slice(1, None))
                ds["delta"] = ds["delta_label"] / 2
        return ds

    @staticmethod
    def _config_trail_chart(ds):
        preset_kwds = load_defaults("preset_kwds", ds, base_chart="trail")
        trail_chart = preset_kwds["chart"]
        if trail_chart in ["line", "both"]:
            ds["x_trail"] = ds["x"].copy()
            ds["y_trail"] = ds["y"].copy()

        if trail_chart in ["scatter", "both"]:
            ds["x_discrete_trail"] = ds["x"].copy()
            ds["y_discrete_trail"] = ds["y"].copy()
        return ds

    def _config_rotate_chart(self, ds):
        num_states = len(ds["state"])
        x_dim = "grid_x" if "grid_x" in ds else "x"
        central_lon = ds.attrs["projection_kwds"].get(
            "central_longitude", ds[x_dim].min()
        )
        if is_scalar(central_lon):
            central_lon_end = ds[x_dim].max()
            central_lons = np.linspace(central_lon, central_lon_end, num_states)
        elif len(to_1d(central_lon)) != num_states:
            central_lons = np.linspace(
                np.min(central_lon), np.max(central_lon), num_states
            )
        else:
            central_lons = central_lon
        ds["central_longitude"] = ("state", central_lons)
        if "projection" not in ds.attrs["projection_kwds"]:
            ds.attrs["projection_kwds"]["projection"] = "Orthographic"
        return ds

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

        ds = xr.concat(scan_ds_list, "state").assign(**ds[stateless_vars])
        for var in ds.data_vars:
            if var not in grid_vars and grid_axis in ds[var].dims:
                ds[var] = ds[var].isel(**{grid_axis: 0})
        ds[grid_scan_axis] = (
            "state",
            np.tile(ds[grid_axis][::scan_stride].values, len(states)),
        )
        ds = ds.transpose(*DIMS["item"], "state", ...)
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
            for item_dim in ["item", "ref_item", "grid_item"]:
                if item_dim in ds.dims:
                    num_items = len(ds[item_dim])
                    break
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

    def _config_chart(self, ds, chart):
        preset = ds.attrs["preset_kwds"].get("preset", "")
        if chart.startswith("bar") or preset in PRESETS["bar"]:
            ds = self._config_bar_chart(ds, preset)
        elif preset == "trail":
            ds = self._config_trail_chart(ds)
        elif preset == "rotate":
            ds = self._config_rotate_chart(ds)
        elif preset.startswith("scan"):
            ds = self._config_scan_chart(ds, preset)

        ds = self._config_legend(ds)
        ds = self._config_grid_axes(ds, chart)
        return ds

    @staticmethod
    def _fill_null(ds):
        for var in ds.data_vars:
            if ds[var].dtype == "O":
                try:
                    ds[var] = ds[var].astype(float)
                except ValueError:
                    ds[var] = fillna(ds[var], how="both")
        return ds

    def _compute_limit_offset(self, limit, margin):
        if is_str(limit):
            return None

        if is_datetime(limit):
            base_diff = self._get_median_diff(limit).astype(float)
            offset = pd.Timedelta(np.nanmin(base_diff) * margin)
        else:
            offset = np.nanmedian(np.abs(limit)) * margin
        return offset

    @staticmethod
    def _get_median_diff(array):
        array = np.atleast_1d(array)
        if len(array) == 1:
            return array
        nan_indices = np.where(np.isnan(array))
        array[nan_indices] = array.ravel()[0]
        if array.ndim > 1 and array.shape[-1] > 1:
            base_diff = np.nanmedian(np.diff(array, axis=1))
        else:
            base_diff = np.nanmedian(np.diff(array.ravel()))
        base_diff = np.abs(base_diff)
        return base_diff

    def _add_xy01_limits(self, ds, chart):
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

        axes_kwds = load_defaults("axes_kwds", ds)
        for key, limit in limits.items():
            # example: xlim0s
            axis = key[0]  # x
            num = int(key[-2])  # 0
            is_lower_limit = num == 0

            axis_limit_key = f"{axis}lim"
            if axes_kwds is not None:
                in_axes_kwds = axis_limit_key in axes_kwds
            else:
                in_axes_kwds = False
            unset_limit = limit is None and not in_axes_kwds
            if in_axes_kwds:
                limit = axes_kwds[axis_limit_key][num]
            elif unset_limit:
                has_other_limit = limits[f"{key[:-2]}{1 - num}s"] is not None
                is_scatter = chart == "scatter"
                is_line_y = chart == "line" and axis == "y"
                is_bar_x = chart.startswith("bar") and axis == "x"
                is_bar_y = chart.startswith("bar") and axis == "y"
                is_fixed = any([is_scatter, is_line_y, is_bar_y, has_other_limit])
                if is_bar_y and is_lower_limit:
                    limit = "zero"
                elif is_bar_x:
                    continue
                elif is_fixed:
                    limit = "fixed_0.05"
                elif num == 1:
                    limit = "explore_0.005"
                else:
                    limit = "explore"

            if isinstance(limit, str):
                if "_" in limit:
                    limit, margin = limit.split("_")
                    margin = float(margin)
                else:
                    margin = 0
                if limit not in OPTIONS["limit"]:
                    raise ValueError(
                        f"Got {limit} for {key}; must be either "
                        f"from {OPTIONS['limit']} or numeric values!"
                    )

                grid_var = f"grid_{axis}"
                if grid_var in ds:
                    var = grid_var
                    item_dim = "grid_item"
                elif axis in ds:
                    var = axis
                    item_dim = "item"
                else:
                    if axis == "x":
                        if "ref_x0" in ds and "ref_x1" in ds:
                            var = "ref_x0" if num == 0 else "ref_x1"
                        elif "ref_x0" in ds:
                            var = "ref_x0"
                        elif "ref_x1" in ds:
                            var = "ref_x1"
                        else:
                            continue
                    elif axis == "y":
                        if "ref_y0" in ds and "ref_y1" in ds:
                            var = "ref_y0" if num == 0 else "ref_y1"
                        elif "ref_y0" in ds:
                            var = "ref_y0"
                        elif "ref_y1" in ds:
                            var = "ref_y1"
                        else:
                            continue
                    item_dim = "ref_item"

                if limit == "zero":
                    limit = 0
                elif limit == "fixed":
                    # fixed bounds, da.min()
                    stat = "min" if is_lower_limit else "max"
                    limit = getattr(ds[var], stat)().values
                elif limit == "explore":
                    stat = "min" if is_lower_limit else "max"
                    # explore bounds, pd.Series(da.min('item')).cummin()
                    if is_str(ds[var]) or item_dim == "grid_item":
                        continue
                    limit = getattr(
                        pd.Series(getattr(ds[var].ffill(item_dim), stat)(item_dim)),
                        f"cum{stat}",
                    )().values
                elif limit == "follow":
                    # follow latest state, da.min('item')
                    stat = "min" if is_lower_limit else "max"
                    limit = getattr(ds[var], stat)(item_dim).values

                if limit is not None and margin != 0:
                    offset = self._compute_limit_offset(limit, margin)
                    if offset is not None:
                        if num == 0:
                            # if I try limit -= offset, UFuncTypeError
                            limit = limit - offset
                        else:
                            limit = limit + offset

                if chart == "barh":
                    axis = "x" if axis == "y" else "y"
                    key = axis + key[1:]
                    # do not overwrite user input
                    if limits[key] is not None:
                        continue

            if limit is not None:
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

        for dim in DIMS["item"]:
            if dim in da.dims:
                if len(da[dim]) > 1:
                    return da

        vals = da.values
        unique_vals = np.unique(vals[~pd.isnull(vals)])
        if len(unique_vals) == 1:
            dim = da.dims[0]
            if dim != "state":
                item = da[dim][0]
                return xr.DataArray(unique_vals[0], dims=(dim,), coords={dim: [item]})
            else:
                return unique_vals[0]
        else:
            return da

    @staticmethod
    def _add_color_kwds(ds, chart):
        if chart is not None:
            if chart.startswith("bar"):
                if set(np.unique(ds["label"])) == set(np.unique(ds["x"])):
                    ds.attrs["legend_kwds"]["show"] = False

        if "c" in ds:
            c_var = "c"
            plot_key = "plot_kwds"
        else:
            c_var = "grid_c"
            plot_key = "grid_plot_kwds"

        cticks_kwds = load_defaults("cticks_kwds", ds)
        if c_var in ds:
            cticks = cticks_kwds.get("ticks")
            if cticks is None:
                num_colors = defaults["cticks_kwds"]["num_colors"]
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
                cmap = ds.attrs[plot_key].get("cmap", "plasma")
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

            ds.attrs["colorbar_kwds"]["show"] = ds.attrs[plot_key].get("colorbar", True)
        elif "colorbar_kwds" in ds.attrs:
            ds.attrs["colorbar_kwds"]["show"] = False
        return ds

    @staticmethod
    def _precompute_base_ticks(ds, base_kwds):
        for xyc in ITEMS["axes"]:
            if xyc in ds:
                if is_str(ds[xyc]):
                    ds.attrs[f"{xyc}ticks_kwds"]["is_str"] = True
                    continue
                elif "c" not in xyc:
                    ds.attrs[f"{xyc}ticks_kwds"]["is_str"] = False

                try:
                    base_kwds[f"{xyc}ticks"] = np.nanmedian(ds[xyc]) / 10
                except TypeError:
                    base_kwds[f"{xyc}ticks"] = np.nanmin(ds[xyc])

                if "c" in xyc:
                    continue

                ds.attrs[f"{xyc}ticks_kwds"]["is_datetime"] = is_datetime(ds[xyc])
        return ds, base_kwds

    def _precompute_base_labels(self, ds, base_kwds):
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
                            if is_datetime(base_diff):
                                base = np.nanmin(base_diff) / 5
                            else:
                                base = np.nanquantile(base_diff, 0.25)
                    if not pd.isnull(base):
                        base_kwds[key] = base
                except Exception as e:
                    if self.debug:
                        warnings.warn(str(e))
        return ds, base_kwds

    def _precompute_base(self, ds):
        base_kwds = {}
        ds, base_kwds = self._precompute_base_ticks(ds, base_kwds)
        ds, base_kwds = self._precompute_base_labels(ds, base_kwds)
        for key in ["s"]:  # for the legend
            if key in ds:
                base_kwds[key] = np.nanmedian(ds[key])

        ds.attrs["base_kwds"] = base_kwds
        return ds

    def _add_margins(self, ds):
        margins_kwds = load_defaults("margins_kwds", ds)
        margins = {}
        for axis in ["x", "y"]:
            keys = [key for key in [f"{axis}lim0s", f"{axis}lim1s"] if key in ds]
            if keys:
                if not is_str(ds[keys[0]]):
                    limit = ds[keys].to_array().max("variable")
                    margin = margins_kwds.get(axis, 0)
                    margins[axis] = self._compute_limit_offset(limit, margin)

        for key in ["xlim0s", "xlim1s", "ylim0s", "ylim1s"]:
            if key in ds.data_vars:  # TODO: test str / dt
                axis = key[0]
                num = int(key[-2])  # 0
                is_lower_limit = num == 0
                margin = margins.get(axis)
                if margin is None:
                    continue
                if is_lower_limit:
                    ds[key] = ds[key] - margin
                else:
                    ds[key] = ds[key] + margin

        return ds

    def _add_durations(self, ds):
        if "fps" in ds.attrs["animate_kwds"]:
            return ds

        num_states = len(ds["state"])
        durations_kwds = load_defaults("durations_kwds", ds)
        transition_frames = durations_kwds.pop("transition_frames")
        aggregate = durations_kwds.pop("aggregate")

        durations = durations_kwds.get("durations", 0.5 if num_states < 8 else 1 / 60)
        if isinstance(durations, (int, float)):
            durations = np.repeat(durations, num_states)

        if np.isnan(durations[-1]):
            durations[-1] = 0
        durations[-1] += durations_kwds["final_frame"]

        if "duration" in ds:
            ds["duration"] = ("state", durations + ds["duration"].values)
        else:
            ds["duration"] = ("state", durations)
        ds["duration"].attrs["transition_frames"] = transition_frames
        ds["duration"].attrs["aggregate"] = aggregate
        return ds

    def _interp_dataset(self, ds):
        subgroup_ds_list = []
        interpolate_kwds = ds.attrs["interpolate_kwds"]

        for kind in ["", "ref_", "grid_"]:
            item_dim = f"{kind}item"
            interp_var = f"{kind}interp"
            ease_var = f"{kind}ease"
            if interp_var not in ds:
                continue

            ds[interp_var] = fillna(ds[interp_var], how="both")
            ds[ease_var] = fillna(ds[ease_var], how="both")

            vars_seen = set([])
            for _, interp_ds in ds.groupby(interp_var):
                interpolate_kwds["interp"] = pop(interp_ds, interp_var, get=-1)
                for _, ease_ds in interp_ds.groupby(ease_var):
                    if not ease_ds:
                        continue
                    if f"stacked_{kind}item_state" in ease_ds.dims:
                        ease_ds = self._drop_state(ease_ds.unstack())
                        if "duration" in ease_ds:
                            ease_ds["duration"] = ease_ds["duration"].isel(
                                **{item_dim: 0}
                            )
                    interpolate_kwds["ease"] = pop(ease_ds, ease_var, get=-1)
                    var_list = []
                    for var in ease_ds.data_vars:
                        has_item = item_dim in ease_ds[var].dims
                        is_stateless = ease_ds[var].dims == ("state",)
                        is_scalar = ease_ds[var].dims == ()
                        var_seen = var in vars_seen
                        if has_item:
                            ease_ds[var].attrs.update(interpolate_kwds)
                            var_list.append(var)
                            vars_seen.add(var)
                        elif (is_stateless or is_scalar) and not var_seen:
                            ease_ds[var].attrs.update(interpolate_kwds)
                            var_list.append(var)
                            vars_seen.add(var)
                    ease_ds = ease_ds[var_list]
                    try:
                        ease_ds = ease_ds.map(self.interpolate, keep_attrs=True)
                    except IndexError as e:
                        if self.debug:
                            raise IndexError(e)
                    subgroup_ds_list.append(ease_ds)

        ds = xr.combine_by_coords(subgroup_ds_list)
        ds = ds.drop_vars(
            var for var in ds.data_vars if "interp" in var or "ease" in var
        )

        if "state" not in ds.dims:
            ds = ds.expand_dims("state").transpose(..., "state")
        ds["state"] = srange(len(ds["state"]))

        if "s" in ds:
            ds["s"] = fillna(ds["s"].where(ds["s"] > 0), how="both")
        return ds

    def _get_crs(self, crs_name, crs_kwds, central_longitude=None):
        import cartopy.crs as ccrs

        if self._crs_names is None:
            self._crs_names = {
                crs_name.lower(): crs_name
                for crs_name in dir(ccrs)
                if "_" not in crs_name
            }

        if crs_name is not None:
            crs_name = crs_name.lower()
        crs_name = self._crs_names.get(crs_name, "PlateCarree")
        if central_longitude is not None:
            crs_kwds["central_longitude"] = central_longitude
        crs_obj = getattr(ccrs, crs_name)(**crs_kwds)
        return crs_obj

    def _add_geo_transforms(self, ds):
        crs_kwds = load_defaults("crs_kwds", ds)
        crs = crs_kwds.pop("crs", None)

        projection_kwds = load_defaults("projection_kwds", ds)
        projection = projection_kwds.pop("projection", None)

        if crs is not None or projection is not None:
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

    def _add_animate_kwds(self, ds):
        animate_kwds = {}
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
        animate_kwds["num_states"] = num_states
        ds.attrs["animate_kwds"].update(**animate_kwds)
        return ds

    def finalize(self):
        if all(ds.attrs.get("finalized", False) for ds in self.data.values()):
            return self

        self_copy = deepcopy(self)

        data = {}
        for i, (rowcol, ds) in enumerate(self_copy.data.items()):
            chart = to_scalar(ds["chart"]) if "chart" in ds else ""
            ds = self._fill_null(ds)
            ds = self._add_xy01_limits(ds, chart)
            ds = self._compress_vars(ds)
            ds = self._add_color_kwds(ds, chart)
            ds = self._config_chart(ds, chart)
            ds = self._precompute_base(ds)
            ds = self._add_margins(ds)
            ds = self._add_durations(ds)
            ds = self._interp_dataset(ds)
            ds = self._add_geo_transforms(ds)
            ds = self._add_animate_kwds(ds)
            ds.attrs["finalized"] = True
            data[rowcol] = ds

        self_copy.data = data
        return self_copy

    def _adapt_input(self, val, num_states, num_items=None, reshape=True, shape=None):
        # TODO: add test
        val = np.array(val)
        if is_scalar(val):
            val = np.repeat(val, num_states)

        if not is_datetime(val):
            # make string '1' into 1
            try:
                val = val.astype(float)
            except (ValueError, TypeError):
                pass

        if reshape:
            if shape is None:
                val = val.reshape(-1, num_states)
            else:
                val = val.reshape(-1, num_states, *shape)

        if num_items is not None and val.shape[0] != num_items:
            val = np.tile(val, (num_items, 1))
        return val

    def _amend_input_vars(self, input_vars):
        # TODO: add test
        for key in list(input_vars.keys()):
            if key == "c":
                # won't show up if
                continue
            key_and_s = key + "s"
            key_strip = key.rstrip("s")
            expected_key = None
            if key_and_s in self._parameters and key_and_s != key:
                warnings.warn(f"Replacing unexpected {key} as {key_and_s}!")
                expected_key = key_and_s
            elif key_strip in self._parameters and key_strip != key:
                warnings.warn(f"Replacing unexpected {key} as {key_strip}!")
                expected_key = key_strip
            if expected_key:
                setattr(self, expected_key, input_vars.pop(key))

        for lim in ITEMS["limit"]:
            if lim in input_vars:
                warnings.warn(f"Replacing unexpected {key} as {key_strip}!")
                if "x" in lim and "0" in lim:
                    expected_key = "xlim0s"
                elif "x" in lim and "1" in lim:
                    expected_key = "xlim1s"
                elif "y" in lim and "0" in lim:
                    expected_key = "ylim0s"
                elif "y" in lim and "1" in lim:
                    expected_key = "ylim1s"
                setattr(self, expected_key, input_vars.pop(key))

        return input_vars

    def _load_data_vars(self, input_vars, num_states):
        # TODO: add test
        if self.chart is None:
            if num_states < 8 or "s" in input_vars:
                chart = "scatter"
            else:
                chart = "line"
        else:
            chart = self.chart

        label = self.label or ""
        group = self.group or ""

        if self.interp is None:
            interp = "cubic" if num_states < 8 else "linear"
        else:
            interp = self.interp
        ease = self.ease or "in_out"

        data_vars = {key: val for key, val in input_vars.items() if val is not None}
        for var in list(data_vars.keys()):
            val = data_vars.pop(var)
            val = self._adapt_input(val, num_states)
            dims = DIMS["ref"] if var.startswith("ref") else DIMS["basic"]
            data_vars[var] = dims, val

        if self.state_labels is not None:
            state_labels = self._adapt_input(
                self.state_labels, num_states, reshape=False
            )
            data_vars["state_label"] = ("state", state_labels)

        num_items = 1
        if self.inline_labels is not None:
            inline_labels = self._adapt_input(self.inline_labels, num_states)
            num_items = inline_labels.shape[0]
            data_vars["inline_label"] = DIMS["basic"], inline_labels

        # pass unique for dataframe
        data_vars["chart"] = (
            DIMS["basic"],
            self._adapt_input(chart, num_states, num_items=num_items),
        )
        data_vars["label"] = (
            DIMS["basic"],
            self._adapt_input(label, num_states, num_items=num_items),
        )
        data_vars["group"] = (
            DIMS["basic"],
            self._adapt_input(group, num_states, num_items=num_items),
        )
        data_vars["interp"] = (
            DIMS["basic"],
            self._adapt_input(interp, num_states, num_items=num_items),
        )
        data_vars["ease"] = (
            DIMS["basic"],
            self._adapt_input(ease, num_states, num_items=num_items),
        )

        return data_vars, num_items

    @staticmethod
    def _match_states(self_ds, other_ds):
        other_num_states = len(other_ds["state"])
        self_num_states = len(self_ds["state"])
        if other_num_states != self_num_states:
            warnings.warn(
                f"The latter dataset has {other_num_states} state(s) while "
                f"the former has {self_num_states} state(s); "
                f"reindexing the latter to match the former."
            )
            other_ds = other_ds.reindex(state=self_ds["state"]).map(
                fillna, keep_attrs=True
            )
        return other_ds

    @staticmethod
    def _shift_items(self_ds, other_ds):
        for item in DIMS["item"]:
            if not (item in self_ds.dims and item in other_ds.dims):
                continue
            has_same_items = (
                len(set(self_ds[item].values) | set(other_ds[item].values)) > 0
            )
            if has_same_items:
                other_ds[item] = other_ds[item].copy()
                other_ds[item] = other_ds[item] + self_ds[item].max()
        return other_ds

    @staticmethod
    def _drop_state(merged_ds):
        for var in VARS["stateless"]:
            if var in merged_ds:
                if "state" in merged_ds[var].dims:
                    merged_ds[var] = merged_ds[var].isel(state=-1)
        return merged_ds

    def _propagate_params(self, self_copy, other, layout=False):
        canvas_params = [
            param
            for param, configurable in PARAMS.items()
            if configurable in CONFIGURABLES["canvas"]
        ]
        self_copy.configurables.update(**other.configurables)
        for param_ in self._parameters:
            not_canvas_param = param_ not in canvas_params
            if callable(param_) or (layout and not_canvas_param):
                continue

            try:
                self_param = getattr(self_copy, param_)
                other_param = getattr(other, param_)
            except AttributeError:
                continue

            try:
                self_null = self_param in NULL_VALS
            except ValueError:
                self_null = False

            try:
                other_null = other_param in NULL_VALS
            except ValueError:
                other_null = False

            if param_ in PARAMS:
                if self_null and not other_null:
                    setattr(self_copy, param_, other_param)
                    self_copy = self_copy.config(PARAMS[param_])
                elif not self_null and other_null:
                    self_copy = self_copy.config(PARAMS[param_])
        return self_copy


class GeographicData(Data):

    crs = param.String(doc="The coordinate reference system to project from")
    projection = param.String(doc="The coordinate reference system to project to")
    central_lon = param.ClassSelector(
        class_=(Iterable, int, float), doc="Longitude to center the map on"
    )

    borders = param.Boolean(default=None, doc="Whether to show borders")
    coastline = param.Boolean(default=None, doc="Whether to show coastlines")
    land = param.Boolean(default=None, doc="Whether to show land surfaces")
    ocean = param.Boolean(default=None, doc="Whether to show ocean surfaces")
    lakes = param.Boolean(default=None, doc="Whether to show lakes")
    rivers = param.Boolean(default=None, doc="Whether to show rivers")
    states = param.Boolean(default=None, doc="Whether to show states")
    worldwide = param.Boolean(default=None, doc="Whether to view globally")

    def __init__(self, num_states, **kwds):
        super().__init__(num_states, **kwds)
        self.configurables["geo"] = CONFIGURABLES["geo"]


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
        **kwds,
    ):
        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = deepcopy(self)
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
                val = kwds[key]
                if isinstance(val, str):
                    if val in ds:
                        kwds[key] = ds[val]

                        if inline_locs is None:
                            has_x0s = kwds["x0s"] is not None
                            has_y0s = kwds["y0s"] is not None
                            if has_x0s and not has_y0s:
                                kwds["inline_locs"] = ds["y"]
                            elif has_y0s and not has_x0s:
                                kwds["inline_locs"] = ds["x"]

            self_copy *= Reference(**kwds)

        return self_copy


class ColorArray(param.Parameterized):

    cs = param.ClassSelector(
        class_=(Iterable,), doc="Array to be mapped to the colorbar"
    )

    cticks = param.ClassSelector(class_=(Iterable,), doc="Colorbar tick locations")
    ctick_labels = param.ClassSelector(class_=(Iterable,), doc="Colorbar tick labels")
    colorbar = param.Boolean(default=None, doc="Whether to show colorbar")
    clabel = param.String(doc="Colorbar label")

    def __init__(self, **kwds):
        self.configurables["color"] = CONFIGURABLES["color"]
        super().__init__(**kwds)


class RemarkArray(param.Parameterized):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.configurables["remark"] = CONFIGURABLES["remark"]

    def _match_values(self, da, values, first, rtol, atol):
        if is_datetime(da):
            values = pd.to_datetime(values)
        values = to_1d(values)
        if first:
            # + 1 because state starts counting at 1
            return xr.concat(
                (da["state"] == (da >= value).argmax() + 1 for value in values),
                "stack",
            ).sum("stack")
        try:
            rtol = rtol or 1e-05
            atol = atol or 1e-08
            return xr.concat(
                (
                    da.where(
                        xr.DataArray(
                            np.isclose(da, value, rtol=rtol, atol=atol),
                            dims=da.dims,
                        )
                    )
                    for value in to_1d(values)
                ),
                "stack",
            ).sum("stack")
        except TypeError as e:
            if self.debug:
                warnings.warn(e)
            return da.isin(values)

    def remark(
        self,
        remarks=None,
        durations=None,
        condition=None,
        xs=None,
        ys=None,
        cs=None,
        state_labels=None,
        inline_labels=None,
        first=False,
        rtol=None,
        atol=None,
        rowcols=None,
    ):
        args = (xs, ys, cs, state_labels, inline_labels, condition)
        args_none = sum([1 for arg in args if arg is None])
        if args_none == len(args):
            raise ValueError(
                "Must supply either xs, ys, cs, state_labels, "
                "inline_labels, or condition!"
            )
        elif args_none != len(args) - 1:
            raise ValueError(
                "Must supply only one of xs, ys, cs, state_labels, "
                "inline_labels, or condition!"
            )

        if durations is None and remarks is None:
            raise ValueError("Must supply at least remarks or durations!")

        if rowcols is None:
            rowcols = self.data.keys()

        if first and condition is not None:
            warnings.warn("Unable to use first with condition!")
        if rtol is not None and condition is not None:
            warnings.warn("Unable to use rtol with condition!")
        if atol is not None and condition is not None:
            warnings.warn("Unable to use atol with condition!")

        self_copy = deepcopy(self)
        data = {}
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            if xs is not None:
                condition = self._match_values(ds["x"], xs, first, rtol, atol)
            elif ys is not None:
                condition = self._match_values(ds["y"], ys, first, rtol, atol)
            elif cs is not None:
                condition = self._match_values(ds["c"], cs, first, rtol, atol)
            elif state_labels is not None:
                condition = self._match_values(
                    ds["state_label"], state_labels, first, rtol, atol
                )
            elif inline_labels is not None:
                condition = self._match_values(
                    ds["inline_label"], inline_labels, first, rtol, atol
                )
            else:
                condition = np.array(condition)

            # condition = condition.broadcast_like(ds)  # TODO: investigate grid
            if remarks is not None:
                if "remark" not in ds:
                    ds["remark"] = (
                        DIMS["basic"],
                        np.full((len(ds["item"].values), len(ds["state"])), ""),
                    )
                if isinstance(remarks, str):
                    if remarks in ds.data_vars:
                        remarks = ds[remarks].astype(str)
                ds["remark"] = xr.where(condition, remarks, ds["remark"])

            if durations is not None:
                if "duration" not in ds:
                    ds["duration"] = (
                        "state",
                        self._adapt_input(
                            np.zeros_like(ds["state"]),
                            len(ds["state"]),
                            reshape=False,
                        ),
                    )
                ds["duration"] = xr.where(condition, durations, ds["duration"])
                if "item" in ds["duration"].dims:
                    ds["duration"] = ds["duration"].max("item")

            data[rowcol] = ds
        self_copy.data = data
        return self_copy


class Array(GeographicData, ReferenceArray, ColorArray, RemarkArray):

    xs = param.ClassSelector(class_=(Iterable,), doc="Array to be mapped to the x-axis")
    ys = param.ClassSelector(class_=(Iterable,), doc="Array to be mapped to the y-axis")

    def __init__(self, xs, ys, **kwds):
        for xys, xys_arr in {"xs": xs, "ys": ys}.items():
            if isinstance(xys_arr, str):
                raise ValueError(
                    f"{xys} be an Iterable, but cannot be str; got {xys_arr}!"
                )

        num_xs = len(to_1d(xs))
        num_ys = len(to_1d(ys))
        if num_xs != num_ys and num_xs != 1 and num_ys != 1:
            raise ValueError(
                f"Length of x ({num_xs}) must match the length of y ({num_ys}) "
                f"if not a scalar!"
            )

        num_states = max(num_xs, num_ys)
        super().__init__(num_states, **kwds)
        ds = self.data[self.rowcol]

        ds = ds.assign(
            **{
                "x": (DIMS["basic"], self._adapt_input(xs, num_states)),
                "y": (DIMS["basic"], self._adapt_input(ys, num_states)),
            }
        )
        if "cs" in kwds:
            cs = kwds.pop("cs")
            ds["c"] = (DIMS["basic"], self._adapt_input(cs, num_states))
        self.data = {self.rowcol: ds}

    def invert(self):
        data = {}
        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            attrs = ds.attrs
            df = ds.to_dataframe().rename_axis(DIMS["basic"][::-1])
            ds = df.to_xarray().assign_attrs(attrs).transpose(*DIMS["basic"])
            data[rowcol] = ds
        self_copy.data = data
        return self_copy


class Array2D(GeographicData, ReferenceArray, ColorArray, RemarkArray):

    chart = param.ObjectSelector(
        default=CHARTS["grid"][0],
        objects=CHARTS["grid"],
        doc=f"Type of chart; {CHARTS['grid']}",
    )

    inline_xs = param.ClassSelector(
        class_=(Iterable, int, float), doc="Inline label's x locations"
    )
    inline_ys = param.ClassSelector(
        class_=(Iterable, int, float), doc="Inline label's y locations"
    )

    def __init__(self, xs, ys, cs, **kwds):
        cs = np.array(cs)
        shape = cs.shape[-2:]
        if cs.ndim > 2:
            num_states = len(cs)
            if shape[0] != len(ys):
                cs = np.swapaxes(cs, -1, -2)
                shape = shape[::-1]  # TODO: auto figure out time dimension
        else:
            num_states = 1
        super().__init__(num_states, **kwds)
        self.configurables["grid"] = CONFIGURABLES["grid"]

        ds = self.data[self.rowcol]
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)
        ds = ds.assign_coords({"x": xs, "y": ys}).assign(
            {
                "c": (
                    DIMS["grid"],
                    self._adapt_input(cs, num_states, shape=shape),
                )
            }
        )

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
                    self._adapt_input(inline_xs, num_states),
                )
                ds["inline_y"] = (
                    DIMS["basic"],
                    self._adapt_input(inline_ys, num_states),
                )

        grid_vars = list(ds.data_vars) + ["x", "y", "item"]
        ds = ds.rename(
            {var: f"grid_{var}" for var in grid_vars if ds[var].dims != ("state",)}
        )

        self.data = {self.rowcol: ds}


class DataStructure(param.Parameterized):

    join = param.ObjectSelector(
        objects=ITEMS["join"], doc=f"Method to join; {ITEMS['join']}"
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)

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
        if kwds_updated.get("style") != "bare":
            if "xlabel" not in kwds:
                kwds_updated["xlabel"] = xs.title()
            if "ylabel" not in kwds:
                kwds_updated["ylabel"] = ys.title()
            if "clabel" not in kwds:
                kwds_updated["clabel"] = cs.title()
            if "title" not in kwds and join == "layout":
                kwds_updated["title"] = label

        if kwds_updated.get("chart") == "barh":
            kwds_updated["xlabel"], kwds_updated["ylabel"] = (
                kwds_updated["ylabel"],
                kwds_updated["xlabel"],
            )

        for key, val in kwds.items():
            if isinstance(val, dict):
                continue
            elif isinstance(val, str):
                if val in keys:
                    val = label_ds[val].values
            kwds_updated[key] = val

        return kwds_updated


class DataFrame(Array, DataStructure):

    df = param.DataFrame(doc="Pandas DataFrame")

    def __init__(self, df, xs, ys, join="overlay", **kwds):
        df = df.reset_index()
        group_key, label_key = self._validate_keys(xs, ys, kwds, df.columns)

        arrays = []
        for group, group_df in self._groupby_key(df, group_key):
            for label, label_df in self._groupby_key(group_df, label_key):
                kwds_updated = self._update_kwds(
                    label_df,
                    label_df.columns,
                    kwds,
                    xs,
                    ys,
                    kwds.get("c", ""),
                    label,
                    join,
                )

                num_rows = len(label_df)
                if num_rows > 2000 and label_key is None:
                    warnings.warn(
                        f"Found more than {num_rows} states "
                        f"which may take a considerable time to animate; "
                        f"set label to group a set of rows as separate items."
                    )

                super().__init__(
                    label_df[xs],
                    label_df[ys],
                    group=group,
                    label=label,
                    **kwds_updated,
                )
                arrays.append(deepcopy(self))
        self.data = merge(arrays, join, quick=True).data


class Dataset(Array2D, DataStructure):

    ds = param.ClassSelector(class_=(xr.Dataset,), doc="Xarray Dataset")

    def __init__(self, ds, xs, ys, cs, join="overlay", **kwds):
        group_key, label_key = self._validate_keys(xs, ys, kwds, ds)

        arrays = []
        for group, group_ds in self._groupby_key(ds, group_key):
            for label, label_ds in self._groupby_key(group_ds, label_key):
                kwds_updated = self._update_kwds(
                    label_ds,
                    list(label_ds.data_vars) + list(label_ds.coords),
                    kwds,
                    xs,
                    ys,
                    cs,
                    label,
                    join,
                )

                super().__init__(
                    label_ds[xs],
                    label_ds[ys],
                    label_ds[cs],
                    group=group,
                    label=label,
                    **kwds_updated,
                )
                arrays.append(deepcopy(self))
        self.data = merge(arrays, join, quick=True).data


class Reference(GeographicData):

    chart = param.ObjectSelector(
        objects=CHARTS["ref"], doc=f"Type of chart; {CHARTS['ref']}"
    )

    x0s = param.ClassSelector(
        class_=(Iterable,), doc="Array to be mapped to lower x-axis"
    )
    x1s = param.ClassSelector(
        class_=(Iterable,), doc="Array to be mapped to upper x-axis"
    )
    y0s = param.ClassSelector(
        class_=(Iterable,), doc="Array to be mapped to lower y-axis"
    )
    y1s = param.ClassSelector(
        class_=(Iterable,), doc="Array to be mapped to lower y-axis"
    )
    inline_locs = param.ClassSelector(
        class_=(Iterable, int, float), doc="Inline label's other axis' location"
    )

    def __init__(self, x0s=None, x1s=None, y0s=None, y1s=None, **kwds):
        ref_kwds = {
            "x0": x0s,
            "x1": x1s,
            "y0": y0s,
            "y1": y1s,
        }
        has_kwds = {key: val is not None for key, val in ref_kwds.items()}
        if not any(has_kwds.values()):
            raise ValueError("Must provide either x0s, x1s, y0s, y1s!")

        has_xs = has_kwds["x0"] and has_kwds["x1"]
        has_ys = has_kwds["y0"] and has_kwds["y1"]
        if has_xs and has_ys:
            kwds["chart"] = "rectangle"
        elif has_kwds["x0"] and has_kwds["y0"]:
            kwds["chart"] = "scatter"
        elif has_kwds["x0"] and has_kwds["x1"]:
            kwds["chart"] = "axvspan"
        elif has_kwds["y0"] and has_kwds["y1"]:
            kwds["chart"] = "axhspan"
        elif has_kwds["x0"]:
            kwds["chart"] = "axvline"
        elif has_kwds["x1"]:
            kwds["chart"] = "axvline"
            ref_kwds["x0"], ref_kwds["x1"] = ref_kwds["x1"], ref_kwds["x0"]
        elif has_kwds["y0"]:
            kwds["chart"] = "axhline"
        elif has_kwds["y1"]:
            kwds["chart"] = "axhline"
            ref_kwds["y0"], ref_kwds["y1"] = ref_kwds["y1"], ref_kwds["y0"]
        else:
            raise NotImplementedError(
                "One of the following combinations must be provided: "
                "x0s, x1s, y0s, y1s"
            )

        for key in list(ref_kwds):
            val = ref_kwds[key]
            if val is not None:
                num_states = to_1d(val, flat=False).shape[-1]
            else:
                ref_kwds.pop(key)

        max_len = max(len(to_1d(val)) for val in ref_kwds.values())
        if max_len > 1:
            for key, val in ref_kwds.items():
                if is_scalar(val):
                    ref_kwds[key] = np.repeat(val, max_len)

        super().__init__(num_states, **kwds)
        self.configurables["ref"] = CONFIGURABLES["ref"]

        ds = self.data[self.rowcol]

        for key, val in ref_kwds.items():
            val = self._adapt_input(val, num_states)
            if val is not None:
                ds[key] = DIMS["ref"], val

        num_items = len(ds["ref_item"])
        if self.inline_labels is not None:
            inline_locs = self.inline_locs
            if inline_locs is None:
                raise ValueError(
                    "Must provide inline_locs if inline_labels is not None!"
                )
            else:
                ds["inline_loc"] = (
                    DIMS["ref"],
                    self._adapt_input(inline_locs, num_states, num_items=num_items),
                )

        for var in ds.data_vars:
            ref_var = f"ref_{var}"
            if "item" in ds[var].dims and "state" not in ds[var].dims:
                if len(ds[var]["item"]) != num_items:
                    values = np.repeat(ds[var].values, num_items)
                else:
                    values = ds[var].values
                ds[ref_var] = ("ref_item", values)
                ds = ds.drop_vars(var)
            elif "item" in ds[var].dims:
                ds[ref_var] = (DIMS["ref"], ds[var].values)
                ds = ds.drop_vars(var)
            elif ds[var].dims != ("state",):
                ds = ds.rename({var: ref_var})

        ds["ref_item"] = srange(len(ds["ref_item"]))
        ds = ds.drop_vars("item")
        self.data = {self.rowcol: ds}
