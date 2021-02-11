import base64
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from io import BytesIO

import dask.delayed
import dask.diagnostics
import imageio
import matplotlib  # noqa
import numpy as np
import pandas as pd
import param
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .configuration import (
    CONFIGURABLES,
    ITEMS,
    OPTIONS,
    TEMP_FILE,
    defaults,
    load_defaults,
)
from .util import is_datetime, is_scalar, pop, srange, to_1d, to_num, to_pydt, to_scalar


class Animation(param.Parameterized):

    save = param.String(default=None, doc="Output file path")
    show = param.Boolean(default=True, doc="Whether to show in Jupyter")

    figsize = param.NumericTuple(
        default=None, length=2, doc="Figure's size as width and height"
    )
    spacing = param.Dict(
        default=None,
        doc=f"Subplot spacing; {OPTIONS['spacing']}",
    )
    suptitle = param.String(
        allow_None=True, doc="Figure's super title (outer top center)"
    )
    watermark = param.String(
        allow_None=True, doc="Figure's watermark (outer bottom right)"
    )

    # compute kwds
    workers = param.Integer(
        default=None,
        bounds=(1, None),
        doc="Number of workers used to render separate static states",
    )
    scheduler = param.ObjectSelector(
        default=None,
        objects=OPTIONS["scheduler"],
        doc=f"Type of workers; {OPTIONS['scheduler']}",
    )

    # animate kwds
    animate = param.ClassSelector(
        default=True,
        class_=(Iterable, int, slice, bool),
        doc="Whether to stitch together the static frames; "
        "head animates only the beginning states, "
        "tail animates only the ending states, any other str animates "
        "select states equally ranged from beginning to ending, and "
        "number of states to animate can be specified by specfiying _{int}; "
        "int renders a single states, slice renders a range of states "
        "bool enables or disables the stitched animation",
    )
    fps = param.Number(default=None, doc="frames (states) animated per second")
    fmt = param.ObjectSelector(
        default=None, objects=OPTIONS["fmt"], doc="Output file format"
    )
    loop = param.ObjectSelector(
        default=None,
        objects=list(np.arange(0, 9999)) + [True, False],
        doc="Number of times the animation plays; "
        "0 or True plays the animation indefinitely",
    )
    durations = param.ClassSelector(
        class_=(Iterable, int, float),
        doc="Seconds to delay per state; Iterables must match number of states",
    )
    pygifsicle = param.Boolean(
        default=None,
        doc="Whether to use pygifsicle to reduce file size. "
        "If save is not set, will temporarily write file to disk first.",
    )

    debug = param.Boolean(
        default=False,
        doc="Show additional debugging info and set scheduler to single-threaded",
    )

    _canvas_kwds = None
    _temp_file = TEMP_FILE
    _path_effects = [withStroke(linewidth=2, alpha=0.5, foreground="white")]

    def __init__(self, **kwds):
        super().__init__(**kwds)

    @staticmethod
    def _get_base_format(num):
        num = to_scalar(num)

        is_timedelta = isinstance(num, np.timedelta64)
        num = to_num(num)
        if isinstance(num, str):
            return "s"

        if is_timedelta:
            num = num / 1e9  # nanoseconds to seconds
            if num < 1:  # 1 second
                return "%S.%f"
            elif num < 60:  # 1 minute
                return "%M:%S"
            elif num < 3600:  # 1 hour
                return "%I:%M %p"
            elif num < 86400:  # 1 day
                return "%HZ %b %d"
            elif num < 604800:  # 7 days
                return "%b %d"
            elif num < 2678400:  # 31 days
                return "%b %d"
            elif num < 15768000:  # 6 months
                return "%b"
            elif num < 31536000:  # 1 year
                return "%b '%y"
            else:
                return "%Y"

        if num == 0:
            return ".1f"

        order_of_magnitude = int(np.floor(np.log10(abs(num))))
        if order_of_magnitude >= 1:
            return ".0f"
        if order_of_magnitude > 0:
            return ".1f"
        else:
            return f".{abs(order_of_magnitude)}f"

    def _update_text(self, kwds, label_key, base=None, apply_format=True):
        label = kwds.get(label_key, None)

        if isinstance(label, Iterable) and not isinstance(label, str):
            labels = []
            for i, sub_label in enumerate(kwds[label_key]):
                sub_kwds = kwds.copy()
                sub_kwds[label_key] = sub_label
                sub_kwds = self._update_text(
                    sub_kwds, label_key, base=base, apply_format=apply_format
                )
                format_ = sub_kwds["format"]
                labels.append(sub_kwds[label_key])
            kwds[label_key] = labels
            kwds["format"] = format_
            kwds = {key: val for key, val in kwds.items() if key in sub_kwds.keys()}
            return kwds

        format_ = kwds.pop("format", "auto").lstrip("%")
        if base is not None and format_ == "auto":
            try:
                format_ = self._get_base_format(base)
            except TypeError:
                pass

        if is_datetime(label):
            if format_ == "auto":
                format_ = "%Y-%m-%d %H:%M"
            label = to_pydt(label)

        if format_ != "auto":
            if apply_format:
                try:
                    label = f"{label:{format_}}"
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Could not apply {format_} on {label} due to {e}")
            else:
                kwds["format"] = format_

        prefix = kwds.pop("prefix", "")
        suffix = kwds.pop("suffix", "")
        if "units" in kwds:
            units = f" [{kwds.pop('units')}]"
        else:
            units = ""
        label = f"{prefix}{label}{suffix}{units}"

        replacements = kwds.pop("replacements", {})
        for key, val in replacements.items():
            label = label.replace(key, val)

        casing = kwds.pop("casing", False)
        if casing:
            label = getattr(label, casing)()
        kwds[label_key] = label if label != "None" else None
        return kwds

    def _update_labels(self, state_ds, ax):
        for label in ITEMS["label"]:
            label_kwds = load_defaults(f"{label}_kwds", state_ds)
            key = label if "title" not in label else "label"
            label_kwds = self._update_text(label_kwds, key)
            if label == "subtitle":
                label = "title"
            getattr(ax, f"set_{label}")(**label_kwds)

        if state_ds.attrs.get("note_kwds"):
            note_kwds = load_defaults("note_kwds", state_ds, transform=ax.transAxes)
            ax.text(**note_kwds)

        caption = state_ds.attrs.get("caption_kwds", {}).get("s")
        if caption is not None:
            y = -0.28 + -0.02 * caption.count("\n")
            caption_kwds = load_defaults(
                "caption_kwds", state_ds, transform=ax.transAxes, y=y
            )
            ax.text(**caption_kwds)

    def _add_state_labels(self, state_ds, ax):
        state_label = pop(state_ds, "state_label", get=-1)
        if state_label is None:
            return
        state_label = to_pydt(state_label)
        state_base = state_ds.attrs["base_kwds"].get("state")

        state_kwds = load_defaults("state_kwds", state_ds, text=state_label)
        state_kwds = self._update_text(state_kwds, "text", base=state_base)
        ax.annotate(**state_kwds)

    @staticmethod
    def _get_color(overlay_ds, plot):
        if isinstance(plot, list):
            plot = plot[0]

        if "cmap" in overlay_ds.attrs["plot_kwds"]:
            color = "black"
        else:
            try:
                color = plot.get_color()
            except AttributeError:
                try:
                    color = plot.get_facecolor()
                except AttributeError:
                    color = plot[0].get_facecolor()

                if len(to_1d(color)) < 1:
                    try:
                        color = plot.get_edgecolor()
                    except AttributeError:
                        color = plot[0].get_edgecolor()

            if isinstance(color, np.ndarray):
                color = color[0]
        return color

    def _plot_chart(self, overlay_ds, ax, chart, xs, ys, plot_kwds):
        if chart == "scatter":
            plot = ax.scatter(xs, ys, **plot_kwds)
        elif chart == "line":
            plot = ax.plot(xs, ys, **plot_kwds)
        elif chart.startswith("bar"):
            plot = getattr(ax, chart)(xs, ys, **plot_kwds)
        color = self._get_color(overlay_ds, plot)

        if xs.ndim == 2:  # a grouped batch with same label
            for p in plot:
                p.set_color(color)
        return plot, color

    def _plot_trails(
        self,
        overlay_ds,
        ax,
        chart,
        color,
        xs,
        ys,
        x_trails,
        y_trails,
        x_discrete_trails,
        y_discrete_trails,
        trail_plot_kwds,
    ):
        all_none = (
            x_trails is None
            and y_trails is None
            and x_discrete_trails is None
            and y_discrete_trails is None
        )
        if all_none:
            return
        preset_kwds = load_defaults("preset_kwds", overlay_ds, base_chart="trail")
        chart = preset_kwds.pop("chart", "both")
        expire = preset_kwds.pop("expire")
        stride = preset_kwds.pop("stride")
        line_preset_kwds = preset_kwds.copy()

        if chart in ["scatter", "both"]:
            non_nan_indices = np.where(~np.isnan(x_discrete_trails))
            x_discrete_trails = x_discrete_trails[non_nan_indices][
                -expire - 1 :: stride
            ]
            y_discrete_trails = y_discrete_trails[non_nan_indices][
                -expire - 1 :: stride
            ]
            preset_kwds.update(**trail_plot_kwds)
            preset_kwds = {
                key: val[non_nan_indices][-expire - 1 :: stride]
                if not is_scalar(val)
                else val
                for key, val in preset_kwds.items()
            }
            preset_kwds["label"] = "_nolegend_"
            ax.scatter(x_discrete_trails, y_discrete_trails, **preset_kwds)

        if chart in ["line", "both"]:
            x_trails = x_trails[-expire * self._num_steps - 1 :]
            y_trails = y_trails[-expire * self._num_steps - 1 :]
            line_preset_kwds["label"] = "_nolegend_"
            ax.plot(x_trails, y_trails, color=color, **line_preset_kwds)

    def _plot_deltas(
        self,
        overlay_ds,
        ax,
        chart,
        x_centers,
        y_centers,
        deltas,
        delta_labels,
        color,
    ):
        if deltas is None:
            return
        preset_kwds = load_defaults("preset_kwds", overlay_ds, base_chart="delta")
        if chart == "bar":
            preset_kwds["yerr"] = deltas
            x_inlines = x_centers
            y_inlines = y_centers + np.abs(deltas)
        else:
            y_centers, x_centers = x_centers, y_centers
            x_inlines = x_centers + np.abs(deltas)
            y_inlines = y_centers
            preset_kwds["xerr"] = deltas

        preset_kwds.pop("bar_label", None)
        with warnings.catch_warnings():
            # the last item is NaN which is expected
            warnings.simplefilter("ignore", UserWarning)
            ax.errorbar(x_centers, y_centers, **preset_kwds)
        self._add_inline_labels(
            overlay_ds,
            ax,
            chart,
            x_inlines,
            y_inlines,
            delta_labels,
            color,
            base_key="delta",
            inline_key="preset_inline_kwds",
        )

    def _plot_scans(
        self,
        overlay_ds,
        ax,
        chart,
        scan_xs,
        scan_ys,
        scan_x_0_inline_labels,
        scan_x_1_inline_labels,
        scan_y_0_inline_labels,
        scan_y_1_inline_labels,
    ):
        if scan_xs is None and scan_ys is None:
            return

        preset_kwds = load_defaults("preset_kwds", overlay_ds, base_chart="scan")
        inline_loc = preset_kwds.pop("inline_loc", None)
        preset_kwds.pop("stride")

        self._plot_ref_chart(
            overlay_ds, ax, chart, scan_xs, None, scan_ys, None, preset_kwds
        )

        if scan_x_0_inline_labels is not None:
            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                scan_xs,
                inline_loc,
                scan_x_0_inline_labels,
                "black",
                ha="right",
                base_key="grid_scan_x_diff_inline",
                inline_key="preset_inline_kwds",
                xytext=(-18, 0),
                clip=True,
            )
            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                scan_xs,
                inline_loc,
                scan_x_1_inline_labels,
                "black",
                ha="left",
                base_key="grid_scan_x_diff_inline",
                inline_key="preset_inline_kwds",
                xytext=(18, 0),
                clip=True,
            )
        else:
            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                inline_loc,
                scan_ys,
                scan_y_0_inline_labels,
                "black",
                va="bottom",
                base_key="grid_scan_y_diff_inline",
                inline_key="preset_inline_kwds",
                xytext=(0, 18),
                clip=True,
            )
            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                inline_loc,
                scan_ys,
                scan_y_1_inline_labels,
                "black",
                va="top",
                base_key="grid_scan_y_diff_inline",
                inline_key="preset_inline_kwds",
                xytext=(0, -18),
                clip=True,
            )

    def _add_remarks(self, state_ds, ax, chart, xs, ys, remarks, color):
        if remarks is None:
            return

        for x, y, remark in zip(xs, ys, remarks):
            if remark == "":
                continue
            remark = to_num(remark)
            remark_inline_kwds = dict(
                text=remark,
                xy=(x, y),
                color=color,
                path_effects=self._path_effects,
            )
            remark_inline_kwds = load_defaults(
                "remark_inline_kwds", state_ds, **remark_inline_kwds
            )
            remark_inline_kwds = self._update_text(
                remark_inline_kwds, "text", base=remark
            )
            ax.annotate(**remark_inline_kwds)

            remark_kwds = load_defaults(
                "remark_plot_kwds", state_ds, x=x, y=y, color=color
            )
            ax.scatter(**remark_kwds)

    def _add_inline_labels(
        self,
        overlay_ds,
        ax,
        chart,
        xs,
        ys,
        inline_labels,
        color,
        base_key="inline",
        inline_key="inline_kwds",
        xytext=(0, 5),
        ha=None,
        va=None,
        clip=False,
    ):
        if inline_labels is None:
            return
        inline_base = overlay_ds.attrs["base_kwds"].get(base_key)

        if ha is None and va is None:
            ha = "center"
            va = "center"
            if chart == "barh":
                ha = "left" if base_key != "preset" else "right"
                xytext = xytext[::-1]
                if base_key != "delta":
                    xs, ys = ys, xs
            elif chart == "bar":
                va = "bottom" if base_key != "bar" else "top"
            elif chart in ["line", "scatter"]:
                ha = "left"
                va = "bottom"
        elif va is None:
            va = "center"
        elif ha is None:
            ha = "center"

        inline_kwds = dict(
            ha=ha,
            va=va,
            color=color,
            xytext=xytext,
            path_effects=self._path_effects,
        )
        if clip:
            inline_kwds["annotation_clip"] = True
        inline_kwds = load_defaults(inline_key, overlay_ds, **inline_kwds)

        # https://stackoverflow.com/questions/25416600/
        # transform is mangled in matplotlib for annotate
        transform = inline_kwds.pop("transform", None)
        if transform is not None:
            inline_kwds["xycoords"] = transform._as_mpl_transform(ax)

        xs = to_1d(xs)
        ys = to_1d(ys)
        inline_labels = to_1d(inline_labels)
        if chart == "line":
            xs = xs[[-1]]
            ys = ys[[-1]]
            inline_labels = inline_labels[[-1]]
        for x, y, inline_label in zip(xs, ys, inline_labels):
            if str(inline_label) == "nan":
                inline_label = "?"

            if str(inline_label) == "" or pd.isnull(x) or pd.isnull(y):
                continue
            inline_kwds["text"] = inline_label
            inline_kwds = self._update_text(inline_kwds, "text", base=inline_base)
            ax.annotate(xy=(x, y), **inline_kwds)

    @staticmethod
    def _reshape_batch(array, chart, get=-1):
        if array is None:
            return array

        if get is not None and chart not in ["bar", "line"]:
            if array.ndim == 2:
                array = array[:, get]
            else:
                array = array[[get]]

        return array.T

    @staticmethod
    def _groupby_key(data, key):
        if key is not None and key != "":
            if isinstance(data, xr.Dataset):
                if key not in data.dims and key not in data.data_vars:
                    data = data.expand_dims(key)
                elif key in data.data_vars and is_scalar(data[key]):
                    data[key] = data[key].expand_dims("group")
            try:
                return data.groupby(key, sort=False)
            except TypeError:  # xarray doesn't have sort keyword
                return data.groupby(key)
        else:
            return zip([""], [data])

    def _get_iter_ds(self, state_ds):
        if len(state_ds.data_vars) == 0:
            return zip([], []), -1
        elif any(group for group in to_1d(state_ds["group"])):
            state_ds = state_ds.drop_vars("label").rename({"group": "label"})
            get = None
            key = "label"
        else:
            state_ds = state_ds.drop_vars("group", errors="ignore")
            get = -1
            key = "item"
        iter_ds = self._groupby_key(state_ds, key)
        return iter_ds, get

    def _update_colorbar(self, state_ds, ax, mappable):
        if "colorbar_kwds" not in state_ds.attrs:
            return

        colorbar_kwds = load_defaults("colorbar_kwds", state_ds, ax=ax)
        if not colorbar_kwds.pop("show", False) or mappable is None:
            return

        divider = make_axes_locatable(ax)
        if colorbar_kwds["orientation"] == "vertical":
            cax = divider.new_horizontal(size="2%", pad=0.1, axes_class=plt.Axes)
        else:
            cax = divider.new_vertical(size="2%", pad=0.1, axes_class=plt.Axes)
        ax.figure.add_axes(cax)

        colorbar = plt.colorbar(mappable, cax=cax, **colorbar_kwds)
        clabel_kwds = load_defaults("clabel_kwds", state_ds)
        clabel_kwds = self._update_text(clabel_kwds, "label")
        if colorbar_kwds["orientation"] == "vertical":
            clabel_kwds["ylabel"] = clabel_kwds.pop("label")
            cticks_kwds = {"axis": "y"}
            long_axis = cax.yaxis
            cax.set_ylabel(**clabel_kwds)
        else:
            clabel_kwds["xlabel"] = clabel_kwds.pop("label")
            cticks_kwds = {"axis": "x"}
            long_axis = cax.xaxis
            cax.set_xlabel(**clabel_kwds)

        if "cticks" in state_ds.attrs["base_kwds"]:
            cticks_base = state_ds.attrs["base_kwds"]["cticks"]
        else:
            cticks_base = state_ds.attrs["base_kwds"]["grid_cticks"]
        cticks_kwds = load_defaults("cticks_kwds", state_ds, **cticks_kwds)
        cticks_kwds.pop("num_colors", None)
        cticks_kwds.pop("num_ticks", None)
        cticks_kwds = self._update_text(
            cticks_kwds, "ticks", base=cticks_base, apply_format=False
        )
        cformat = cticks_kwds.pop("format")
        cformatter = FormatStrFormatter(f"%{cformat}")
        cticks = cticks_kwds.pop("ticks")
        ctick_labels = cticks_kwds.pop("tick_labels", None)
        if cticks is not None:
            colorbar.set_ticks(np.array(cticks).astype(float))
        if ctick_labels is not None:
            long_axis.set_ticklabels(ctick_labels)
        else:
            long_axis.set_major_formatter(cformatter)
        cax.tick_params(**cticks_kwds)

    @staticmethod
    def _strip_dict(kwds):
        """Primarily to remove empty color strings."""
        stripped_kwds = {}
        for key, val in kwds.items():
            unique_vals = np.unique(val)
            if len(unique_vals) == 1:
                unique_val = unique_vals.item()
                if unique_val == "" or pd.isnull(unique_val):
                    continue
            stripped_kwds[key] = val
        return stripped_kwds

    @staticmethod
    def _subset_vars(state_ds, key=None):
        if key is None:
            state_ds = state_ds[
                [
                    var
                    for var in state_ds.data_vars
                    if not var.startswith("ref_") and not var.startswith("grid_")
                ]
            ]
        else:
            keyu = f"{key}_"
            extra_vars = [f"{keyu}item"]
            if key == "grid":
                extra_vars += ["grid_x", "grid_y"]
            state_ds = state_ds[
                [var for var in state_ds.data_vars if var.startswith(keyu)]
            ]
            state_ds = state_ds.rename(
                {
                    var: var.replace(f"{keyu}", "")
                    for var in list(state_ds) + extra_vars
                    if var in state_ds
                }
            )
        return state_ds

    def _process_base_vars(self, state_ds, ax):
        base_state_ds = self._subset_vars(state_ds)

        iter_ds, get = self._get_iter_ds(base_state_ds)
        mappable = None
        for _, overlay_ds in iter_ds:
            overlay_ds = overlay_ds.where(overlay_ds["chart"] != "", drop=True)
            if len(to_1d(overlay_ds["state"])) == 0:
                continue
            chart = pop(overlay_ds, "chart", get=0)
            if pd.isnull(chart):
                chart = str(chart)
                continue

            xs = self._reshape_batch(pop(overlay_ds, "x"), chart, get=get)
            ys = self._reshape_batch(pop(overlay_ds, "y"), chart, get=get)

            inline_labels = self._reshape_batch(
                pop(overlay_ds, "inline_label"), chart, get=get
            )

            x_trails = self._reshape_batch(pop(overlay_ds, "x_trail"), chart, get=None)
            y_trails = self._reshape_batch(pop(overlay_ds, "y_trail"), chart, get=None)

            x_discrete_trails = self._reshape_batch(
                pop(overlay_ds, "x_discrete_trail"), chart, get=None
            )
            y_discrete_trails = self._reshape_batch(
                pop(overlay_ds, "y_discrete_trail"), chart, get=None
            )

            x_centers = self._reshape_batch(pop(overlay_ds, "x_center"), chart)
            y_centers = self._reshape_batch(pop(overlay_ds, "y_center"), chart)

            deltas = self._reshape_batch(pop(overlay_ds, "delta"), chart)
            delta_labels = self._reshape_batch(pop(overlay_ds, "delta_label"), chart)

            bar_labels = self._reshape_batch(pop(overlay_ds, "bar_label"), chart)

            remarks = self._reshape_batch(pop(overlay_ds, "remark"), chart)

            trail_plot_kwds = {
                var: self._reshape_batch(pop(overlay_ds, var), chart, get=None)
                for var in list(overlay_ds.data_vars)
            }
            trail_plot_kwds = self._strip_dict(
                load_defaults("plot_kwds", overlay_ds, **trail_plot_kwds)
            )
            if "alpha" in trail_plot_kwds:
                trail_plot_kwds["alpha"] = to_scalar(trail_plot_kwds["alpha"])

            plot_kwds = self._strip_dict(
                {key: to_scalar(val) for key, val in trail_plot_kwds.items()}
            )

            if "label" in plot_kwds:
                plot_kwds["label"] = to_scalar(plot_kwds["label"])
            if "zorder" not in plot_kwds:
                plot_kwds["zorder"] = 2

            plot, color = self._plot_chart(overlay_ds, ax, chart, xs, ys, plot_kwds)
            if "c" in plot_kwds:
                mappable = plot

            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                xs,
                ys,
                inline_labels,
                color,
                xytext=(5, 5) if not chart.startswith("bar") else (0, 5),
            )

            self._plot_trails(
                overlay_ds,
                ax,
                chart,
                color,
                xs,
                ys,
                x_trails,
                y_trails,
                x_discrete_trails,
                y_discrete_trails,
                trail_plot_kwds,
            )

            self._plot_deltas(
                overlay_ds,
                ax,
                chart,
                x_centers,
                y_centers,
                deltas,
                delta_labels,
                color,
            )

            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                xs,
                ys,
                bar_labels,
                "black",
                base_key="preset",
                inline_key="preset_inline_kwds",
                xytext=(0, -5) if chart == "barh" else (0, -15),
            )

            self._add_remarks(overlay_ds, ax, chart, xs, ys, remarks, color)

        return mappable

    def _process_grid_vars(self, state_ds, ax):
        grid_state_ds = self._subset_vars(state_ds, "grid")
        grid_iter_ds, _ = self._get_iter_ds(grid_state_ds)
        mappable = None
        for _, overlay_ds in grid_iter_ds:
            label = pop(overlay_ds, "label", get=0) or "_nolegend_"
            chart = pop(overlay_ds, "chart", get=0)

            xs = pop(overlay_ds, "x")
            ys = pop(overlay_ds, "y")
            cs = pop(overlay_ds, "c")

            scan_xs = pop(overlay_ds, "scan_x")
            scan_ys = pop(overlay_ds, "scan_y")
            scan_chart = "axvline" if scan_xs is not None else "axhline"
            scan_x_0_inline_labels = pop(overlay_ds, "scan_x_0_inline_label")
            scan_x_1_inline_labels = pop(overlay_ds, "scan_x_1_inline_label")
            pop(overlay_ds, "scan_x_diff_inline_label")
            scan_y_0_inline_labels = pop(overlay_ds, "scan_y_0_inline_label")
            scan_y_1_inline_labels = pop(overlay_ds, "scan_y_1_inline_label")
            pop(overlay_ds, "scan_y_diff_inline_label")

            inline_xs = pop(overlay_ds, "inline_x")
            inline_ys = pop(overlay_ds, "inline_y")
            inline_labels = pop(overlay_ds, "inline_label")

            plot_kwds = self._strip_dict(
                {var: pop(overlay_ds, var, get=0) for var in list(overlay_ds.data_vars)}
            )
            if chart in ["contourf", "contour"] and "levels" not in plot_kwds:
                plot_kwds["levels"] = overlay_ds.attrs["cticks"].get("ticks")
            plot_kwds = load_defaults(
                "grid_plot_kwds",
                overlay_ds,
                base_chart=chart,
                label=label,
                shading="auto",
                **plot_kwds,
            )

            mappable = getattr(ax, chart)(xs, ys, cs, **plot_kwds)

            self._plot_scans(
                overlay_ds,
                ax,
                scan_chart,
                scan_xs,
                scan_ys,
                scan_x_0_inline_labels,
                scan_x_1_inline_labels,
                scan_y_0_inline_labels,
                scan_y_1_inline_labels,
            )
            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                inline_xs,
                inline_ys,
                inline_labels,
                "black",
                base_key="grid_inline",
                inline_key="grid_inline_kwds",
            )

        return mappable

    def _plot_ref_chart(self, overlay_ds, ax, chart, x0s, x1s, y0s, y1s, plot_kwds):
        if plot_kwds.get("transform"):
            if chart in ["rectangle", "scatter"]:
                pass

            if "axh" in chart:
                x0s = -179.99
                x1s = 179.99
                if "line" in chart:
                    y1s = y0s
            elif "axv" in chart:
                y0s = -89.99
                y1s = 89.99
                if "line" in chart:
                    x1s = x0s

            if "span" in chart:
                chart = "rectangle"
            else:
                plot_kwds["scalex"] = False
                plot_kwds["scaley"] = False
                chart = "geoline"

        if chart == "scatter":
            plot = ax.scatter(x0s, y0s, **plot_kwds)
        elif chart == "rectangle":
            width = x1s - x0s
            height = y1s - y0s
            rectangle = Rectangle(
                xy=[x0s, y0s], width=width, height=height, **plot_kwds
            )
            plot = ax.add_patch(rectangle)
        elif chart == "axhspan":
            plot = ax.axhspan(y0s, y1s, **plot_kwds)
        elif chart == "axvspan":
            plot = ax.axvspan(x0s, x1s, **plot_kwds)
        elif chart == "geoline":
            plot = ax.plot([x0s, x1s], [y0s, y1s], **plot_kwds)
        elif chart == "axvline":
            plot = ax.axvline(x0s, **plot_kwds)
        elif chart == "axhline":
            plot = ax.axhline(y0s, **plot_kwds)
        color = self._get_color(overlay_ds, plot)
        return color

    def _process_ref_vars(self, state_ds, ax):
        ref_state_ds = self._subset_vars(state_ds, "ref")
        ref_iter_ds, get = self._get_iter_ds(ref_state_ds)
        for _, overlay_ds in ref_iter_ds:
            label = pop(overlay_ds, "label", get=-1) or "_nolegend_"
            chart = pop(overlay_ds, "chart", get=-1)

            x0s = pop(overlay_ds, "x0", get=get)
            x1s = pop(overlay_ds, "x1", get=get)
            y0s = pop(overlay_ds, "y0", get=get)
            y1s = pop(overlay_ds, "y1", get=get)

            inline_loc = pop(overlay_ds, "inline_loc", get=-1)
            inline_labels = pop(overlay_ds, "inline_label", get=-1)

            if chart == "rectangle":
                inline_xs = x0s
                inline_ys = y0s
                xytext = (0, 5)
            if chart == "axvline":
                inline_xs = x0s
                inline_ys = inline_loc
                xytext = (-5, 5)
            elif chart == "axhline":
                inline_xs = inline_loc
                inline_ys = y0s
                xytext = (0, 5)
            elif chart == "axvspan":
                inline_xs = x0s
                inline_ys = inline_loc
                xytext = (-5, 5)
            elif chart == "axhspan":
                inline_xs = inline_loc
                inline_ys = y0s
                xytext = (0, 5)
            elif chart == "scatter":
                inline_xs = x0s
                inline_ys = y0s
                xytext = (0, 5)

            plot_kwds = self._strip_dict(
                {var: pop(overlay_ds, var, get=0) for var in list(overlay_ds.data_vars)}
            )
            plot_kwds = load_defaults(
                "ref_plot_kwds",
                overlay_ds,
                base_chart=chart,
                label=label,
                **plot_kwds,
            )

            color = self._plot_ref_chart(
                overlay_ds, ax, chart, x0s, x1s, y0s, y1s, plot_kwds
            )

            self._add_inline_labels(
                overlay_ds,
                ax,
                chart,
                inline_xs,
                inline_ys,
                inline_labels,
                color,
                base_key="ref_inline",
                inline_key="ref_inline_kwds",
                xytext=xytext,
                ha="right",
                va="bottom",
            )

    def _prep_figure(self, rows, cols):
        figure_kwds = self._canvas_kwds["figure_kwds"]
        if figure_kwds.get("figsize") is None:
            width = 7.5 + 7.5 * (cols - 1)
            height = 5 + 5 * (rows - 1)
            figsize = (width, height)
        else:
            figsize = None
        figure_kwds = load_defaults("figure_kwds", figure_kwds, figsize=figsize)
        figure = plt.figure(**figure_kwds)

        if "suptitle_kwds" in self._canvas_kwds:
            suptitle_kwds = load_defaults(
                "suptitle_kwds", self._canvas_kwds["suptitle_kwds"]
            )
            suptitle_kwds = self._update_text(suptitle_kwds, "t")
            figure.suptitle(**suptitle_kwds)
        return figure

    def _prep_axes(self, state_ds, rows, cols, irowcol):
        axes_kwds = load_defaults("axes_kwds", state_ds)
        axes_kwds["projection"] = pop(state_ds, "projection", squeeze=True)

        style = axes_kwds.pop("style", "")
        if style == "minimal":
            for axis in ["x", "y"]:
                axis_min = float(state_ds[axis].values.min())
                axis_max = float(state_ds[axis].values.max())
                axis_lim = axes_kwds.get(f"{axis}lim", None)
                if axis_lim is not None:
                    axis_min = max(axis_min, axis_lim[0])
                    axis_max = min(axis_max, axis_lim[1])
                axes_kwds[f"{axis}ticks"] = to_pydt(axis_min, axis_max)
        elif style == "bare":
            axes_kwds["xticks"] = []
            axes_kwds["yticks"] = []

        ax = plt.subplot(rows, cols, irowcol, **axes_kwds)

        if style == "bare":
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
            ax.set_frame_on(False)
        return ax

    def _update_grid(self, state_ds, ax):
        show = state_ds.attrs["grid_kwds"].get("show")
        if isinstance(show, str):
            state_ds.attrs["grid_kwds"]["show"] = True
            state_ds.attrs["grid_kwds"]["axis"] = show

        grid_kwds = load_defaults("grid_kwds", state_ds)
        grid = grid_kwds.pop("show", False)
        if not grid:
            return
        if "transform" in grid_kwds:
            axis = grid_kwds.pop("axis", None)
            gridlines = ax.gridlines(**grid_kwds)

            if "PlateCarree" in str(grid_kwds["transform"]):
                gridlines.xlines = False
                gridlines.ylines = False

            if axis == "x":
                gridlines.top_labels = False
                gridlines.bottom_labels = False
            elif axis == "y":
                gridlines.left_labels = False
                gridlines.right_labels = False
            else:
                gridlines.top_labels = False
                gridlines.right_labels = False
        else:
            ax.grid(**grid_kwds)
            ax.set_axisbelow(True)
            gridlines = None
        return gridlines

    def _update_limits(self, state_ds, ax):
        limits = {
            var: pop(state_ds, var, get=-1)
            for var in list(state_ds.data_vars)
            if var[1:4] == "lim"
        }

        margins_kwds = load_defaults("margins_kwds", state_ds)
        transform = margins_kwds.pop("transform", None)
        if transform is not None:
            from cartopy import feature as cfeature

            unset_limits = all(limit is None for limit in limits)
            if state_ds.attrs["limits_kwds"].get("worldwide"):
                ax.set_global()
            elif not unset_limits:
                ax.set_extent(
                    [
                        limits.get("xlim0s", -179),
                        limits.get("xlim1s", 179),
                        limits.get("ylim0s", -89),
                        limits.get("ylim1s", 89),
                    ],
                    transform,
                )
            for feature in CONFIGURABLES["geo"]:
                if feature in ["projection", "crs"]:
                    continue
                feature_key = f"{feature}_kwds"
                feature_kwds = load_defaults(feature_key, state_ds)
                if feature_kwds.pop(feature, False):
                    feature_obj = getattr(cfeature, feature.upper())
                    ax.add_feature(feature_obj, **feature_kwds)
        else:
            for axis in ["x", "y"]:
                axis_lim0 = to_scalar(limits.get(f"{axis}lim0s"))
                axis_lim1 = to_scalar(limits.get(f"{axis}lim1s"))
                if axis_lim0 is not None or axis_lim1 is not None:
                    getattr(ax, f"set_{axis}lim")(to_pydt(axis_lim0, axis_lim1))
        ax.margins(**margins_kwds)

    def _update_ticks(self, state_ds, ax, gridlines):
        chart = to_scalar(state_ds.get("chart", ""))
        tick_labels = pop(state_ds, "tick_label")

        xticks_base = state_ds.attrs["base_kwds"].get("xticks")
        xticks_kwds = load_defaults("xticks_kwds", state_ds, labels=tick_labels)
        xticks_kwds = self._update_text(
            xticks_kwds, "labels", base=xticks_base, apply_format=False
        )
        xticks = xticks_kwds.pop("ticks", None)
        xformat = xticks_kwds.pop("format", "g")
        xticks_labels = xticks_kwds.pop("labels")
        x_is_datetime = xticks_kwds.pop("is_datetime", False)
        x_is_str = xticks_kwds.pop("is_str", False)

        yticks_base = state_ds.attrs["base_kwds"].get("yticks")
        yticks_kwds = load_defaults("yticks_kwds", state_ds, labels=tick_labels)
        yticks_kwds = self._update_text(
            yticks_kwds, "labels", base=yticks_base, apply_format=False
        )
        yticks = yticks_kwds.pop("ticks", None)
        yformat = yticks_kwds.pop("format", "g")
        yticks_labels = yticks_kwds.pop("labels")
        y_is_datetime = yticks_kwds.pop("is_datetime", False)
        y_is_str = yticks_kwds.pop("is_str", False)

        if gridlines is not None:  # geoaxes
            from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter

            gridlines.yformatter = LatitudeFormatter()
            gridlines.xformatter = LongitudeFormatter()
            for key in ["axis", "which", "length", "labelsize"]:
                if key == "labelsize":
                    xticks_kwds["size"] = xticks_kwds.pop(
                        key, defaults["ticks_kwds"]["labelsize"]
                    )
                    yticks_kwds["size"] = yticks_kwds.pop(
                        key, defaults["ticks_kwds"]["labelsize"]
                    )
                else:
                    xticks_kwds.pop(key, "")
                    yticks_kwds.pop(key, "")
            gridlines.ylabel_style = yticks_kwds
            gridlines.xlabel_style = xticks_kwds

            if xticks is not None:
                gridlines.xlocator = FixedLocator(xticks)
            if yticks is not None:
                gridlines.ylocator = FixedLocator(yticks)
        else:
            preset = state_ds.attrs["preset_kwds"].get("preset")
            is_not_series = preset != "series"
            if chart.startswith("bar") and is_not_series:
                preset_kwds = load_defaults("preset_kwds", state_ds, base_chart=preset)
                xs = pop(state_ds, "x")
                limit = preset_kwds.get("limit", None)
                limit1 = len(xs) + 0.5
                limit0 = limit1 - limit if limit is not None else -1
                if chart == "bar":
                    ax.set_xticks(xs)
                    ax.set_xticklabels(xticks_labels)
                    if limit0 >= 0:
                        ax.set_xlim(limit0, limit1)
                elif chart == "barh":
                    ax.set_yticks(xs)
                    ax.set_yticklabels(yticks_labels)
                    if limit0 >= 0:
                        ax.set_ylim(limit0, limit1)
            else:
                if not x_is_datetime and not x_is_str:
                    xformatter = FormatStrFormatter(f"%{xformat}")
                    ax.xaxis.set_major_formatter(xformatter)
                elif not x_is_str:
                    xlocator = AutoDateLocator(minticks=5, maxticks=10)
                    xformatter = ConciseDateFormatter(xlocator)
                    ax.xaxis.set_major_locator(xlocator)
                    ax.xaxis.set_major_formatter(xformatter)

                if not y_is_datetime and not y_is_str:
                    yformatter = FormatStrFormatter(f"%{yformat}")
                    ax.yaxis.set_major_formatter(yformatter)
                elif not y_is_str:
                    ylocator = AutoDateLocator(minticks=5, maxticks=10)
                    yformatter = ConciseDateFormatter(ylocator)
                    ax.yaxis.set_major_locator(ylocator)
                    ax.yaxis.set_major_formatter(yformatter)

                if xticks is not None:
                    ax.set_xticks(xticks)
                if yticks is not None:
                    ax.set_yticks(yticks)
            ax.tick_params(**xticks_kwds)
            ax.tick_params(**yticks_kwds)

    def _update_legend(self, state_ds, ax):
        handles, legend_labels = ax.get_legend_handles_labels()
        legend_items = dict(zip(legend_labels, handles))
        num_labels = len(legend_labels)
        ncol = int(num_labels / 5) or 1
        legend_kwds = dict(
            handles=legend_items.values(), labels=legend_items.keys(), ncol=ncol
        )
        show = state_ds.attrs["legend_kwds"].get("show")
        if isinstance(show, str):
            state_ds.attrs["legend_kwds"]["show"] = True
            state_ds.attrs["legend_kwds"]["loc"] = show
        legend_kwds = load_defaults("legend_kwds", state_ds, show=show, **legend_kwds)
        legend_kwds.pop("sortby", None)

        if not legend_labels or not legend_kwds.get("show"):
            return

        legend = ax.legend(
            **{
                key: val
                for key, val in legend_kwds.items()
                if key not in ["replacements", "casing", "format", "show"]
            }
        )

        s_base = state_ds.attrs["base_kwds"].get("s")
        if s_base:
            for legend_handle in legend.legendHandles:
                legend_handle.set_sizes([s_base])

        for legend_label in legend.get_texts():
            legend_label.set_path_effects(self._path_effects)
        legend.get_frame().set_linewidth(0)

    def _draw_subplot(self, state_ds, ax):
        self._update_labels(state_ds, ax)
        self._add_state_labels(state_ds, ax)
        self._update_limits(state_ds, ax)

        base_mappable = self._process_base_vars(state_ds, ax)
        grid_mappable = self._process_grid_vars(state_ds, ax)
        self._process_ref_vars(state_ds, ax)

        gridlines = self._update_grid(state_ds, ax)
        self._update_ticks(state_ds, ax, gridlines)
        self._update_legend(state_ds, ax)

        if grid_mappable is not None:
            mappable = grid_mappable
        elif base_mappable is not None:
            mappable = base_mappable
        else:
            mappable = None
        self._update_colorbar(state_ds, ax, mappable)

    def _apply_hooks(self, state_ds, figure, ax):  # TODO: implement
        hooks = state_ds.attrs.get("hooks_kwds", {}).get("hooks", [])
        for hook in hooks:
            if not callable(hook):
                continue
            hook(figure, ax)

    def _update_watermark(self, figure):
        watermark_kwds = load_defaults(
            "watermark_kwds", self._canvas_kwds["watermark_kwds"]
        )
        if watermark_kwds["s"]:
            figure.text(**watermark_kwds)

    def _update_spacing(self, state_ds_rowcols):
        top = bottom = wspace = None
        for state_ds in state_ds_rowcols:
            suptitle = state_ds.attrs.get("suptitle_kwds", {}).get("t")
            if suptitle is not None:
                top = 0.825

            caption = state_ds.attrs.get("caption_kwds", {}).get("s")
            if caption is not None:
                bottom = 0.2 + 0.008 * self.caption.count("\n")

            clabel = state_ds.attrs.get("clabel_kwds", {}).get("text")
            if clabel is not None:
                wspace = 0.25

        spacing_kwds = self._canvas_kwds["spacing_kwds"]
        if "spacing" in spacing_kwds:
            spacing_kwds.update(**spacing_kwds.pop("spacing"))
        spacing_kwds = load_defaults(
            "spacing_kwds",
            spacing_kwds,
            top=top,
            bottom=bottom,
            wspace=wspace,
        )
        plt.subplots_adjust(**spacing_kwds)

    def _buffer_frame(self, state):
        buf = BytesIO()
        savefig_kwds = load_defaults("savefig_kwds", self._canvas_kwds["savefig_kwds"])
        try:
            plt.savefig(buf, **savefig_kwds)
            buf.seek(0)
            return buf
        except Exception as e:
            error_msg = f"Failed to render state={state} due to {e}!"
            if self.debug:
                raise RuntimeError(error_msg)
            else:
                warnings.warn(error_msg)
            return
        finally:
            plt.close()

    @dask.delayed()
    def _draw_frame(self, state_ds_rowcols, rows, cols):
        figure = self._prep_figure(rows, cols)
        for irowcol, state_ds in enumerate(state_ds_rowcols, 1):
            ax = self._prep_axes(state_ds, rows, cols, irowcol)
            self._draw_subplot(state_ds, ax)
            self._apply_hooks(state_ds, figure, ax)
        self._update_watermark(figure)
        self._update_spacing(state_ds_rowcols)
        state = pop(state_ds, "state", get=-1)
        buf = self._buffer_frame(state)
        return buf

    def _create_frames(self, data, rows, cols):
        num_states = self._canvas_kwds["animate_kwds"].pop("num_states")
        states = self._canvas_kwds["animate_kwds"].pop("states")
        if states is not None:
            negative_indices = states < 0
            states[negative_indices] = num_states + states[negative_indices] - 1
        else:
            states = srange(num_states)
        states = np.array(states).astype(int)

        jobs = []
        for state in states:
            state_ds_rowcols = []
            for ds in data.values():
                is_stateless = "state" not in ds.dims
                is_line = np.any(ds.get("chart", "") == "line")
                is_trail = ds.attrs["preset_kwds"].get("preset") == "trail"
                is_series = ds.attrs["preset_kwds"].get("preset", "") == "series"
                if is_stateless:
                    ds_sel = ds
                elif is_line or is_trail or is_series:
                    ds_sel = ds.sel(state=slice(None, state))
                    # this makes legend labels appear in order if values exist
                    if "item" in ds_sel.dims:
                        ds_last = ds_sel.isel(state=-1)
                        not_nan_items = ds_last["y"].dropna("item")["item"]
                        ds_sel = ds_sel.sel(item=not_nan_items.values)
                        ds_sel["item"] = srange(len(ds_sel["item"]))
                else:
                    ds_sel = ds.sel(state=state)

                state_ds_rowcols.append(ds_sel)
            job = self._draw_frame(state_ds_rowcols, rows, cols)
            jobs.append(job)

        scheduler = "single-threaded" if self.debug else None
        compute_kwds = load_defaults(
            "compute_kwds",
            self._canvas_kwds["compute_kwds"],
            scheduler=scheduler,
        )
        num_workers = compute_kwds["num_workers"]
        if num_states < num_workers:
            warnings.warn(
                f"There is less states to process than the number of workers!"
                f"Setting workers={num_states} from {num_workers}."
            )
            num_workers = num_states

        if num_workers == 1:
            if compute_kwds["scheduler"] != "single-threaded":
                warnings.warn(
                    "Only 1 worker found; setting scheduler='single-threaded'"
                )
            compute_kwds["scheduler"] = "single-threaded"
        elif num_workers > 1 and compute_kwds["scheduler"] != "processes":
            if compute_kwds["scheduler"] != "processes":
                warnings.warn("Found multiple workers; setting scheduler='processes'")
            compute_kwds["scheduler"] = "processes"

        with dask.diagnostics.ProgressBar(minimum=1):
            buf_list = [
                buf for buf in dask.compute(jobs, **compute_kwds)[0] if buf is not None
            ]
        return buf_list

    def _write_rendered(self, buf_list, durations):
        # TODO: breakdown function
        delays_kwds = {}
        if durations is not None:
            durations = getattr(durations, durations.attrs["aggregate"])(
                "item", keep_attrs=True
            )
            durations = durations.where(
                durations > 0, durations.attrs["transition_frames"]
            ).squeeze()
            durations = to_1d(durations)[: len(buf_list)].tolist()
            delays_kwds["duration"] = durations
        else:
            fps = self._canvas_kwds["animate_kwds"].get("fps")
            delays_kwds["fps"] = fps

        static = self._canvas_kwds["animate_kwds"].pop("static")
        stitch = self._canvas_kwds["animate_kwds"].pop("stitch")
        not_animated = static or not stitch
        fmt = self._canvas_kwds["animate_kwds"].get("format")
        if fmt is None:
            fmt = "png" if not_animated else "gif"

        animate_kwds = dict(format=fmt, subrectangles=True, **delays_kwds)
        animate_kwds = load_defaults(
            "animate_kwds",
            self._canvas_kwds["animate_kwds"],
            **animate_kwds,
        )

        loop = animate_kwds.pop("loop")
        if isinstance(loop, bool):
            loop = int(not loop)
        animate_kwds["loop"] = loop

        save = self._canvas_kwds["output_kwds"].pop("save", None)
        if save is not None:
            file, ext = os.path.splitext(save)
            fmt = animate_kwds["format"]
            dot_fmt = f".{fmt}"
            if ext == "":
                ext = dot_fmt
            elif ext.lower() != dot_fmt.lower():
                warnings.warn(
                    f"Extension in save file path {ext} differs from "
                    f"{dot_fmt} provided file format; using {ext}."
                )
            out_obj = f"{file}{ext}"
            if not os.path.isabs(out_obj):
                out_obj = os.path.join(os.getcwd(), out_obj)
        else:
            out_obj = BytesIO()
            fmt = animate_kwds["format"]
            ext = f".{fmt}"

        ext = ext.lower()
        pygifsicle = animate_kwds.pop("pygifsicle", None)
        show = self._canvas_kwds["output_kwds"].get("show")
        if save is None and pygifsicle and ext == ".gif" and stitch and show:
            # write temporary file since pygifsicle only accepts file paths
            for i in np.arange(0, 100):
                if os.path.exists(self._temp_file):
                    self._temp_file = f"{i:03d}{self._temp_file}"
                else:
                    break
            out_obj = self._temp_file

        if ext != ".gif":
            durations = animate_kwds.pop("duration", None)
            if "fps" not in animate_kwds and not not_animated:
                fps = 1 / np.min(durations)
                animate_kwds["fps"] = fps
                warnings.warn(
                    f"Only GIFs support setting explicit durations; "
                    f"defaulting fps to {fps} from 1 / min(durations)"
                )
            animate_kwds.pop("subrectangles", None)
            animate_kwds.pop("loop", None)
        animate_kwds["format"] = ext.lstrip(".")

        is_file = isinstance(out_obj, str)
        if static:
            if is_file:
                image = imageio.imread(buf_list[0])
                imageio.imwrite(out_obj, image)
            else:
                out_obj = buf_list[0]
        elif stitch:
            if ext == ".mp4" and save is None:
                raise NotImplementedError("Cannot output video as BytesIO; set save.")
            with imageio.get_writer(out_obj, **animate_kwds) as writer:
                for buf in buf_list:
                    image = imageio.imread(buf)
                    writer.append_data(image)
                    buf.close()
            if ext == ".gif" and pygifsicle and is_file:
                try:
                    from pygifsicle import optimize

                    optimize(out_obj)
                except ImportError:
                    warnings.warn(
                        "pip install pygifsicle to reduce size of output gif!"
                    )
        elif save is not None:
            file_dir = file
            if not os.path.isabs(file_dir):
                file_dir = os.path.join(os.getcwd(), file_dir)
            os.makedirs(file_dir, exist_ok=True)
            zfill = len(str(len(buf_list)))
            out_obj = []
            for state, buf in enumerate(buf_list, 1):
                path = os.path.join(file_dir, f"{state:0{zfill}d}{ext}")
                image = imageio.imread(buf)
                imageio.imwrite(path, image)
                out_obj.append(path)
        else:
            out_obj = buf_list
        return out_obj, ext

    @staticmethod
    def _show_output_file(out_obj, ext):
        from IPython import display

        if isinstance(out_obj, str):
            with open(out_obj, "rb") as fi:
                b64 = base64.b64encode(fi.read()).decode("ascii")
        else:
            b64 = base64.b64encode(out_obj.getvalue()).decode()

        if ext == ".gif":
            out_obj = display.HTML(f'<img src="data:image/gif;base64,{b64}" />')
        elif ext == ".mp4":
            out_obj = display.Video(b64, embed=True)
        elif ext in [".jpg", ".png"]:
            out_obj = display.HTML(
                f'<img src="data:image/{ext.lstrip(".")};base64,{b64}" />'
            )
        else:
            raise NotImplementedError(f"No method implemented to show {ext}!")

        return out_obj

    def _show_output(self, *args):
        show = self._canvas_kwds["output_kwds"].get("show")
        if show is None:
            try:
                get_ipython
                show = True
            except NameError:
                show = False

        if show:
            return self._show_output_file(*args)

    def render(self):
        data = self.finalize().data

        self._canvas_kwds = defaultdict(dict)
        for ds in data.values():
            for configurable in CONFIGURABLES["canvas"]:
                key = f"{configurable}_kwds"
                self._canvas_kwds[key] = ds.attrs.pop(key)
            break
        stitch = self._canvas_kwds["animate_kwds"]["stitch"]
        static = self._canvas_kwds["animate_kwds"]["static"]

        if self.debug:
            print(data)

        # unrecognized durations keyword if not popped
        if "duration" in ds.data_vars:
            durations = xr.concat(
                (pop(ds, "duration", to_numpy=False) for ds in data.values()),
                "item",
            )
        if self._canvas_kwds["animate_kwds"].get("fps") is not None:
            durations = None

        rows, cols = [max(rowcol) for rowcol in zip(*data.keys())]
        buf_list = self._create_frames(data, rows, cols)

        out_obj, ext = self._write_rendered(buf_list, durations)

        if self.show and (stitch or static):
            out_obj = self._show_output(out_obj, ext)

        if os.path.exists(self._temp_file):
            os.remove(self._temp_file)

        return out_obj
