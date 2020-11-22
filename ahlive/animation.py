import os
import base64
import warnings
from io import BytesIO
from copy import deepcopy
from collections.abc import Iterable

import param
import imageio
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import dask.delayed
import dask.diagnostics
from pygifsicle import optimize
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from .config import defaults, load_defaults
from .util import to_pydt, to_1d, to_num, to_scalar, is_datetime, is_scalar, pop, srange


OPTIONS = {
    'limit': ['fixed', 'follow']
}
GEO_FEATURES = [
    'borders',
    'coastline',
    'land',
    'lakes',
    'ocean',
    'rivers',
    'states',
]
FIGURE_KEYS = [
    'figure',
    'suptitle',
    'animate',
    'frame',
    'watermark',
    'spacing',
    'durations',
]


class Animation(param.Parameterized):

    path = param.Path(default='untitled.gif')
    figsize = param.NumericTuple(
        default=defaults['figure']['figsize'], length=2, precedence=100)
    suptitle = param.String()
    watermark = param.String(default='Animated using Ahlive')
    durations = param.ClassSelector(class_=(Iterable, int, float))
    fps = param.Number(default=None)
    export = param.Boolean(default=True)
    show = param.Boolean(default=None)
    animate = param.ClassSelector(
        default=True, class_=(Iterable, int, slice, bool))
    workers = param.Integer(default=2)
    debug = param.Boolean(default=None)

    _is_finalized = False
    _subset_states = None
    _animate = None
    _is_static = None
    _crs_names = {}
    _figure_kwds = {}
    _path_effects = [withStroke(linewidth=3, alpha=0.5, foreground='white')]

    def __init__(self, **kwds):
        super().__init__(**kwds)

    @staticmethod
    def _get_base_format(num):
        num = to_scalar(num)

        is_timedelta = isinstance(num, np.timedelta64)
        num = to_num(num)
        if isinstance(num, str):
            return 's'

        if is_timedelta:
            num = num / 1e9  # nanoseconds to seconds
            if num < 1:  # 1 second
                return '%S.%f'
            elif num < 60:  # 1 minute
                return '%M:%S'
            elif num < 3600:  # 1 hour
                return '%I:%M %p'
            elif num < 86400:  # 1 day
                return '%b %d %HZ'
            elif num < 604800:  # 7 days
                return '%b %d'
            elif num < 31536000:  # 1 year
                return '%b \'%y'
            else:
                return '%Y'

        if num == 0:
            return '.1f'

        order_of_magnitude = int(np.floor(np.log10(abs(num))))
        if order_of_magnitude >= 1:
            return '.1f'
        else:
            return f'.{abs(order_of_magnitude)}f'

    def _update_text(self, kwds, label_key, base=None, apply_format=True):
        label = kwds.get(label_key, None)

        if isinstance(label, Iterable) and not isinstance(label, str):
            labels = []
            for i, sub_label in enumerate(kwds[label_key]):
                sub_kwds = kwds.copy()
                sub_kwds[label_key] = sub_label
                format_ = sub_kwds['format']
                sub_kwds = self._update_text(
                    sub_kwds, label_key, base=base,
                    apply_format=apply_format)
                labels.append(sub_kwds[label_key])
            kwds[label_key] = labels
            kwds['format'] = format_
            kwds = {
                key: val for key, val in kwds.items()
                if key in sub_kwds.keys()}
            return kwds

        format_ = kwds.pop('format', 'auto')
        if base is not None and format_ == 'auto':
            format_ = self._get_base_format(base)

        if format_ != 'auto':
            if apply_format:
                try:
                    label = f'{label:{format_}}'
                except (ValueError, TypeError) as e:
                    warnings.warn(f'Could not apply {format_} on {label}')
                    pass
            else:
                kwds['format'] = format_

        prefix = kwds.pop('prefix', '')
        suffix = kwds.pop('suffix', '')
        if 'units' in kwds:
            units = f" [{kwds.pop('units')}]"
        else:
            units = ''
        label = f'{prefix}{label}{suffix}{units}'

        replacements = kwds.pop('replacements', {})
        for key, val in replacements.items():
            label = label.replace(key, val)

        casing = kwds.pop('casing', False)
        if casing:
            label = getattr(label, casing)()
        kwds[label_key] = label if label != 'None' else None
        return kwds

    def _prep_figure(self):
        figure_kwds = load_defaults('figure', self._figure_kwds['figure'])
        figure = plt.figure(**figure_kwds)

        if self.suptitle != '':
            suptitle_kwds = load_defaults(
                'suptitle', self._figure_kwds['suptitle'])
            suptitle_kwds = self._update_text(suptitle_kwds, 't')
            figure.suptitle(**suptitle_kwds)
        return figure

    def _prep_axes(self, state_ds, rows, cols, irowcol):
        limits = {
            var: pop(state_ds, var, get=-1)
            for var in list(state_ds.data_vars)
            if var[1:4] == 'lim'}

        axes_kwds = state_ds.attrs['axes']
        style = axes_kwds.pop('style', '')
        if style == 'minimal':
            for axis in ['x', 'y']:
                axis_min =  float(state_ds[axis].values.min())
                axis_max =  float(state_ds[axis].values.max())
                axis_lim = axes_kwds.get(f'{axis}lim', None)
                if axis_lim is not None:
                    axis_min = max(axis_min, axis_lim[0])
                    axis_max = min(axis_max, axis_lim[1])
                axes_kwds[f'{axis}ticks'] = to_pydt(
                    axis_min, axis_max)

                axis_label = axes_kwds.get(f'{axis}label', None)
                if axis_label is not None:
                    axes_kwds[f'{axis}label'] = f'Higher {axis_label} ➜'
        elif style == 'bare':
            axes_kwds['xticks'] = []
            axes_kwds['yticks'] = []

        axes_kwds['projection'] = pop(state_ds, 'projection', squeeze=True)
        axes_kwds = load_defaults('axes', state_ds, **axes_kwds)
        transform = axes_kwds.pop('transform', None)
        ax = plt.subplot(rows, cols, irowcol, **axes_kwds)

        if transform is not None:
            from cartopy import feature as cfeature
            if state_ds.attrs['settings']['worldwide']:
                ax.set_global()
            else:
                ax.set_extent([
                    limits.get('xlim0s', -180),
                    limits.get('xlim1s', 180),
                    limits.get('ylim0s', -90),
                    limits.get('ylim1s', 90)
                ], transform)
            for feature in GEO_FEATURES:
                feature_kwds = load_defaults(f'{feature}', state_ds)
                if feature_kwds.pop(feature, False):
                    feature_obj = getattr(cfeature, feature.upper())
                    ax.add_feature(feature_obj, **feature_kwds)
        else:
            for axis in ['x', 'y']:
                axis_lim0 = to_scalar(limits.get(f'{axis}lim0s'))
                axis_lim1 = to_scalar(limits.get(f'{axis}lim1s'))
                if axis_lim0 is not None or axis_lim1 is not None:
                    getattr(ax, f'set_{axis}lim')(
                        to_pydt(axis_lim0, axis_lim1))
        return ax

    def _update_labels(self, state_ds, ax):
        for label in ['xlabel', 'ylabel', 'title', 'subtitle']:
            label_kwds = load_defaults(f'{label}', state_ds)
            key = label if 'title' not in label else 'label'
            label_kwds = self._update_text(label_kwds, key)
            if label == 'subtitle':
                label = 'title'
            getattr(ax, f'set_{label}')(**label_kwds)

        if self.note:
            note_kwds = load_defaults(
                'note', state_ds, transform=ax.transAxes)
            ax.text(**note_kwds)

        if self.caption:
            y = -0.28 + -0.02 * self.caption.count('\n')
            caption_kwds = load_defaults(
                'caption', state_ds, transform=ax.transAxes, y=y)
            ax.text(**caption_kwds)

    def _update_margins(self, state_ds, ax):
        margins_kwds = load_defaults('margins', state_ds)
        ax.margins(**margins_kwds)

    def _add_state_labels(self, state_ds, ax):
        state_label = pop(state_ds, 'state_label', get=-1)
        if state_label is None:
            return
        state_label = to_pydt(state_label)
        state_base = state_ds.attrs['base'].get('state')

        state_kwds = load_defaults('state', state_ds, text=state_label)
        state_kwds = self._update_text(state_kwds, 'text', base=state_base)
        ax.annotate(**state_kwds)

    @staticmethod
    def _get_color(overlay_ds, plot):
        if isinstance(plot, list):
            plot = plot[0]

        if 'cmap' in overlay_ds.attrs['plot']:
            color = 'black'
        else:
            try:
                color = plot.get_color()
            except AttributeError as e:
                color = plot.get_facecolor()

            if isinstance(color, np.ndarray):
                color = color[0]
        return color

    def _plot_chart(self, overlay_ds, ax, chart, xs, ys, plot_kwds):
        if chart == 'scatter':
            # select last state
            plot = ax.scatter(xs, ys, **plot_kwds)
        elif chart == 'line':
            plot = ax.plot(xs, ys, **plot_kwds)
        elif chart.startswith('bar'):
            plot = getattr(ax, chart)(xs, ys, **plot_kwds)
        color = self._get_color(overlay_ds, plot)

        if xs.ndim == 2:  # a batch with same label
            for p in plot:
                p.set_color(color)
        return plot, color

    def _plot_trails(self, overlay_ds, ax, chart, color, xs, ys,
                     x_trails, y_trails, x_discrete_trails, y_discrete_trails,
                     trail_plot_kwds):
        all_none = (
            x_trails is None and y_trails is None and
            x_discrete_trails is None and y_discrete_trails is None)
        if all_none:
            return
        chart_kwds = load_defaults('chart', overlay_ds, base_chart=chart)
        chart = chart_kwds.pop('chart', 'both')
        expire = chart_kwds.pop('expire')
        stride = chart_kwds.pop('stride')
        line_chart_kwds = chart_kwds.copy()

        if chart in ['scatter', 'both']:
            non_nan_indices = np.where(~np.isnan(x_discrete_trails))
            x_discrete_trails = (
                x_discrete_trails[non_nan_indices][-expire - 1::stride])
            y_discrete_trails = (
                y_discrete_trails[non_nan_indices][-expire - 1::stride])
            chart_kwds.update(**trail_plot_kwds)
            chart_kwds = {
                key: val[non_nan_indices][-expire - 1::stride]
                if not is_scalar(val) else val
                for key, val in chart_kwds.items()}
            chart_kwds['label'] = '_nolegend_'
            plot = ax.scatter(
                x_discrete_trails, y_discrete_trails, **chart_kwds)

        if chart in ['line', 'both']:
            x_trails = x_trails[-expire * self._num_steps - 1:]
            y_trails = y_trails[-expire * self._num_steps - 1:]
            line_chart_kwds['label'] = '_nolegend_'
            plot = ax.plot(x_trails, y_trails, color=color, **line_chart_kwds)


    def _plot_deltas(self, overlay_ds, ax, chart,
                     x_centers, y_centers,
                     deltas, delta_labels, color):
        if deltas is None:
            return
        chart_kwds = load_defaults(
            'chart', overlay_ds, base_chart=chart)
        if chart == 'bar':
            chart_kwds['yerr'] = deltas
            x_inlines = x_centers
            y_inlines = y_centers + np.abs(deltas)
        else:
            y_centers, x_centers = x_centers, y_centers
            x_inlines = x_centers + np.abs(deltas)
            y_inlines = y_centers
            chart_kwds['xerr'] = deltas

        ax.errorbar(x_centers, y_centers, **chart_kwds)
        self._add_inline_labels(
            overlay_ds, ax, chart, x_inlines, y_inlines,
            delta_labels, color, base_key='delta')

    def _add_remarks(self, state_ds, ax, chart,
                     xs, ys, remarks, color):
        if remarks is None:
            return

        for x, y, remark in zip(xs, ys, remarks):
            if remark  == '':
                continue
            remark = to_num(remark)
            remark_inline_kwds = dict(
                text=remark, xy=(x, y),
                color=color, path_effects=self._path_effects
            )
            remark_inline_kwds = load_defaults(
                'remark_inline', state_ds,
                **remark_inline_kwds)
            remark_inline_kwds = self._update_text(
                remark_inline_kwds, 'text', base=remark)
            ax.annotate(**remark_inline_kwds)

            remark_kwds = load_defaults(
                'remark', state_ds, x=x, y=y, color=color)
            ax.scatter(**remark_kwds)

    def _plot_ref_chart(self, overlay_ds, ax):
        label = pop(overlay_ds, 'label', get=0) or '_nolegend_'
        chart = pop(overlay_ds, 'chart', get=0)

        x0s = pop(overlay_ds, 'x0')
        x1s = pop(overlay_ds, 'x1')
        y0s = pop(overlay_ds, 'y0')
        y1s = pop(overlay_ds, 'y1')
        inline_loc = pop(overlay_ds, 'inline_loc')
        inline_labels = pop(overlay_ds, 'inline_label', squeeze=True)

        plot_kwds = {
            var: pop(overlay_ds, var, get=0)
            for var in list(overlay_ds.data_vars)}
        plot_kwds = load_defaults(
            'ref_plot', overlay_ds,
            base_chart=chart, label=label, **plot_kwds)

        if chart == 'axvline':
            plot = ax.axvline(x0s[-1], **plot_kwds)
            inline_x = x0s
            inline_y = inline_loc
        elif chart == 'axhline':
            plot = ax.axhline(y0s[-1], **plot_kwds)
            inline_x = inline_loc
            inline_y = y0s
        elif chart == 'axhspan':
            plot = ax.axhspan(y0s[-1], y1s[-1], **plot_kwds)
            inline_x = inline_loc
            inline_y = [np.max([y0s, y1s])]
        elif chart == 'axvspan':
            plot = ax.axvspan(x0s[-1], x1s[-1], **plot_kwds)
            inline_x = [np.max([y0s, y1s])]
            inline_y = inline_loc

        if inline_labels is not None:
            color = self._get_color(overlay_ds, plot)
            self._add_inline_labels(
                overlay_ds, ax, chart, inline_x, inline_y,
                inline_labels, color, base_key='ref_inline',
                inline_key='ref_inline')

    def _add_inline_labels(self, overlay_ds, ax, chart,
                           xs, ys, inline_labels, color,
                           base_key='inline', inline_key='inline'):
        if inline_labels is None:
            return
        inline_base = overlay_ds.attrs['base'].get(base_key)

        ha = 'center'
        va = 'center'
        xytext = (0, 5)
        if chart == 'barh':
            ha = 'left' if base_key != 'bar' else 'right'
            xytext = xytext[::-1]
            if base_key != 'delta':
                xs, ys = ys, xs
        elif chart == 'bar':
            va = 'bottom' if base_key != 'bar' else 'top'
        elif chart in ['line', 'scatter']:
            ha = 'left'
            va = 'bottom'
        elif chart in ['axhline', 'axvline']:
            ha = 'left'
            va = 'bottom'

        inline_labels = to_scalar(inline_labels)
        xs = to_scalar(xs)
        ys = to_scalar(ys)
        if str(inline_labels) == 'nan':
            inline_labels = '?'
        elif str(inline_labels) == '':
            return
        inline_kwds = dict(
            text=inline_labels, xy=(xs, ys), ha=ha, va=va,
            color=color, xytext=xytext, path_effects=self._path_effects)
        inline_kwds = load_defaults(
            inline_key, overlay_ds, **inline_kwds)
        inline_kwds = self._update_text(
            inline_kwds, 'text', base=inline_base)
        ax.annotate(**inline_kwds)

    def _update_grid(self, state_ds, ax):
        grid_kwds = load_defaults('grid', state_ds)
        grid = grid_kwds.pop('grid', False)
        if not grid:
            return
        if 'transform' in grid_kwds:
            axis = grid_kwds.pop('axis')
            if 'draw_labels' not in grid_kwds:
                grid_kwds['draw_labels'] = True
            gridlines = ax.gridlines(draw_labels=True)

            if 'PlateCarree' in str(grid_kwds['transform']):
                gridlines.xlines = False
                gridlines.ylines = False

            if axis == 'x':
                gridlines.top_labels = False
                gridlines.bottom_labels = False
            elif axis == 'y':
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

    def _update_ticks(self, state_ds, ax, chart, gridlines):
        if chart.startswith('bar'):
            state_ds = state_ds[:, -1].T
        tick_labels = pop(state_ds, 'tick_label')

        xticks_base = state_ds.attrs['base'].get('xticks')
        xticks_kwds = load_defaults(
            'xticks', state_ds, labels=tick_labels)
        xticks_kwds = self._update_text(
            xticks_kwds, 'labels', base=xticks_base,
            apply_format=False)
        xticks = xticks_kwds.pop('ticks')
        xformat = xticks_kwds.pop('format', 'g')
        xticks_labels = xticks_kwds.pop('labels')
        x_is_datetime = xticks_kwds.pop('is_datetime', False)

        yticks_base = state_ds.attrs['base'].get('yticks')
        yticks_kwds = load_defaults(
            'yticks', state_ds, labels=tick_labels)
        yticks_kwds = self._update_text(
            yticks_kwds, 'labels', base=yticks_base,
            apply_format=False)
        yticks = yticks_kwds.pop('ticks')
        yformat = yticks_kwds.pop('format', 'g')
        yticks_labels = yticks_kwds.pop('labels')
        y_is_datetime = yticks_kwds.pop('is_datetime', False)

        if gridlines is not None:  # geoaxes
            from cartopy.mpl.gridliner import (
                LatitudeFormatter, LongitudeFormatter)
            gridlines.yformatter = LatitudeFormatter()
            gridlines.xformatter = LongitudeFormatter()
            for key in ['axis', 'which', 'length', 'labelsize']:
                if key == 'labelsize':
                    xticks_kwds['size'] = xticks_kwds.pop(
                        key, defaults['ticks']['labelsize'])
                    yticks_kwds['size'] = yticks_kwds.pop(
                        key, defaults['ticks']['labelsize'])
                else:
                    xticks_kwds.pop(key, '')
                    yticks_kwds.pop(key, '')
            gridlines.ylabel_style = yticks_kwds
            gridlines.xlabel_style = xticks_kwds

            if xticks is not None:
                gridlines.xlocator = FixedLocator(xticks)
            if yticks is not None:
                gridlines.ylocator = FixedLocator(yticks)
        else:
            if chart.startswith('bar'):
                xs = pop(state_ds, 'x')
                if chart == 'bar':
                    ax.set_xticks(xs)
                elif chart == 'barh':
                    ax.set_yticks(xs)
                ax.set_xtickslabels(xticks_labels)
                ax.set_ytickslabels(yticks_labels)
            else:
                if not x_is_datetime:
                    xformatter = FormatStrFormatter(f'%{xformat}')
                    ax.xaxis.set_major_formatter(xformatter)
                else:
                    xlocator = AutoDateLocator(minticks=5, maxticks=10)
                    xformatter = ConciseDateFormatter(xlocator)
                    ax.xaxis.set_major_locator(xlocator)
                    ax.xaxis.set_major_formatter(xformatter)

                if not y_is_datetime:
                    yformatter = FormatStrFormatter(f'%{yformat}')
                    ax.yaxis.set_major_formatter(yformatter)
                else:
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
        ncol = int(len(legend_labels) / 5) or 1
        legend_kwds = dict(
            handles=legend_items.values(), labels=legend_items.keys(),
            ncol=ncol)
        legend_kwds = load_defaults(
            'legend', state_ds, **legend_kwds)

        if not legend_labels or not legend_kwds.pop('show'):
            return

        legend = ax.legend(**{
            key: val for key, val in legend_kwds.items()
            if key not in ['replacements', 'casing', 'format']
        })

        if 's' in state_ds:
            s = legend_kwds.get('s', np.nanmean(state_ds['s']))
            for legend_handle in legend.legendHandles:
                legend_handle.set_sizes([s])

        for legend_label in legend.get_texts():
            legend_label.set_path_effects(self._path_effects)
        legend.get_frame().set_linewidth(0)

    def _update_colorbar(self, state_ds, ax, plot):
        if plot is None:
            return

        colorbar_kwds = load_defaults('colorbar', state_ds, ax=ax)
        if not colorbar_kwds.pop('show'):
            return

        divider = make_axes_locatable(ax)
        if colorbar_kwds['orientation'] == 'vertical':
            cax = divider.new_horizontal(
                size='2%', pad=0.1, axes_class=plt.Axes)
        else:
            cax = divider.new_vertical(
                size='2%', pad=0.1, axes_class=plt.Axes)
        ax.figure.add_axes(cax)

        colorbar = plt.colorbar(plot, cax=cax, **colorbar_kwds)
        clabel_kwds = load_defaults('clabel', state_ds)
        clabel_kwds = self._update_text(clabel_kwds, 'label')
        if colorbar_kwds['orientation'] == 'vertical':
            clabel_kwds['ylabel'] = clabel_kwds.pop('label')
            cticks_kwds = {'axis': 'y'}
            cax.set_ylabel(**clabel_kwds)
        else:
            clabel_kwds['xlabel'] = clabel_kwds.pop('label')
            cticks_kwds = {'axis': 'x'}
            cax.set_xlabel(**clabel_kwds)

        cticks_base = state_ds.attrs['base']['cticks']
        cticks_kwds = load_defaults('cticks', state_ds, **cticks_kwds)
        cticks_kwds.pop('num_ticks', None)
        cticks_kwds = self._update_text(
            cticks_kwds, 'ticks', base=cticks_base,
            apply_format=False)
        cformat = cticks_kwds.pop('format')
        cformatter = FormatStrFormatter(f'%{cformat}')

        if colorbar_kwds['orientation'] == 'vertical':
            cax.yaxis.set_major_formatter(cformatter)
        else:
            cax.xaxis.set_major_formatter(cformatter)
        cticks = cticks_kwds.pop('ticks')
        if cticks is not None:
            colorbar.set_ticks(np.array(cticks).astype(float))
            colorbar.set_ticklabels(cticks)
        cax.tick_params(**cticks_kwds)

    def _update_watermark(self, figure):
        if self.watermark:
            watermark_kwds = load_defaults(
                'watermark', self._figure_kwds['watermark'])
            figure.text(**watermark_kwds)

    def _update_spacing(self):
        if self.caption:
            bottom = 0.2 + 0.008 * self.caption.count('\n')
        else:
            bottom = defaults['spacing']['bottom']
        spacing_kwds = load_defaults(
            'spacing', self._figure_kwds['spacing'], bottom=bottom)
        plt.subplots_adjust(**spacing_kwds)

    def _apply_hooks(self, state_ds, figure, ax):  # TODO: implement
        hooks = state_ds.attrs['settings'].pop('hooks', [])
        for hook in hooks:
            if not callable(hook):
                continue
            hook(figure, ax)

    def _buffer_frame(self, state):
        buf = BytesIO()
        frame_kwds = load_defaults('frame', self._figure_kwds['frame'])
        try:
            plt.savefig(buf, **frame_kwds)
            buf.seek(0)
            plt.close()
            return buf
        except Exception as e:
            error_msg = f'Failed to render state={state} due to {e}!'
            if self.debug:
                raise RuntimeError(error_msg)
            else:
                print(error_msg)
            return

    @staticmethod
    def _reshape_batch(array, chart, get=-1):
        if array is None:
            return array

        if get is not None and chart != 'line':
            if array.ndim == 2:
                array = array[:, get]
            else:
                array = array[[get]]

        return array.T

    @staticmethod
    def _groupby_key(data, key):
        if key is not None and key != '':
            if isinstance(data, xr.Dataset):
                if key not in data.dims and key not in data.data_vars:
                    data = data.expand_dims(key)
                elif key in data.data_vars and is_scalar(data[key]):
                    data[key] = data[key].expand_dims('group')
            return data.groupby(key)  # TODO: explore unsorted
        else:
            return zip([''], [data])

    def _get_iter_ds(self, state_ds):
        if len(state_ds.data_vars) == 0:
            return zip([], [])
        elif any(group for group in to_1d(state_ds['group'])):
            state_ds = state_ds.drop('label').rename({'group': 'label'})
            key = 'label'
        else:
            state_ds = state_ds.drop('group', errors='ignore')
            key = 'item'
        iter_ds = self._groupby_key(state_ds, key)
        return iter_ds

    def _draw_subplot(self, state_ds, ax):
        self._update_labels(state_ds, ax)
        self._update_margins(state_ds, ax)
        self._add_state_labels(state_ds, ax)

        chart = ''
        plot = None
        base_vars = [
            var for var in state_ds.data_vars if not var.startswith('ref_')]
        base_state_ds = state_ds[base_vars]
        iter_ds = self._get_iter_ds(base_state_ds)
        for _, overlay_ds in iter_ds:
            overlay_ds = overlay_ds.where(overlay_ds['chart'] != '', drop=True)
            if len(to_1d(overlay_ds['state'])) == 0:
                continue
            chart = pop(overlay_ds, 'chart', get=0)
            if pd.isnull(chart):
                chart = str(chart)
                continue
            xs = self._reshape_batch(
                pop(overlay_ds, 'x'), chart)
            ys = self._reshape_batch(
                pop(overlay_ds, 'y'), chart)
            x_trails = self._reshape_batch(
                pop(overlay_ds, 'x_trail'), chart, get=None)
            y_trails = self._reshape_batch(
                pop(overlay_ds, 'y_trail'), chart, get=None)
            x_discrete_trails = self._reshape_batch(
                pop(overlay_ds, 'x_discrete_trail'), chart, get=None)
            y_discrete_trails = self._reshape_batch(
                pop(overlay_ds, 'y_discrete_trail'), chart, get=None)
            x_centers = self._reshape_batch(
                pop(overlay_ds, 'x_center'), chart)
            y_centers = self._reshape_batch(
                pop(overlay_ds, 'y_center'), chart)
            deltas = self._reshape_batch(
                pop(overlay_ds, 'delta'), chart)
            delta_labels = self._reshape_batch(
                pop(overlay_ds, 'delta_label'), chart)
            bar_labels = self._reshape_batch(
                pop(overlay_ds, 'bar_label'), chart)
            inline_labels = self._reshape_batch(
                pop(overlay_ds, 'inline_label'), chart)
            remarks = self._reshape_batch(
                pop(overlay_ds, 'remark'), chart)

            trail_plot_kwds = {
                var: self._reshape_batch(pop(overlay_ds, var), chart, get=None)
                for var in list(overlay_ds.data_vars)}
            trail_plot_kwds = load_defaults(
                'plot', overlay_ds, **trail_plot_kwds)
            plot_kwds = {
                key: to_scalar(val) for key, val in trail_plot_kwds.items()}
            if 'label' in plot_kwds:
                plot_kwds['label'] = to_scalar(plot_kwds['label'])
            if 'zorder' not in plot_kwds:
                plot_kwds['zorder'] = 2
            plot, color = self._plot_chart(
                overlay_ds, ax, chart, xs, ys, plot_kwds)
            self._plot_trails(
                overlay_ds, ax, chart, color, xs, ys, x_trails, y_trails,
                x_discrete_trails, y_discrete_trails, trail_plot_kwds)
            self._plot_deltas(
                overlay_ds, ax, chart, x_centers, y_centers,
                deltas, delta_labels, color)
            self._add_inline_labels(
                overlay_ds, ax, chart, xs, ys, bar_labels, 'black',
                base_key='bar')
            self._add_inline_labels(
                overlay_ds, ax, chart, xs, ys, inline_labels, color)
            self._add_remarks(
                overlay_ds, ax, chart, xs, ys, remarks, color)

        ref_state_ds = state_ds.drop(base_vars + ['item'], errors='ignore')
        ref_state_ds = ref_state_ds.rename({
            var: var.replace('ref_', '')
            for var in list(ref_state_ds) + ['ref_item']
            if var in state_ds
        })
        ref_iter_ds = self._get_iter_ds(ref_state_ds)
        for _, ref_overlay_ds in ref_iter_ds:
            self._plot_ref_chart(ref_overlay_ds, ax)

        gridlines = self._update_grid(state_ds, ax)
        self._update_ticks(state_ds, ax, chart, gridlines)
        self._update_legend(state_ds, ax)
        self._update_colorbar(state_ds, ax, plot)

    @dask.delayed()
    def _draw_frame(self, state_ds_rowcols, rows, cols):
        figure = self._prep_figure()
        for irowcol, state_ds in enumerate(state_ds_rowcols, 1):
            ax = self._prep_axes(state_ds, rows, cols, irowcol)
            self._draw_subplot(state_ds, ax)
            self._apply_hooks(state_ds, figure, ax)
        self._update_watermark(figure)
        self._update_spacing()
        state = pop(state_ds, 'state', get=-1)
        buf = self._buffer_frame(state)
        return buf

    def _config_chart(self, ds, chart):
        if chart.startswith('bar'):
            chart_type = ds.attrs['chart'].pop('chart_type', 'race')
            bar_label = ds.attrs['chart'].pop('bar_label', True)
            ds.coords['tick_label'] = ds['x']
            if bar_label:
                ds['bar_label'] = ds['x']
            if chart_type == 'race':
                ds['x'] = ds['y'].rank('item')
            else:
                ds['x'] = ds['x'].rank('item')
                if chart_type == 'delta':
                    x_delta = ds['x'].diff('item').mean() / 2
                    ds['x_center'] = ds['x'] - x_delta
                    ds['delta_label'] = ds['y'].diff('item')
                    ds['y_center'] = (
                        ds['y'].shift(item=1) + ds['delta_label'] / 2)
                    ds['delta_label'] = ds['delta_label'].isel(
                        item=slice(1, None))
                    ds['delta'] = ds['delta_label'] / 2
        elif chart == 'scatter':
            chart_type = ds.attrs['chart'].pop('chart_type', 'basic')
            if chart_type == 'trail':
                trail_chart = ds.attrs['chart'].get('chart', 'scatter')
                if trail_chart in ['line', 'both']:
                    ds['x_trail'] = ds['x'].copy()
                    ds['y_trail'] = ds['y'].copy()

                if trail_chart in ['scatter', 'both']:
                    ds['x_discrete_trail'] = ds['x'].copy()
                    ds['y_discrete_trail'] = ds['y'].copy()

        legend_sortby = ds.attrs['legend'].pop('sortby', None)
        if legend_sortby and 'label' in ds:
            items = ds.mean('state').sortby(
                legend_sortby, ascending=False)['item']
            ds = ds.sel(item=items)
            ds['item'] = srange(ds['item'])

        if self.style == 'bare':
            ds.attrs['grid']['b'] = False
        elif chart == 'barh':
            ds.attrs['grid']['axis'] = (
                ds.attrs['grid'].get('axis', 'x'))
        elif chart == 'bar':
            ds.attrs['grid']['axis'] = (
                ds.attrs['grid'].get('axis', 'y'))
        else:
            ds.attrs['grid']['axis'] = (
                ds.attrs['grid'].get('axis', 'both'))

        return ds

    def _add_xy01_limits(self, ds, chart):
        limits = {
            key: val
            for key, val in ds.attrs['settings'].items()
            if key[1:4] == 'lim'}

        for axis in ['x', 'y']:
            axis_lim = limits.pop(f'{axis}lims')
            if axis_lim is None:
                continue

            axis_lim0 = f'{axis}lim0s'
            axis_lim1 = f'{axis}lim1s'

            has_axis_lim0 = limits[axis_lim0] is not None
            has_axis_lim1 = limits[axis_lim1] is not None
            if has_axis_lim0 or has_axis_lim1:
                warnings.warn(
                    'Overwriting `{axis_lim0}` and `{axis_lim1}` '
                    'with set `{axis_lim}` {axis_lim}!')
            if isinstance(axis_lim, str):
                limits[axis_lim0] = axis_lim
                limits[axis_lim1] = axis_lim
            else:
                limits[axis_lim0] = axis_lim[0]
                limits[axis_lim1] = axis_lim[1]

        if ds.attrs['settings']['worldwide'] is None:
            if any(limit is not None for limit in limits.values()):
                ds.attrs['settings']['worldwide'] = False
            else:
                ds.attrs['settings']['worldwide'] = True

        if ds.attrs['settings']['worldwide']:
            return ds

        axes_kwds = ds.attrs['axes']
        margins_kwds = ds.attrs['margins']

        for key, limit in limits.items():
            # example: xlim0s
            axis = key[0]  # x
            num = int(key[-2])  # 0
            is_lower_limit = num == 0

            axis_limit_key = f'{axis}lim'
            if axes_kwds is not None:
                in_axes_kwds = axis_limit_key in axes_kwds
            else:
                in_axes_kwds = False
            unset_limit = limit is None and not in_axes_kwds
            has_other_limit = f'{key[:-1]}{1 - num}' is not None
            is_scatter = chart == 'scatter'
            is_line_y = chart == 'line' and axis == 'y'
            is_bar_x = chart.startswith('bar') and axis == 'x'
            is_bar_y = chart.startswith('bar') and axis == 'y'
            is_fixed = any([
                is_scatter,
                is_line_y,
                is_bar_y,
                has_other_limit
            ])
            if unset_limit and is_bar_y and is_lower_limit:
                limit = 0
            elif unset_limit and is_bar_x:
                continue
            elif unset_limit and is_fixed:
                limit = 'fixed'
            elif isinstance(limit, str):
                if not any(limit.startswith(op) for op in OPTIONS['limit']):
                    raise ValueError(
                        f"Got {limit} for {key}; must be either "
                        f"from {OPTIONS['limit']} or numeric values!"
                    )


            input_ = limit
            if isinstance(limit, str):
                if '_' in limit:
                    limit, offset = limit.split('_')
                else:
                    offset = 0

                if axis in ds:
                    var = axis
                else:
                    ref_vars = ['ref_x0', 'ref_y0', 'ref_y0', 'ref_y1']
                    if is_lower_limit:
                        ref_vars = ref_vars[::-1]
                    for var in ref_vars:
                        if var in ds and axis in var:
                            break
                    else:
                        continue

                if limit == 'fixed':
                    stat = 'min' if is_lower_limit else 'max'
                    limit = getattr(ds[var], stat)().values
                elif limit == 'follow':
                    stat = 'max' if is_lower_limit else 'min'
                    limit = getattr(ds[var], stat)('item').values

                if not chart.startswith('bar'):
                    if is_lower_limit:
                        limit = limit - float(offset)
                        limit -= limit * margins_kwds.get(axis, 0)
                    else:
                        limit = limit + float(offset)
                        limit += limit * margins_kwds.get(axis, 0)

            if limit is not None:
                if chart == 'barh':
                    axis = 'x' if axis == 'y' else 'y'
                    key = axis + key[1:]
                if is_scalar(limit) == 1:
                    limit = [limit] * self.num_states
                ds[key] = ('state', limit)
        return ds

    def _add_durations(self, ds):
        if self.durations is None:
            durations = 0.5 if self.num_states < 10 else 1 / 60
        else:
            durations = self._figure_kwds['durations']['durations']

        if isinstance(durations, (int, float)):
            durations = np.repeat(durations, self.num_states)
        transition_frames = (
            defaults['durations']['transition_frames'])

        durations_kwds = load_defaults(
            'durations', self._figure_kwds['durations'],
            transition_frames=transition_frames)
        aggregate = durations_kwds.pop('aggregate')

        if np.isnan(durations[-1]):
            durations[-1] = 0
        durations[-1] += durations_kwds['final_frame']

        if 'duration' in ds:
            ds['duration'] = ('state', durations + ds['duration'].values)
        else:
            ds['duration'] = ('state', durations)
        ds['duration'].attrs['transition_frames'] = transition_frames
        ds['duration'].attrs['aggregate'] = aggregate
        return ds

    def _get_crs(self, crs_name, crs_kwds):
        import cartopy.crs as ccrs
        if len(self._crs_names) == 0:
            self._crs_names = {
                crs_name.lower(): crs_name for crs_name in dir(ccrs)
                if '_' not in crs_name}

        crs_name = self._crs_names.get(crs_name.lower(), 'PlateCarree')
        crs_obj = getattr(ccrs, crs_name)(**crs_kwds)
        return crs_obj

    def _add_geo_transforms(self, ds):
        crs_kwds = load_defaults('crs', ds)
        crs = crs_kwds.pop('crs', None)

        projection_kwds = load_defaults('projection', ds)
        projection = projection_kwds.pop('projection', None)

        if crs != '' or projection != '':
            crs_obj = self._get_crs(crs, crs_kwds)
            ds.attrs['plot']['transform'] = crs_obj
            ds.attrs['inline']['transform'] = crs_obj
            ds.attrs['grid']['transform'] = crs_obj
            ds.attrs['axes']['transform'] = crs_obj

            projection_obj = self._get_crs(projection, projection_kwds)
            ds['projection'] = projection_obj
        return ds

    @staticmethod
    def _fill_null(ds):
        for var in ds.data_vars:
            if ds[var].dtype == 'O':
                try:
                    ds[var] = ds[var].astype(float)
                except ValueError:
                    ds[var] = ds[var].where(~pd.isnull(ds[var]), '')
        return ds

    def _compress_vars(self, da):
        if isinstance(da, xr.Dataset):
            if da.get('item', 1) == 1 and da.get('ref_item') == 1:
                return da
            da = da.map(self._compress_vars, keep_attrs=True)
            return da

        unique_vals = np.unique(da.values)
        if len(unique_vals) == 1:
            return unique_vals[0]
        else:
            return da

    @staticmethod
    def _add_color_kwds(ds, chart):
        if chart.startswith('bar'):
            color = None
            if set(np.unique(ds['label'])) == set(np.unique(ds['x'])):
                ds.attrs['legend']['show'] = False

        if 'c' in ds:
            cticks = ds.attrs['cticks'].get('ticks')
            if cticks is None:
                num_ticks = defaults['cticks']['num_ticks']
            else:
                num_ticks = len(cticks)
            num_ticks = ds.attrs['cticks'].pop('num_ticks', num_ticks)
            if num_ticks < 3:
                raise ValueError('There must be at least 3 ticks for cticks!')
            vmin = ds.attrs['plot'].get('vmin')
            vmax = ds.attrs['plot'].get('vmax')

            ds.attrs['plot']['cmap'] = plt.get_cmap(
                ds.attrs['plot'].get('cmap', 'plasma'), num_ticks)

            if cticks is not None:
                ds.attrs['plot']['norm'] = ds.attrs['plot'].get(
                    'norm', BoundaryNorm(cticks, num_ticks))
            elif vmin is None and vmax is None:
                cticks = np.linspace(
                    np.nanmin(ds['c'].values),
                    np.nanmax(ds['c'].values),
                    num_ticks)
                ds.attrs['plot']['norm'] = ds.attrs['plot'].get(
                    'norm', BoundaryNorm(cticks, num_ticks))
            ds.attrs['colorbar']['show'] = ds.attrs['plot'].get(
                'show', True)
        else:
            ds.attrs['colorbar']['show'] = False
        return ds

    @staticmethod
    def _add_base_kwds(ds):
        base_kwds = {}
        for xyc in ['x', 'y', 'c']:
            if xyc in ds:
                try:
                    base_kwds[f'{xyc}ticks'] = (
                        np.nanquantile(ds[xyc], 0.5) / 10)
                except TypeError:
                    base_kwds[f'{xyc}ticks'] = np.nanmin(ds[xyc])
                if xyc == 'c':
                    continue
                ds.attrs[f'{xyc}ticks']['is_datetime'] = is_datetime(ds[xyc])

        keys = ['inline', 'state', 'delta', 'bar', 'ref_inline']
        for key in keys:
            key_label = f'{key}_label'
            if key_label in ds:
                try:
                    if np.issubdtype(ds[key_label].values.dtype, np.datetime64):
                        base_kwds[key] = abs(np.diff(ds[key_label]).min() / 10)
                    else:
                        base_kwds[key] = np.nanmin(np.diff(ds[key_label]))
                except Exception:
                    pass

        ds.attrs['base'] = base_kwds
        return ds

    def _interp_dataset(self, ds):
        ds = ds.reset_coords()
        ds = ds.map(self.interpolate, keep_attrs=True)

        if 'x' in ds.data_vars and 'item' not in ds.dims:
            ds = ds.expand_dims('item')
            ds['item'] = srange(ds['item'])

        if 'ref_chart' in ds.data_vars and 'ref_item' not in ds.dims:
            ds = ds.expand_dims('ref_item')
            ds['ref_item'] = srange(ds['ref_item'])

        ds['state'] = srange(len(ds['state']))
        return ds

    def finalize(self):
        if self._is_finalized:
            return self

        if isinstance(self.animate, slice):
            start = self.animate.start
            stop = self.animate.stop
            step = self.animate.step or 1
            self._subset_states = range(start, stop, step)
            self._animate = True
            self._is_static = is_scalar(self._subset_states)
        elif isinstance(self.animate, bool):
            self._subset_states = None
            self._animate = self.animate
            self._is_static = False
        elif isinstance(self.animate, (Iterable, int)):
            self._subset_states = to_1d(self.animate, flat=False)
            self._animate = self.animate
            if self._subset_states[0] == 0:
                warnings.warn(
                    'State 0 detected in `animate`; shifting by 1.')
                self._subset_states += 1
            self._is_static = True if isinstance(self.animate, int) else False

        self_copy = deepcopy(self)
        if not self_copy._is_configured:
            self_copy = self_copy.config()

        data = {}
        for i, (rowcol, ds) in enumerate(self_copy.data.items()):
            if i == 0:
                for key in FIGURE_KEYS:
                    self_copy._figure_kwds[key] = ds.attrs[key]

            chart = to_scalar(ds['chart']) if 'chart' in ds else ''
            ds = self._fill_null(ds)
            ds = self._compress_vars(ds)
            if chart != '':
                ds = self._add_color_kwds(ds, chart)
            ds = self._config_chart(ds, chart)
            ds = self._add_xy01_limits(ds, chart)
            ds = self._add_base_kwds(ds)
            if self.fps is None:
                ds = self._add_durations(ds)
            ds = self._interp_dataset(ds)
            ds = self._add_geo_transforms(ds)
            ds.attrs['finalized'] = True
            self_copy._is_finalized = True
            data[rowcol] = ds
        self_copy.data = data
        return self_copy

    def _create_frames(self, data, rows, cols):
        jobs = []
        if self._subset_states is not None:
            states = self._subset_states
        else:
            states = srange(self.num_states)

        for state in states:
            state_ds_rowcols = [
                ds.sel(state=slice(None, state))
                if 'line' in ds.get('chart', ds.get('ref_chart'))
                or 'x_trail' in ds.data_vars
                or 'x_discrete_trail' in ds.data_vars
                else ds.sel(state=state)
                for ds in data.values()
            ]
            job = self._draw_frame(state_ds_rowcols, rows, cols)
            jobs.append(job)

        if self.num_states >= self.workers:
            num_workers = self.workers
        else:
            num_workers = self.num_states

        scheduler = 'single-threaded' if self.debug else 'processes'
        with dask.diagnostics.ProgressBar(minimum=0.5):
            buf_list = [
                buf for buf in dask.compute(
                    jobs, num_workers=num_workers,
                    scheduler=scheduler)[0]
                if buf is not None]
        return buf_list

    def _decide_speed(self, ext, buf_list, durations, animate_kwds):
        if self.fps is not None:
            animate_kwds['fps'] = self.fps
            return animate_kwds

        durations = getattr(
            durations, durations.attrs['aggregate']
        )('item', keep_attrs=True)
        durations = durations.where(
            durations > 0, durations.attrs['transition_frames']).squeeze()
        durations = to_1d(durations)[:len(buf_list)]

        if ext != '.gif':
            fps = 1 / durations.min()
            warnings.warn(
                f'Only GIFs support setting explicit durations; '
                f'defaulting `fps` to {fps} from 1 / min(durations)')
            animate_kwds['fps'] = fps
        else:
            # imageio requires native list
            animate_kwds['duration'] = durations.tolist()
        return animate_kwds

    def _export_rendered(self, buf_list, durations):
        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        file, ext = os.path.splitext(self.path)
        if self._is_static or not self._animate:
            ext = '.png'
        elif ext == '':
            ext = '.' + defaults["animate"]["format"]
            warnings.warn(
                f'{path} has no extension; defaulting to {ext}')
        else:
            ext = ext.lower()

        path = f'{file}{ext}'
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        if self._is_static:
            image = imageio.imread(buf_list[0])
            imageio.imwrite(path, image)
        elif self._animate:
            animate_kwds = dict(loop=loop, format=ext)
            animate_kwds = self._decide_speed(
                ext, buf_list, durations, animate_kwds)
            if ext == '.gif':
                animate_kwds['subrectangles'] = True

            animate_kwds = load_defaults(
                'animate', self._figure_kwds['animate'], **animate_kwds)
            with imageio.get_writer(path, **animate_kwds) as writer:
                for buf in buf_list:
                    image = imageio.imread(buf)
                    writer.append_data(image)
                    buf.close()
        else:
            file_dir = file
            if not os.path.isabs(file_dir):
                file_dir = os.path.join(os.getcwd(), file_dir)
            os.makedirs(file_dir, exist_ok=True)
            zfill = len(str(len(buf_list)))
            for state, buf in enumerate(buf_list, 1):
                path = os.path.join(file_dir, f'{state:0{zfill}d}{ext}')
                image = imageio.imread(buf)
                imageio.imwrite(path, image)
        return path, ext

    @staticmethod
    def _show_output_file(path, ext):
        from IPython import display
        with open(path, 'rb') as fi:
            b64 = base64.b64encode(fi.read()).decode('ascii')
        if ext == '.gif':
            return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')
        elif ext == '.mp4':
            return display.Video(path)
        elif ext in ['.jpg', '.png']:
            return display.Image(path)
        else:
            raise NotImplementedError(f'No method implemented to show {ext}!')

    @staticmethod
    def _show_output_buf(buf):
        from IPython import display
        image = imageio.imread(buf)
        return display.Image(image)

    def _show_output(self, *args):
        if self.show is None:
            try:
                get_ipython
                show = True
            except NameError as e:
                show = False
        else:
            show = self.show

        if show and not self._animate:
            warnings.warn('Unable to show unmerged output!')
            return args[0]
        elif not show or not self.export:
            return args[0]

        if self._is_static:
            args[0] = args[0][0]

        try:
            if isinstance(args[0], str):
                return self._show_output_file(*args)
            else:
                return self._show_output_buf(*args)
        except Exception as e:
            warnings.warn(
                f'Unable to show output in notebook due to {e}!')
            return args[0]

    def render(self, path=None, export=None, show=None,
               animate=None, workers=None):
        for key, val in locals().items():
            if key in ['self']:
                continue
            elif val is not None:
                setattr(self, key, val)

        # rather than directly setting data = self.data, this
        # way triggers computing num_states automatically
        self.data = self.finalize().data
        data = self.data
        print(data)

        # pop before creating frames
        if self.fps is None:
            durations = xr.concat((
                pop(ds, 'duration', to_numpy=False) for ds in data.values()
            ), 'item')
        else:
            durations = None

        rows, cols = [max(rowcol) for rowcol in zip(*data.keys())]
        buf_list = self._create_frames(data, rows, cols)

        if self.export:
            path, ext = self._export_rendered(buf_list, durations)
            output = self._show_output(path, ext)
        else:
            output = self._show_output(buf_list)
        return output
