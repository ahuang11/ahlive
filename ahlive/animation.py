import os
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
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from .config import defaults, load_defaults
from .util import to_pydt, to_num, is_scalar, pop, srange


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


class Animation(param.Parameterized):

    save_path = param.String(default='untitled.gif')
    figsize = param.NumericTuple(
        default=defaults['fig_kwds']['figsize'], length=2)
    suptitle = param.String()
    watermark = param.String(default='Animated using Ahlive')
    delays = param.ClassSelector(class_=(Iterable, int, float))

    fig_kwds = param.Dict()
    animate_kwds = param.Dict()
    frame_kwds = param.Dict()
    num_workers = param.Integer(default=4, bounds=(1, None))
    return_out = param.Boolean(default=True)
    suptitle_kwds = param.Dict()
    watermark_kwds = param.Dict()
    delays_kwds = param.Dict()
    debug = param.Integer(default=None)

    _num_states = 0
    _path_effects = [withStroke(linewidth=3, alpha=0.5, foreground='white')]
    _crs_names = {}

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.debug = self.debug or 0

    @staticmethod
    def _get_base_format(num):
        if isinstance(num, xr.DataArray):
            num = num.values.item()

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
                sub_kwds = self._update_text(
                    sub_kwds, label_key, base=base,
                    apply_format=apply_format)
                format_ = sub_kwds['format']
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
                except ValueError:
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

        casing = kwds.pop('casing', 'title')
        if casing:
            label = getattr(label, casing)()
        kwds[label_key] = label
        return kwds

    def _prep_figure(self):
        fig_kwds = load_defaults(
            'fig_kwds', self.fig_kwds, figsize=self.figsize)
        fig = plt.figure(**fig_kwds)

        if self.suptitle != '':
            suptitle_kwds = load_defaults(
                'suptitle_kwds', self.suptitle_kwds, t=self.suptitle)
            suptitle_kwds = self._update_text(suptitle_kwds, 't')
            fig.suptitle(**suptitle_kwds)
        return fig

    def _prep_axes(self, state_ds, rows, cols, irowcol):
        limits = {
            var: pop(state_ds, var)[-1]
            for var in list(state_ds.data_vars)
            if var[1:4] == 'lim'}

        axes_kwds = state_ds.attrs['axes_kwds']
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
        axes_kwds = load_defaults('axes_kwds', state_ds, **axes_kwds)
        transform = axes_kwds.pop('transform', None)
        ax = plt.subplot(rows, cols, irowcol, **axes_kwds)

        if transform is not None:
            from cartopy import feature as cfeature
            if state_ds.attrs['finalize_kwds']['worldwide']:
                ax.set_global()
            else:
                ax.set_extent([
                    limits.get('xlim0', -180),
                    limits.get('xlim1', 180),
                    limits.get('ylim0', -90),
                    limits.get('ylim1', 90)
                ], transform)
            for feature in GEO_FEATURES:
                feature_kwds = load_defaults(f'{feature}_kwds', state_ds)
                if feature_kwds.pop(feature, False):
                    feature_obj = getattr(cfeature, feature.upper())
                    ax.add_feature(feature_obj, **feature_kwds)
        else:
            if 'xlim0' in limits or 'xlim1' in limits:
                axes_kwds['xlim'] = to_pydt(
                    limits.get('xlim0'), limits.get('xlim1'))
            if 'ylim0' in limits or 'ylim1' in limits:
                axes_kwds['ylim'] = to_pydt(
                    limits.get('ylim0'), limits.get('ylim1'))
        return ax

    def _update_labels(self, state_ds, ax):
        for label in ['xlabel', 'ylabel', 'title']:
            label_kwds = load_defaults(f'{label}_kwds', state_ds)
            key = label if label != 'title' else 'label'
            label_kwds = self._update_text(label_kwds, key)
            getattr(ax, f'set_{label}')(**label_kwds)

    def _update_margins(self, state_ds, ax):
        margins_kwds = load_defaults('margins_kwds', state_ds)
        ax.margins(**margins_kwds)

    def _add_state_labels(self, state_ds, ax):
        state_label = pop(state_ds, 'state_label', get=0)
        if state_label is None:
            return
        state_label = to_pydt(state_label)
        state_base = state_ds.attrs['base_kwds'].get('state')

        state_kwds = load_defaults('state_kwds', state_ds, text=state_label)
        state_kwds = self._update_text(state_kwds, 'text', base=state_base)
        ax.annotate(**state_kwds)


    def _plot_ref_chart(self, overlay_ds, ax):
        label = pop(overlay_ds, 'label', get=0) or '_nolegend_'
        chart = pop(overlay_ds, 'chart', get=0)

        x0s = pop(overlay_ds, 'x0')
        x1s = pop(overlay_ds, 'x1')
        y0s = pop(overlay_ds, 'y0')
        y1s = pop(overlay_ds, 'y1')
        inline_loc = pop(overlay_ds, 'inline_loc')
        inline_labels = pop(overlay_ds, 'inline_label')

        plot_kwds = {
            var: pop(overlay_ds, var, get=0)
            for var in list(overlay_ds.data_vars)}
        plot_kwds = load_defaults(
            'plot_kwds', overlay_ds,
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
            if inline_labels[-1] != '':
                self._add_inline_labels(
                    overlay_ds, ax, chart, inline_x, inline_y,
                    inline_labels, color, base_key='ref_inline',
                    inline_key='ref_inline_kwds')

    @staticmethod
    def _get_color(plot):
        if isinstance(plot, list):
            plot = plot[0]
        try:
            color = plot.get_color()
        except AttributeError as e:
            color = plot.get_facecolor()

        if isinstance(color, np.ndarray):
            color = color[0]
        return color

    def _plot_chart(self, overlay_ds, ax, chart, xs, ys, plot_kwds):
        s = plot_kwds.pop('s', None)
        if chart == 'scatter':
            # select last state
            plot = ax.scatter(xs, ys, s=s, **plot_kwds)
        elif chart == 'line':
            plot = ax.plot(xs, ys, **plot_kwds)
        elif chart.startswith('bar'):
            plot = getattr(ax, chart)(
                xs, ys, **plot_kwds)
        color = self._get_color(plot)
        return plot, color

    def _plot_trails(self, overlay_ds, ax, chart,
                     xs, ys, x_trails, y_trails, color, plot_kwds):
        if x_trails is None or y_trails is None:
            return
        plot_kwds['alpha'] = 0.5
        chart_kwds = load_defaults(
            'chart_kwds', overlay_ds, base_chart=chart, **plot_kwds)
        chart_kwds['label'] = '_nolegend_'
        chart = chart_kwds.pop('chart', 'scatter', squeeze=True)
        expire = chart_kwds.pop('expire', squeeze=True)
        stride = chart_kwds.pop('stride', squeeze=True)
        indices = np.where(~np.isnan(x_trails))
        x_trails = x_trails[indices][-expire - 1::stride]
        y_trails = y_trails[indices][-expire - 1::stride]
        if chart == 'line':
            x_trails = np.concatenate([x_trails, xs])
            y_trails = np.concatenate([y_trails, ys])
            chart_kwds.pop('s')
            chart = 'plot'
        plot = getattr(ax, chart)(x_trails, y_trails, **chart_kwds)

    def _plot_deltas(self, overlay_ds, ax, chart,
                     x_centers, y_centers,
                     deltas, delta_labels, color):
        if deltas is None:
            return
        chart_kwds = load_defaults(
            'chart_kwds', overlay_ds, base_chart=chart)
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

    def _add_inline_labels(self, overlay_ds, ax, chart,
                           xs, ys, inline_labels, color,
                           base_key='inline', inline_key='inline_kwds'):
        if inline_labels is None:
            return
        inline_base = overlay_ds.attrs['base_kwds'].get(base_key)

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

        inline_kwds = dict(
            text=inline_labels, xy=(xs, ys), ha=ha, va=va,
            color=color, xytext=xytext, path_effects=self._path_effects)
        inline_kwds = load_defaults(
            inline_key, overlay_ds, **inline_kwds)
        inline_kwds = self._update_text(
            inline_kwds, 'text', base=inline_base)
        ax.annotate(**inline_kwds)

    def _add_annotations(self, state_ds, ax, chart,
                         xs, ys, annotations, color):
        if annotations is None:
            return
        for x, y, annotation in zip(xs, ys, annotations):
            if annotation  == '':
                continue
            annotation = to_num(annotation)
            annotation_kwds = dict(
                text=annotation, xy=(x, y),
                color=color, path_effects=self._path_effects
            )
            annotation_kwds = load_defaults(
                'annotation_kwds', self.annotation_kwds, **annotation_kwds)
            annotation_kwds = self._update_text(
                annotation_kwds, 'text', base=annotation)
            ax.annotate(**annotation_kwds)
            ax.scatter(x, y, color=color)

    def _update_grid(self, state_ds, ax):
        grid_kwds = load_defaults('grid_kwds', state_ds)
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

        xtick_base = state_ds.attrs['base_kwds'].get('xtick')
        xtick_kwds = load_defaults(
            'xtick_kwds', state_ds, labels=tick_labels)
        xtick_kwds = self._update_text(
            xtick_kwds, 'labels', base=xtick_base,
            apply_format=False)
        xformat = xtick_kwds.pop('format', 'g')
        xtick_labels = xtick_kwds.pop('labels')
        x_is_datetime = xtick_kwds.pop('is_datetime', False)

        ytick_base = state_ds.attrs['base_kwds'].get('ytick')
        ytick_kwds = load_defaults(
            'ytick_kwds', state_ds, labels=tick_labels)
        ytick_kwds = self._update_text(
            ytick_kwds, 'labels', base=ytick_base,
            apply_format=False)
        yformat = ytick_kwds.pop('format', 'g')
        ytick_labels = ytick_kwds.pop('labels')
        y_is_datetime = ytick_kwds.pop('is_datetime', False)

        if gridlines is not None:
            from cartopy.mpl.gridliner import (
                LatitudeFormatter, LongitudeFormatter)
            gridlines.yformatter = LatitudeFormatter()
            gridlines.xformatter = LongitudeFormatter()
            for key in ['axis', 'which', 'length', 'labelsize']:
                if key == 'labelsize':
                    xtick_kwds['size'] = xtick_kwds.pop(
                        key, defaults['tick_kwds']['labelsize'])
                    ytick_kwds['size'] = ytick_kwds.pop(
                        key, defaults['tick_kwds']['labelsize'])
                else:
                    xtick_kwds.pop(key, '')
                    ytick_kwds.pop(key, '')
            gridlines.ylabel_style = ytick_kwds
            gridlines.xlabel_style = xtick_kwds
        else:
            if chart.startswith('bar'):
                xs = pop(state_ds, 'x')
                if chart == 'bar':
                    ax.set_xticks(xs)
                elif chart == 'barh':
                    ax.set_yticks(xs)
                ax.set_xticklabels(xtick_labels)
                ax.set_yticklabels(ytick_labels)
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
            ax.tick_params(**xtick_kwds)
            ax.tick_params(**ytick_kwds)

    def _update_legend(self, state_ds, ax):
        legend_labels = ax.get_legend_handles_labels()[1]
        ncol = int(len(legend_labels) / 5)
        if ncol == 0:
            ncol += 1
        legend_kwds = dict(labels=legend_labels, ncol=ncol)
        legend_kwds = load_defaults(
            'legend_kwds', state_ds, **legend_kwds)

        if not legend_labels or not legend_kwds.pop('show'):
            return

        legend = ax.legend(**{
            key: val for key, val in legend_kwds.items()
            if key not in ['replacements', 'casing', 'format']
        })

        if 's' in state_ds:
            s = legend_kwds.get('s', state_ds['s'].values.flat[0])
            for legend_handle in legend.legendHandles:
                legend_handle.set_sizes([s])

        for legend_label in legend.get_texts():
            legend_label.set_path_effects(self._path_effects)
        legend.get_frame().set_linewidth(0)

    def _update_colorbar(self, state_ds, ax, plot):
        if plot is None:
            return

        colorbar_kwds = load_defaults('colorbar_kwds', state_ds, ax=ax)
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

        colorbar_kwds.pop('num_colors', None)
        colorbar = plt.colorbar(plot, cax=cax, **colorbar_kwds)
        clabel_kwds = load_defaults(
            'clabel_kwds', self.clabel_kwds, label=self.clabel)
        clabel_kwds = self._update_text(clabel_kwds, 'label')
        if colorbar_kwds['orientation'] == 'vertical':
            clabel_kwds['ylabel'] = clabel_kwds.pop('label')
            ctick_kwds = {'axis': 'y'}
            cax.set_ylabel(**clabel_kwds)
        else:
            clabel_kwds['xlabel'] = clabel_kwds.pop('label')
            ctick_kwds = {'axis': 'x'}
            cax.set_xlabel(**clabel_kwds)

        cticks = colorbar.get_ticks()
        ctick_base = state_ds.attrs['base_kwds']['ctick']
        ctick_kwds = load_defaults('ctick_kwds', self.ctick_kwds,
            ticks=cticks, **ctick_kwds)
        ctick_kwds = self._update_text(
            ctick_kwds, 'ticks', base=ctick_base,
            apply_format=False)
        cformat = ctick_kwds.pop('format')
        cformatter = FormatStrFormatter(f'%{cformat}')

        cticks = np.array(ctick_kwds.pop('ticks')).astype(float)
        if colorbar_kwds['orientation'] == 'vertical':
            cax.yaxis.set_major_formatter(cformatter)
        else:
            cax.xaxis.set_major_formatter(cformatter)
        colorbar.set_ticks(cticks)
        cax.tick_params(**ctick_kwds)

    def _update_watermark(self, fig):
        watermark_kwds = load_defaults(
            'watermark_kwds', self.watermark_kwds, s=self.watermark)
        fig.text(.995, .005, **watermark_kwds)

    def _apply_hooks(self, fig, ax):  # TODO: implement
        hooks = [self.hooks] if callable(self.hooks) else self.hooks
        for hook in self.hooks:
            if not callable(hook):
                continue
            hook(fig, ax)

    def _buffer_frame(self, state):
        buf = BytesIO()
        frame_kwds = load_defaults('frame_kwds', self.frame_kwds)
        try:
            if self.debug:
                debug_dir = 'AHLIVE_DEBUGGER'
                fmt = frame_kwds['format']
                debug_path = os.path.join(debug_dir, f'{state}.{fmt}')
                os.makedirs(debug_dir, exist_ok=True)
                plt.savefig(debug_path, **frame_kwds)
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
    def _reshape_batch(array, chart):
        if array is None:
            return array

        if chart != 'line':
            try:
                return array[:, -1].T
            except IndexError:
                return array.T
        else:
            return array.T

    @staticmethod
    def _groupby_key(data, key):
        if key is not None and key != '':
            if isinstance(data, xr.Dataset):
                if key not in data.dims:
                    data = data.expand_dims(key)
            return data.groupby(key)
        else:
            return zip([''], [data])

    def _get_iter_ds(self, state_ds):
        batch = pop(state_ds, 'batch', dflt=False, get=0)
        if len(state_ds.data_vars) == 0:
            iter_ds = zip([], [])
        elif batch:
            state_ds = state_ds.drop('label')
            state_ds = state_ds.rename({'group': 'label'})
            iter_ds = self._groupby_key(state_ds, 'label')
        else:
            state_ds = state_ds.drop('group', errors='ignore')
            iter_ds = self._groupby_key(state_ds, 'item')
        return iter_ds

    def _draw_subplot(self, state_ds, ax):
        self._update_labels(state_ds, ax)
        self._update_margins(state_ds, ax)
        self._add_state_labels(state_ds, ax)

        ref_vars = [
            var for var in state_ds.data_vars if var.startswith('REF_')]
        ref_state_ds = state_ds[ref_vars].rename({
            var: var.replace('REF_', '') for var in ref_vars + ['REF_item']})
        ref_iter_ds = self._get_iter_ds(ref_state_ds)
        for _, ref_overlay_ds in ref_iter_ds:
            self._plot_ref_chart(ref_overlay_ds, ax)

        chart = ''
        plot = None
        state_ds = state_ds.drop_vars(ref_vars)
        iter_ds = self._get_iter_ds(state_ds)
        for _, overlay_ds in iter_ds:
            chart = pop(overlay_ds, 'chart', dflt='', get=0)
            xs = self._reshape_batch(
                pop(overlay_ds, 'x'), chart)
            ys = self._reshape_batch(
                pop(overlay_ds, 'y'), chart)
            x_trails = self._reshape_batch(
                pop(overlay_ds, 'x_trail'), chart)
            y_trails = self._reshape_batch(
                pop(overlay_ds, 'y_trail'), chart)
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
            annotations = self._reshape_batch(
                pop(overlay_ds, 'annotation'), chart)
            plot_kwds = {
                var: self._reshape_batch(pop(overlay_ds, var), chart)
                for var in list(overlay_ds.data_vars)
            }
            plot_kwds = load_defaults('plot_kwds', overlay_ds, **plot_kwds)

            if 'label' in plot_kwds:
                plot_kwds['label'] = plot_kwds['label'].squeeze()
                if not isinstance(plot_kwds['label'], str):
                    plot_kwds.pop('label')

            plot, color = self._plot_chart(
                overlay_ds, ax, chart, xs, ys, plot_kwds)
            self._plot_trails(
                overlay_ds, ax, chart, xs, ys, x_trails, y_trails, color,
                plot_kwds)
            self._plot_deltas(
                overlay_ds, ax, chart, x_centers, y_centers,
                deltas, delta_labels, color)
            self._add_inline_labels(
                overlay_ds, ax, chart, xs, ys, bar_labels, 'black',
                base_key='bar')
            self._add_inline_labels(
                overlay_ds, ax, chart, xs, ys, inline_labels, color)
            self._add_annotations(
                overlay_ds, ax, chart, xs, ys, annotations, color)

        gridlines = self._update_grid(state_ds, ax)
        self._update_ticks(state_ds, ax, chart, gridlines)
        self._update_legend(state_ds, ax)
        self._update_colorbar(state_ds, ax, plot)

    @dask.delayed()
    def _draw_frame(self, state_ds_list, rows, cols):
        fig = self._prep_figure()
        for irowcol, state_ds in enumerate(state_ds_list, 1):
            ax = self._prep_axes(state_ds, rows, cols, irowcol)
            self._draw_subplot(state_ds, ax)

        if self.watermark:
            self._update_watermark(fig)

        state = pop(state_ds, 'state', get=-1)
        buf = self._buffer_frame(state)
        return buf

    def _config_chart(self, ds, chart):
        if chart.startswith('bar'):
            kind = ds.attrs['chart_kwds'].pop('kind', 'race')
            bar_label = ds.attrs['chart_kwds'].pop('bar_label', True)
            ds.coords['tick_label'] = ds['x']
            if bar_label:
                ds['bar_label'] = ds['x']
            if kind == 'race':
                ds['x'] = ds['y'].rank('item')
            else:
                ds['x'] = ds['x'].rank('item')
                if kind == 'delta':
                    x_delta = ds['x'].diff('item').mean() / 2
                    ds['x_center'] = ds['x'] - x_delta
                    ds['delta_label'] = ds['y'].diff('item')
                    ds['y_center'] = (
                        ds['y'].shift(item=1) + ds['delta_label'] / 2)
                    ds['delta_label'] = ds['delta_label'].isel(
                        item=slice(1, None))
                    ds['delta'] = ds['delta_label'] / 2
        elif chart == 'scatter':
            kind = ds.attrs['chart_kwds'].pop('kind', 'basic')
            if kind == 'trail':
                ds['x_trail'] = ds['x'].copy()
                ds['y_trail'] = ds['y'].copy()

        legend_sortby = ds.attrs['legend_kwds'].pop('sortby', 'y')
        if legend_sortby and 'label' in ds:
            items = ds.mean('state').sortby(
                legend_sortby, ascending=False)['item']
            ds = ds.sel(item=items)
            ds['item'] = srange(len(ds['item']))

        if self.style == 'bare':
            ds.attrs['grid_kwds']['b'] = False
        elif chart == 'barh':
            ds.attrs['grid_kwds']['axis'] = (
                ds.attrs['grid_kwds'].get('axis', 'x'))
        elif chart == 'bar':
            ds.attrs['grid_kwds']['axis'] = (
                ds.attrs['grid_kwds'].get('axis', 'y'))
        else:
            ds.attrs['grid_kwds']['axis'] = (
                ds.attrs['grid_kwds'].get('axis', 'both'))

        return ds

    def _add_xy01_limits(self, ds, chart):
        limits = {
            'xlim0': ds.attrs['finalize_kwds'].pop('xlim0s'),
            'xlim1': ds.attrs['finalize_kwds'].pop('xlim1s'),
            'ylim0': ds.attrs['finalize_kwds'].pop('ylim0s'),
            'ylim1': ds.attrs['finalize_kwds'].pop('ylim1s')
        }

        if ds.attrs['finalize_kwds']['worldwide'] is None:
            if any(limit is not None for limit in limits.values()):
                ds.attrs['finalize_kwds']['worldwide'] = False
            else:
                ds.attrs['finalize_kwds']['worldwide'] = True

        if ds.attrs['finalize_kwds']['worldwide']:
            return ds

        axes_kwds = ds.attrs['axes_kwds']
        margins_kwds = ds.attrs['margins_kwds']

        for key, limit in limits.items():
            axis = key[0]
            num = int(key[-1])
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

                stat = 'min' if is_lower_limit else 'max'
                if limit == 'fixed':
                    limit = getattr(ds[axis], stat)().values
                elif limit == 'follow':
                    limit = getattr(ds[axis], stat)('item').values

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
                    limit = [limit] * self._num_states
                ds[key] = ('state', limit)
        return ds

    def _add_delays(self, ds):
        if self.delays is None:
            delays = 0.5 if self._num_states < 10 else 1 / 60
        else:
            delays = self.delays

        if isinstance(delays, (int, float)):
            transition_frames = delays
            delays = np.repeat(delays, self._num_states)
        else:
            transition_frames = (
                defaults['delays_kwds']['transition_frames'])

        delays_kwds = load_defaults(
            'delays_kwds', self.delays_kwds,
            transition_frames=transition_frames)
        aggregate = delays_kwds.pop('aggregate')

        if np.isnan(delays[-1]):
            delays[-1] = 0
        delays[-1] += delays_kwds['final_frame']

        if 'delay' in ds:
            ds['delay'] = ('state', delays + ds['delay'].values)
        else:
            ds['delay'] = ('state', delays)
        ds['delay'].attrs['transition_frames'] = transition_frames
        ds['delay'].attrs['aggregate'] = aggregate
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
        crs_kwds = load_defaults('crs_kwds', ds)
        crs = crs_kwds.pop('crs', None)

        projection_kwds = load_defaults('projection_kwds', ds)
        projection = projection_kwds.pop('projection', None)

        if crs != '' or projection != '':
            crs_obj = self._get_crs(crs, crs_kwds)
            ds.attrs['plot_kwds']['transform'] = crs_obj
            ds.attrs['inline_kwds']['transform'] = crs_obj
            ds.attrs['grid_kwds']['transform'] = crs_obj
            ds.attrs['axes_kwds']['transform'] = crs_obj

            projection_obj = self._get_crs(projection, projection_kwds)
            ds['projection'] = projection_obj
        return ds

    @staticmethod
    def _fill_null(ds):
        for var in ds.data_vars:
            if ds[var].dtype == 'O':
                ds[var] = ds[var].astype(float)
                # ds[var] = ds[var].where(~pd.isnull(ds[var]), '')
        return ds

    @staticmethod
    def _add_color_kwds(ds, chart):
        if chart.startswith('bar'):
            color = None
            if set(np.unique(ds['label'])) == set(np.unique(ds['x'])):
                ds.attrs['legend_kwds']['show'] = False
        elif 'c' not in ds and 'color' not in ds:
            color = 'black' if len(np.unique(ds['label'])) == 1 else None
            ds.attrs['plot_kwds']['c'] = ds.attrs['plot_kwds'].get(
                'c', color)

        num_colors = ds.attrs['colorbar_kwds'].pop(
            'num_colors', defaults['colorbar_kwds']['num_colors'])
        if 'c' in ds:  # TODO: add number of colors to options
            ds.attrs['plot_kwds']['cmap'] = plt.get_cmap(
                ds.attrs['plot_kwds'].get('cmap', 'plasma'), num_colors)
            ds.attrs['plot_kwds']['vmin'] = ds.attrs['plot_kwds'].get(
                'vmin', float(ds['c'].values.min()))
            ds.attrs['plot_kwds']['vmax'] = ds.attrs['plot_kwds'].get(
                'vmax', float(ds['c'].values.max()))
            ds.attrs['colorbar_kwds']['show'] = ds.attrs['plot_kwds'].get(
                'show', True)
        else:
            ds.attrs['colorbar_kwds']['show'] = False
        return ds

    @staticmethod
    def _add_base_kwds(ds):
        base_kwds = {}
        for xyc in ['x', 'y', 'c']:
            if xyc in ds:
                base_kwds[f'{xyc}tick'] = ds[xyc].min().values
                if xyc == 'c':
                    continue
                ds.attrs[f'{xyc}tick_kwds']['is_datetime'] = np.issubdtype(
                    ds[xyc].values.dtype, np.datetime64)

        for key in ['inline', 'state', 'delta', 'bar', 'ref_inline']:
            key_label = f'{key}_label'
            if key_label in ds:
                try:
                    base_kwds[key] = np.nanmin(np.diff(ds[key_label]))
                except TypeError:
                    try:
                        base_kwds[key] = np.diff(ds[key_label]).min()
                    except TypeError:
                        pass

        ds.attrs['base_kwds'] = base_kwds
        return ds

    def _interp_dataset(self, ds):
        ds = ds.reset_coords()
        ds = ds.map(self.interpolate, keep_attrs=True)
        self._num_states = len(ds['state'])

        if 'x' in ds.data_vars and 'item' not in ds.dims:
            ds = ds.expand_dims('item')
            ds['item'] = srange(len(ds['item']))

        if 'ref_chart' in ds.data_vars and 'ref_item' not in ds.dims:
            ds = ds.expand_dims('ref_item')
            ds['ref_item'] = srange(len(ds['ref_item']))
        ds['state'] = srange(self._num_states)
        return ds

    def _compress_vars(self, da):
        if isinstance(da, xr.Dataset):
            da = da.map(self._compress_vars, keep_attrs=True)
            return da

        unique_vals = np.unique(da.values)
        if len(unique_vals) == 1:
            return unique_vals[0]
        else:
            return da

    def finalize(self):
        data = deepcopy(self.data)
        if all(ds.attrs.get('finalized') for ds in data.values()):
            return data

        for rowcol, ds in data.items():
            self._num_states = len(ds['state'])
            chart = ds['chart'].values.flat[0] if 'chart' in ds else ''
            ds = self._fill_null(ds)
            ds = self._compress_vars(ds)
            if chart != '':
                ds = self._add_color_kwds(ds, chart)
            ds = self._config_chart(ds, chart)
            ds = self._add_xy01_limits(ds, chart)
            ds = self._add_base_kwds(ds)
            ds = self._add_delays(ds)
            ds = self._interp_dataset(ds)
            ds = self._add_geo_transforms(ds)
            ds.attrs['finalized'] = True
            if self.debug > 0:
                ds = ds.isel(state=slice(None, self.debug))
            self._num_states = len(ds['state'])
            data[rowcol] = ds
        return data

    def animate(self):
        data = self.finalize()
        print(data)
        rows, cols = [max(rowcol) for rowcol in zip(*data.keys())]

        delays = xr.concat((
            pop(ds, 'delay', to_numpy=False) for ds in data.values()
        ), 'item')
        delays = getattr(
            delays, delays.attrs['aggregate'])('item', keep_attrs=True)
        delays = delays.where(
            delays > 0, delays.attrs['transition_frames']).squeeze()

        jobs = []
        for state in srange(self._num_states):
            state_ds_list = [
                ds.sel(state=slice(None, state))
                if pop(ds, 'chart', get=0) == 'line' else
                ds.sel(state=state)
                for ds in data.values()
            ]
            job = self._draw_frame(state_ds_list, rows, cols)
            jobs.append(job)

        if self._num_states >= self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = self._num_states

        scheduler = 'single-threaded' if self.debug else 'processes'
        with dask.diagnostics.ProgressBar(minimum=0.5):
            buf_list = [
                buf for buf in dask.compute(
                    jobs, num_workers=num_workers,
                    scheduler=scheduler)[0]
                if buf is not None]

        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        delays = np.atleast_1d(delays)[:len(buf_list)].tolist()
        animate_kwds = dict(loop=loop, duration=delays)
        animate_kwds = load_defaults(
            'animate_kwds', self.animate_kwds, **animate_kwds)
        with imageio.get_writer(self.save_path, **animate_kwds) as writer:
            for buf in buf_list:
                image = imageio.imread(buf)
                writer.append_data(image)
                buf.close()
        optimize(self.save_path)
        print(self.save_path)
