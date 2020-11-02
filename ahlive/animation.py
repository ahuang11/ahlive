from io import BytesIO
from copy import deepcopy
from collections.abc import Iterable

import param
import imageio
import numpy as np
import pandas as pd
import xarray as xr
import dask.delayed
import dask.diagnostics
from pygifsicle import optimize
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patheffects import withStroke
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from . import config, util


OPTIONS = {
    'limit': ['fixed', 'follow']
}


class Animation(param.Parameterized):

    out_fp = param.String(default='untitled.gif')
    figsize = param.NumericTuple(
        default=config.defaults['fig_kwds']['figsize'], length=2)
    watermark = param.String(default='Animated using Ahlive')
    delays = param.ClassSelector(class_=(Iterable, int, float))

    fig_kwds = param.Dict()
    animate_kwds = param.Dict()
    frame_kwds = param.Dict()
    num_workers = param.Integer(default=8, bounds=(1, None))
    return_out = param.Boolean(default=True)
    watermark_kwds = param.Dict()
    delays_kwds = param.Dict()
    debug = param.Boolean(default=True)

    _num_states = 0
    _path_effects = [withStroke(linewidth=3, alpha=0.5, foreground='white')]

    def __init__(self, **kwds):
        super().__init__(**kwds)

    @staticmethod
    def _get_base_format(num):
        if isinstance(num, xr.DataArray):
            num = num.values.item()
        is_datetime = isinstance(num, np.timedelta64)

        num = util.to_num(num)
        if isinstance(num, str):
            return 's'

        if is_datetime:
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
            return '0.1f'

        order_of_magnitude = int(np.floor(np.log10(abs(num))))
        if order_of_magnitude >= 1:
            return '0.0f'
        else:
            return f'.{abs(order_of_magnitude)}f'

    def _update_text(self, kwds, label_key, ref=None, apply_format=True):
        label = kwds.get(label_key, None)
        if isinstance(label, Iterable) and not isinstance(label, str):
            labels = []
            for i, sub_label in enumerate(kwds[label_key]):
                sub_kwds = kwds.copy()
                sub_kwds['labels'] = sub_label
                sub_kwds = self._update_text(
                    sub_kwds, label_key, ref=ref, apply_format=apply_format)
                format_ = sub_kwds['format']
                labels.append(sub_kwds[label_key])
            kwds[label_key] = labels
            kwds['format'] = format_
            kwds = {
                key: val for key, val in kwds.items()
                if key in sub_kwds.keys()}
            return kwds

        format_ = kwds.pop('format', 'auto')
        if ref is not None and format_ == 'auto':
            format_ = self._get_base_format(ref)

        if format_ != 'auto':
            if apply_format:
                label = f'{label:{format_}}'
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
        label = getattr(label, casing)()
        kwds[label_key] = label
        return kwds

    def _prep_figure(self):
        fig_kwds = util.load_defaults(
            'fig_kwds', self.fig_kwds, figsize=self.figsize)
        fig = plt.figure(**fig_kwds)
        return fig

    def _prep_axes(self, ds_state, rows, cols, irowcol):
        limits = {
            var: util.pop(ds_state, var)[-1]
            for var in list(ds_state.data_vars)
            if var[1:4] == 'lim'}

        axes_kwds = ds_state.attrs['axes_kwds']

        if 'xlim0' in limits or 'xlim1' in limits:
            axes_kwds['xlim'] = util.try_to_pydatetime(
                limits.get('xlim0'), limits.get('xlim1'))
        if 'ylim0' in limits or 'ylim1' in limits:
            axes_kwds['ylim'] = util.try_to_pydatetime(
                limits.get('ylim0'), limits.get('ylim1'))

        style = axes_kwds.pop('style', '')
        if style == 'minimal':
            for axis in ['x', 'y']:
                axis_min =  float(ds_state[axis].values.min())
                axis_max =  float(ds_state[axis].values.max())
                axis_lim = axes_kwds.get(f'{axis}lim', None)
                if axis_lim is not None:
                    axis_min = max(axis_min, axis_lim[0])
                    axis_max = min(axis_max, axis_lim[1])
                axes_kwds[f'{axis}ticks'] = util.try_to_pydatetime(
                    axis_min, axis_max)

                axis_label = axes_kwds.get(f'{axis}label', None)
                if axis_label is not None:
                    axes_kwds[f'{axis}label'] = f'Higher {axis_label} ➜'
        elif style == 'bare':
            axes_kwds['xticks'] = []
            axes_kwds['yticks'] = []

        axes_kwds = util.load_defaults('axes_kwds', ds_state, **axes_kwds)
        ax = plt.subplot(rows, cols, irowcol, **axes_kwds)
        return ax

    def _update_labels(self, ds_state, ax):
        xlabel_kwds = util.load_defaults('xlabel_kwds', ds_state)
        xlabel_kwds = self._update_text(xlabel_kwds, 'xlabel')

        ylabel_kwds = util.load_defaults('ylabel_kwds', ds_state)
        ylabel_kwds = self._update_text(ylabel_kwds, 'ylabel')

        title_kwds = util.load_defaults('title_kwds', ds_state)
        title_kwds = self._update_text(title_kwds, 'label')

        ax.set_xlabel(**xlabel_kwds)
        ax.set_ylabel(**ylabel_kwds)
        ax.set_title(**title_kwds)
        return ax

    def _update_grid(self, ds_state, ax):
        grid_kwds = util.load_defaults('grid_kwds', ds_state)
        ax.grid(**grid_kwds)
        ax.set_axisbelow(True)

    def _update_margins(self, ds_state, ax):
        margins_kwds = util.load_defaults('margins_kwds', ds_state)
        ax.margins(**margins_kwds)

    def _add_state_labels(self, ds_state, ax):
        state_label = util.pop(ds_state, 'state_label')
        if state_label is None:
            return
        state_label = util.try_to_pydatetime(state_label[-1])
        state_ref = ds_state.attrs['ref_kwds']['state']

        state_kwds = util.load_defaults('state_kwds', ds_state, s=state_label)
        state_kwds = self._update_text(state_kwds, 's', ref=state_ref)
        ax.annotate(**state_kwds)

    def _update_legend(self, ds_state, ax):
        legend_labels = ax.get_legend_handles_labels()[1]
        ncol = int(len(legend_labels) / 5)
        if ncol == 0:
            ncol += 1
        legend_kwds = dict(labels=legend_labels, ncol=ncol)
        legend_kwds = util.load_defaults(
            'legend_kwds', ds_state, **legend_kwds)

        if not legend_labels or not legend_kwds.pop('show'):
            return

        legend = ax.legend(**{
            key: val for key, val in legend_kwds.items()
            if key not in ['replacements', 'casing', 'format']
        })
        for legend_label in legend.get_texts():
            legend_label.set_path_effects(self._path_effects)
        legend.get_frame().set_linewidth(0)

    @staticmethod
    def _get_color(plot):
        try:
            color = plot.get_edgecolor()
            return tuple(color[0])
        except AttributeError:
            color = plot[0].get_facecolor()
            return color

    def _plot_chart(self, ds_overlay, ax, chart, xs, ys):
        plot_kwds = {
            var: util.pop(ds_overlay, var)[-1]
            for var in list(ds_overlay.data_vars)
        }
        plot_kwds = util.load_defaults('plot_kwds', ds_overlay, **plot_kwds)
        s = plot_kwds.pop('s', None)
        if chart == 'scatter':
            # select last state
            plot = ax.scatter(xs[-1], ys[-1], s=s, **plot_kwds)
            color = self._get_color(plot)
        elif chart == 'line':
            # select last state
            plot = ax.scatter(xs[-1], ys[-1], s=s, **plot_kwds)
            plot_kwds.pop('label', '')
            # squeeze out item
            color = self._get_color(plot)
            plot_kwds['color'] = color
            _ = ax.plot(xs, ys, **plot_kwds)
        elif chart.startswith('bar'):
            plot = getattr(ax, chart)(
                xs[-1], ys[-1], **plot_kwds)
            color = self._get_color(plot)
        return plot, color

    def _plot_trails(self, ds_overlay, ax, chart,
                     xs, ys, x_trails, y_trails, color):
        if x_trails is None or y_trails is None:
            return
        plot_kwds = {
            var: util.pop(ds_overlay, var)[-1]
            for var in list(ds_overlay.data_vars)
        }
        plot_kwds = util.load_defaults(
            'plot_kwds', ds_overlay, **plot_kwds)
        plot_kwds['alpha'] = 0.5
        chart_kwds = util.load_defaults(
            'chart_kwds', ds_overlay, base_chart=chart, **plot_kwds)
        chart_kwds['label'] = '_nolegend_'
        chart = chart_kwds.pop('chart', 'scatter')
        expire = chart_kwds.pop('expire')
        stride = chart_kwds.pop('stride')
        indices = np.where(~np.isnan(x_trails))
        x_trails = x_trails[indices][-expire - 1::stride]
        y_trails = y_trails[indices][-expire - 1::stride]
        if chart == 'line':
            x_trails = np.concatenate([x_trails, xs[-1:]])
            y_trails = np.concatenate([y_trails, ys[-1:]])
            chart_kwds.pop('s')
            chart = 'plot'
        plot = getattr(ax, chart)(x_trails, y_trails, **chart_kwds)

    def _plot_deltas(self, ds_overlay, ax, chart,
                     x_centers, y_centers,
                     deltas, delta_labels, color):
        if deltas is None:
            return
        chart_kwds = util.load_defaults(
            'chart_kwds', ds_overlay, base_chart=chart)
        if chart == 'bar':
            chart_kwds['yerr'] = deltas[-1]
            x_inlines = x_centers
            y_inlines = y_centers + np.abs(deltas)
        else:
            y_centers, x_centers = x_centers, y_centers
            x_inlines = x_centers + np.abs(deltas)
            y_inlines = y_centers
            chart_kwds['xerr'] = deltas[-1]

        ax.errorbar(x_centers[-1], y_centers[-1], **chart_kwds)
        self._add_inline_labels(
            ds_overlay, ax, chart, x_inlines, y_inlines,
            delta_labels, color, ref_key='delta')

    def _add_inline_labels(self, ds_overlay, ax, chart,
                           xs, ys, inline_labels, color,
                           ref_key='inline'):
        if inline_labels is None:
            return
        inline_ref = ds_overlay.attrs['ref_kwds'][ref_key]

        ha = 'center'
        va = 'center'
        xytext = (0, 5)
        if chart == 'barh':
            ha = 'left' if ref_key != 'bar' else 'right'
            xytext = xytext[::-1]
            if ref_key != 'delta':
                xs, ys = ys, xs
        elif chart == 'bar':
            va = 'bottom' if ref_key != 'bar' else 'top'
        elif chart in ['line', 'scatter']:
            ha = 'left'
            va = 'bottom'

        inline_kwds = dict(
            s=inline_labels[-1], xy=(xs[-1], ys[-1]), ha=ha, va=va,
            color=color, xytext=xytext, path_effects=self._path_effects)
        inline_kwds = util.load_defaults(
            'inline_kwds', ds_overlay, **inline_kwds)
        inline_kwds = self._update_text(inline_kwds, 's', ref=inline_ref)
        ax.annotate(**inline_kwds)
        return ax

    def _add_annotations(self, ds_state, ax, chart,
                         xs, ys, annotations, color):
        if annotations is None:
            return
        for x, y, annotation in zip(xs, ys, annotations):
            if annotation  == '':
                continue
            annotation = util.to_num(annotation)
            annotation_kwds = dict(
                s=annotation, xy=(x, y),
                color=color, path_effects=self._path_effects
            )
            annotation_kwds = util.load_defaults(
                'annotation_kwds', self.annotation_kwds, **annotation_kwds)
            annotation_kwds = self._update_text(
                annotation_kwds, 's', ref=annotation)
            ax.annotate(**annotation_kwds)
            ax.scatter(x, y, color=color)

    def _update_ticks(self, ds_state, ax, chart):
        if chart.startswith('bar'):
            ds_state = ds_state.isel(state=-1)
        tick_labels = util.pop(ds_state, 'tick_label')

        xtick_ref = ds_state.attrs['ref_kwds']['xtick']
        xtick_kwds = util.load_defaults(
            'xtick_kwds', ds_state, labels=tick_labels)
        xtick_kwds = self._update_text(
            xtick_kwds, 'labels', ref=xtick_ref, apply_format=False)
        xformat = xtick_kwds.pop('format')
        xtick_labels = xtick_kwds.pop('labels')
        x_is_datetime = xtick_kwds.pop('is_datetime')

        ytick_ref = ds_state.attrs['ref_kwds']['ytick']
        ytick_kwds = util.load_defaults(
            'ytick_kwds', ds_state, labels=tick_labels)
        ytick_kwds = self._update_text(
            ytick_kwds, 'labels', ref=ytick_ref, apply_format=False)
        yformat = ytick_kwds.pop('format')
        ytick_labels = ytick_kwds.pop('labels')
        y_is_datetime = ytick_kwds.pop('is_datetime')

        if chart.startswith('bar'):
            xs = util.pop(ds_state, 'x')
            if chart == 'bar':
                ax.set_xticks(xs)
                ax.set_xticklabels(xtick_labels)
            elif chart == 'barh':
                ax.set_yticks(xs)
                ax.set_yticklabels(ytick_labels)
        else:
            if not x_is_datetime:
                xformatter = FormatStrFormatter(f'%{xformat}')
                ax.xaxis.set_major_formatter(xformatter)
            else:
                xlocator = mdates.AutoDateLocator(minticks=5, maxticks=10)
                xformatter = mdates.ConciseDateFormatter(xlocator)
                ax.xaxis.set_major_locator(xlocator)
                ax.xaxis.set_major_formatter(xformatter)

            if not y_is_datetime:
                yformatter = FormatStrFormatter(f'%{yformat}')
                ax.yaxis.set_major_formatter(yformatter)
            else:
                ylocator = mdates.AutoDateLocator(minticks=5, maxticks=10)
                yformatter = mdates.ConciseDateFormatter(ylocator)
                ax.yaxis.set_major_locator(ylocator)
                ax.yaxis.set_major_formatter(yformatter)

        ax.tick_params(**xtick_kwds)
        ax.tick_params(**ytick_kwds)

    def _update_colorbar(self, ds_state, ax, plot):
        colorbar_kwds = util.load_defaults(
            'colorbar_kwds', ds_state, ax=ax)

        if not colorbar_kwds.pop('show'):
            return

        colorbar = plt.colorbar(plot, **colorbar_kwds)
        cax = colorbar.ax

        clabel_kwds = util.load_defaults(
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

        tick_labels = colorbar.get_ticks()
        ctick_kwds = util.load_defaults('ctick_kwds', self.ctick_kwds,
            labels=tick_labels, **ctick_kwds)
        ctick_kwds = self._update_text(
            ctick_kwds, 'labels', ref=self.cmin, apply_format=False)
        cformat = ctick_kwds.pop('format')
        cformatter = FormatStrFormatter(f'%{cformat}')
        ctick_labels = ctick_kwds.pop('labels')
        if colorbar_kwds['orientation'] == 'vertical':
            cax.set_yticklabels(ctick_labels)
            cax.yaxis.set_major_formatter(cformatter)
        else:
            cax.set_xticklabels(ctick_labels)
            cax.xaxis.set_major_formatter(cformatter)
        cax.tick_params(**ctick_kwds)

    def _update_watermark(self, fig):
        watermark_kwds = util.load_defaults(
            'watermark_kwds', self.watermark_kwds, s=self.watermark)
        fig.text(.995, .005, **watermark_kwds)

    def _apply_hooks(self, fig, ax):
        hooks = [self.hooks] if callable(self.hooks) else self.hooks
        for hook in self.hooks:
            if not callable(hook):
                continue
            hook(fig, ax)

    def _buffer_frame(self):
        buf = BytesIO()
        frame_kwds = util.load_defaults('frame_kwds', self.frame_kwds)
        plt.savefig(buf, **frame_kwds)
        buf.seek(0)
        plt.close()
        return buf

    def _draw_subplot(self, ds_state, ax):
        self._update_labels(ds_state, ax)
        self._update_grid(ds_state, ax)
        self._update_margins(ds_state, ax)
        self._add_state_labels(ds_state, ax)

        for item in np.atleast_1d(ds_state['item']):
            ds_overlay = ds_state.sel(item=item)
            chart = util.pop(ds_overlay, 'chart').item()

            xs = util.pop(ds_overlay, 'x')
            ys = util.pop(ds_overlay, 'y')
            x_trails = util.pop(ds_overlay, 'x_trail')
            y_trails = util.pop(ds_overlay, 'y_trail')
            x_centers = util.pop(ds_overlay, 'x_center')
            y_centers = util.pop(ds_overlay, 'y_center')
            deltas = util.pop(ds_overlay, 'delta')
            delta_labels = util.pop(ds_overlay, 'delta_label')
            bar_labels = util.pop(ds_overlay, 'bar_label')
            inline_labels = util.pop(ds_overlay, 'inline_label')
            annotations = util.pop(ds_overlay, 'annotation')

            plot, color = self._plot_chart(ds_overlay, ax, chart, xs, ys)
            self._plot_trails(
                ds_overlay, ax, chart, xs, ys, x_trails, y_trails, color)
            self._plot_deltas(
                ds_overlay, ax, chart, x_centers, y_centers,
                deltas, delta_labels, color)
            self._add_inline_labels(
                ds_overlay, ax, chart, xs, ys, bar_labels, 'black',
                ref_key='bar')
            self._add_inline_labels(
                ds_overlay, ax, chart, xs, ys, inline_labels, color)
            self._add_annotations(
                ds_overlay, ax, chart, xs, ys, annotations, color)
        self._update_ticks(ds_state, ax, chart)
        self._update_legend(ds_state, ax)
        self._update_colorbar(ds_state, ax, plot)

    @dask.delayed()
    def _draw_frame(self, data, state, rows, cols):
        if state == 0:
            return

        fig = self._prep_figure()

        for irowcol, ds in enumerate(data.values(), 1):
            ds_state = ds.isel(state=slice(None, state))
            ax = self._prep_axes(ds_state, rows, cols, irowcol)
            self._draw_subplot(ds_state, ax)

        if self.watermark:
            self._update_watermark(fig)

        buf = self._buffer_frame()
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
            ds['item'] = np.arange(len(ds['item']))

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
            'xlim0': ds.attrs.pop('xlim0s'),
            'xlim1': ds.attrs.pop('xlim1s'),
            'ylim0': ds.attrs.pop('ylim0s'),
            'ylim1': ds.attrs.pop('ylim1s')
        }

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
                if util.is_scalar(limit) == 1:
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
                config.defaults['delays_kwds']['transition_frames'])

        delays_kwds = util.load_defaults(
            'delays_kwds', self.delays_kwds,
            transition_frames=transition_frames)
        aggregate = delays_kwds.pop('aggregate')

        if np.isnan(delays[-1]):
            delays[-1] = 0
        delays[-2:] += delays_kwds['final_frame']
        if 'delay' in ds:
            ds['delay'] = ds['delay'] + delays
        else:
            ds['delay'] = ('state', delays)
        ds['delay'].attrs['transition_frames'] = transition_frames
        ds['delay'].attrs['aggregate'] = aggregate
        return ds

    @staticmethod
    def _fill_null(ds):
        for var in ds.data_vars:
            if ds[var].dtype == 'O':
                ds[var] = ds[var].where(~pd.isnull(ds[var]), '')
        return ds

    @staticmethod
    def _add_color_kwds(ds, chart):
        if chart.startswith('bar'):
            color = None
            if set(np.unique(ds['label'])) == set(np.unique(ds['x'])):
                ds.attrs['legend_kwds']['show'] = False
        else:
            color = 'darkgray' if len(np.unique(ds['label'])) == 1 else None

        ds.attrs['plot_kwds']['color'] = ds.attrs['plot_kwds'].get(
            'color', color)

        if 'c' in ds:
            ds.attrs['plot_kwds']['cmap'] = ds.attrs['plot_kwds'].get(
                'cmap', 'RdBu_r')
            ds.attrs['vmin'] = ds.attrs['plot_kwds'].get(
                'vmin', float(ds['c'].values.min()))
            ds.attrs['vmax'] = ds.attrs['plot_kwds'].get(
                'vmax', float(ds['c'].values.max()))
            ds.attrs['colorbar_kwds']['show'] = ds.attrs['plot_kwds'].get(
                'show', True)
        else:
            ds.attrs['colorbar_kwds']['show'] = False
        return ds

    @staticmethod
    def _add_ref_kwds(ds):
        ref_kwds = {}
        for xyc in ['x', 'y', 'c']:
            if xyc in ds:
                ref_kwds[f'{xyc}tick'] = ds[xyc].min().values
                if xyc == 'c':
                    continue
                ds.attrs[f'{xyc}tick_kwds']['is_datetime'] = np.issubdtype(
                    ds[xyc].values.dtype, np.datetime64)

        for key in ['inline', 'state', 'delta', 'bar']:
            key_label = f'{key}_label'
            if key_label in ds:
                try:
                    ref_kwds[key] = np.nanmin(np.diff(ds[key_label]))
                except TypeError:
                    ref_kwds[key] = ''

        ds.attrs['ref_kwds'] = ref_kwds
        return ds

    def _interp_dataset(self, ds):
        ds = ds.reset_coords()
        ds = ds.map(self.interpolate, keep_attrs=True)
        ds = ds.set_coords('root')
        interp_indices = ds['root'] > 0
        ds['root'] = np.cumsum(ds['root'])
        ds['root'] = ds['root'].where(interp_indices)
        self._num_states = len(ds['state'])
        return ds

    def finalize(self):
        data = deepcopy(self.data)
        if all(ds.attrs.get('finalized') for ds in data.values()):
            return data

        for rowcol, ds in data.items():
            chart = ds['chart'].values.flat[0]
            ds = self._fill_null(ds)
            ds = self._add_color_kwds(ds, chart)
            ds = self._config_chart(ds, chart)
            ds = self._add_xy01_limits(ds, chart)
            ds = self._add_ref_kwds(ds)
            ds = self._add_delays(ds)
            ds = self._interp_dataset(ds)
            ds.attrs['finalized'] = True
            data[rowcol] = ds

        return data

    def animate(self):
        data = self.finalize()
        print(data)
        rows, cols = [max(rowcol) for rowcol in zip(*data.keys())]

        delays = xr.concat((
            util.pop(ds, 'delay', to_numpy=False) for ds in data.values()
        ), 'item')
        delays = getattr(
            delays, delays.attrs['aggregate'])('item', keep_attrs=True)
        delays = delays.where(delays != 0, delays.attrs['transition_frames'])

        jobs = []
        for state in range(self._num_states):
            job = self._draw_frame(data, state, rows, cols)
            jobs.append(job)

        if self._num_states > self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = self._num_states

        scheduler = 'single-threaded' if self.debug else 'processes'
        with dask.diagnostics.ProgressBar(minimum=0.5):
            buf_list = dask.compute(
                jobs, num_workers=num_workers,
                scheduler=scheduler)[0]

        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        animate_kwds = dict(loop=loop, duration=list(delays))
        animate_kwds = util.load_defaults(
            'animate_kwds', self.animate_kwds, **animate_kwds)
        with imageio.get_writer(self.out_fp, **animate_kwds) as writer:
            for buf in buf_list:
                if buf is None:
                    continue
                image = imageio.imread(buf)
                writer.append_data(image)
                buf.close()
        print(self.out_fp)
        optimize(self.out_fp)
