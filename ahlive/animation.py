from io import BytesIO

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

from . import util


EXCLUDED_VARS = [
    'duration',
    'tick_label',
    'state_label',
    'inline_label',
    'x0_limit',
    'x1_limit',
    'y0_limit',
    'y1_limit'
]

SIZES = {
    'small': 15,
    'medium': 18,
    'large': 20,
    'super': 42
}


class Animation(param.Parameterized):

    ds = param.ObjectSelector(default=xr.Dataset())
    ref_ds = param.ObjectSelector(default=xr.Dataset())
    chart = param.ObjectSelector(
        default=None, objects=['scatter', 'line', 'barh', 'bar'])
    out_fp = param.String(default='untitled.gif')
    style = param.ObjectSelector(
        default=None, objects=['graph', 'minimal', 'bare'])

    figsize = param.Tuple(default=(16, 10))
    title = param.String(default=None)
    xlabel = param.String(default=None)
    ylabel = param.String(default=None)

    fig_kwds = param.Dict(default={})
    axes_kwds = param.Dict(default={})
    chart_kwds = param.Dict(default={})
    margins_kwds = param.Dict(default={})
    tick_kwds = param.Dict(default={})
    grid_kwds = param.Dict(default={})
    state_kwds = param.Dict(default={})
    inline_kwds = param.Dict(default={})
    legend_kwds = param.Dict(default={})
    colorbar_kwds = param.Dict(default={})
    frame_kwds = param.Dict(default={})
    gif_kwds = param.Dict(default={})

    num_workers = param.Integer(default=8, bounds=(1, None))
    hooks = param.List(default=None)
    logo = param.Boolean(default=True)

    def __init__(self, **kwds):
        super().__init__(**kwds)

        plt.rc('font', size=SIZES['small'])
        plt.rc('axes', labelsize=SIZES['medium'])
        plt.rc('xtick', labelsize=SIZES['small'])
        plt.rc('ytick', labelsize=SIZES['small'])
        plt.rc('legend', fontsize=SIZES['small'])
        plt.rc('figure', titlesize=SIZES['large'])

    @staticmethod
    def _get_base_format(num):
        is_datetime = isinstance(num, np.timedelta64)

        try:
            num = float(num)
        except ValueError:
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
            return '.0f'
        else:
            return f'.{abs(order_of_magnitude)}f'

    def _prep_figure(self):
        fig_kwds = {}
        fig_kwds['figsize'] = self.figsize
        fig_kwds.update(self.fig_kwds)
        fig = plt.figure(**fig_kwds)
        return fig

    def _update_style(self, ds_state, axes_kwds):
        if self.style == 'minimal':
            xlabel = axes_kwds.pop('xlabel')
            ylabel = axes_kwds.pop('ylabel')
            if xlabel:
                axes_kwds['xlabel'] = f'Higher {xlabel} ➜'
            if ylabel:
                axes_kwds['ylabel'] = f'Higher {ylabel} ➜'
            axes_kwds['xticks'] = util.try_to_pydatetime(
                float(ds_state['x'].min()),
                float(ds_state['x'].max()),
            )
            axes_kwds['yticks'] = util.try_to_pydatetime(
                float(ds_state['y'].min()),
                float(ds_state['y'].max()),
            )
        elif self.style == 'bare':
            axes_kwds['xticks'] = []
            axes_kwds['yticks'] = []
        return axes_kwds

    def _prep_axes(self, ds_state):
        limits = {
            var: ds_state[var].values[-1]
            for var in ds_state.data_vars
            if var.endswith('_limit')}
        axes_kwds = {
            'title': self.title,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'frame_on': False
        }
        if 'x0_limit' in limits or 'x1_limit' in limits:
            axes_kwds['xlim'] = util.try_to_pydatetime(
                limits.get('x0_limit'), limits.get('x1_limit'))
        if 'y0_limit' in limits or 'y1_limit' in limits:
            axes_kwds['ylim'] = util.try_to_pydatetime(
                limits.get('y0_limit'), limits.get('y1_limit'))
        axes_kwds = self._update_style(ds_state, axes_kwds)
        axes_kwds.update(self.axes_kwds)
        ax = plt.axes(**axes_kwds)
        return ax

    def _update_grid(self, ax):
        grid_kwds = {}
        if self.style == 'bare':
            grid_kwds['b'] = False
        elif self.chart == 'barh':
            grid_kwds['axis'] = 'x'
        elif self.chart == 'bar':
            grid_kwds['axis'] = 'y'
        else:
            grid_kwds['axis'] = 'both'
        grid_kwds.update(self.grid_kwds)
        ax.grid(**grid_kwds)

    def _update_margins(self, ax):
        margins_kwds = self.margins_kwds.copy()
        ax.margins(**margins_kwds)

    def _prep_kwds(self, ds_state, has_colors):
        color = 'darkgray' if len(np.unique(ds_state['label'])) == 1 else None
        base_kwds = {'color': color}
        if has_colors:
            base_kwds['cmap'] = 'RdBu_r'
            base_kwds['vmin'] = self.vmin.values
            base_kwds['vmax'] = self.vmax.values
        return base_kwds

    def _update_kwds(self, base_kwds, ds_overlay, label):
        chart_kwds = base_kwds.copy()
        chart_kwds['label'] = label
        chart_kwds.update({
            var: ds_overlay[var].values
            for var in ds_overlay
        })
        chart_kwds.update(self.chart_kwds)
        return chart_kwds

    def _add_state_labels(self, ax, chart_kwds):
        state_label = util.try_to_pydatetime(
            chart_kwds.pop('state_label')[-1])
        state_formatter = self._get_base_format(self.state_label_step)
        state_kwds = dict(
            s=f'{state_label:{state_formatter}}',
            xy=(0.975, 0.925), ha='right', va='top',
            alpha=0.5, xycoords='axes fraction',
            fontsize=SIZES['super'])
        state_kwds.update(self.state_kwds)
        ax.annotate(**state_kwds)

    def _add_inline_labels(self, ax, xs, ys, chart, chart_kwds):
        ha = 'center'
        va = 'center'
        xytext = (0, 1.5)
        if self.chart == 'barh':
            ha = 'left'
            xytext = xytext[::-1]
        elif self.chart == 'bar':
            va = 'bottom'
        elif self.chart == 'line':
            ha = 'left'
            va = 'bottom'

        color = plt.getp(chart, 'edgecolor')[0]
        inline_labels = chart_kwds.pop('inline_label')[:, -1]
        inline_formatter = self._get_base_format(self.inline_label_step)
        for i, inline_label in enumerate(inline_labels):
            inline_kwds = dict(
                s=f'{inline_label:{inline_formatter}}',
                xy=(xs[i][-1], ys[i][-1]), ha=ha, va=va,
                color=color, xytext=xytext, textcoords='offset points',
                fontsize=SIZES['medium'])
            inline_kwds.update(**self.inline_kwds)
            ax.annotate(**inline_kwds)
        return ax

    def _update_legend(self, ax, legend_labels):
        ncol = int(len(legend_labels) / 5)
        if ncol == 0:
            ncol += 1
        legend_kwds = dict(
            loc='upper left', ncol=ncol, framealpha=0,
            bbox_to_anchor=(0.025, 0.95))
        legend_kwds.update(self.legend_kwds)
        legend = ax.legend(**legend_kwds)
        legend.get_frame().set_linewidth(0)

    def _plot_chart(self, ax, xs, ys, chart_kwds):
        plot_kwds = {
            key: val for key, val in chart_kwds.items()
            if key not in EXCLUDED_VARS}
        plot_kwds['label'] = plot_kwds['label'].flat[0]
        if self.chart == 'scatter':
            chart = ax.scatter(xs, ys, **plot_kwds)
        elif self.chart == 'line':
            for x, y in zip(xs, ys): # plot each line separately
                chart = ax.scatter(x[-1], y[-1], **plot_kwds)
                plot_kwds.pop('s', '')
                plot_kwds.pop('label', '')
                _ = ax.plot(x, y, **plot_kwds)
        elif self.chart.startswith('bar'):
            chart = getattr(ax, self.chart)(
                xs.ravel(), ys.ravel(), **plot_kwds)
        return chart

    def _update_colorbar(ax, chart):
        colorbar_kwds = {'ax': ax}
        colorbar_kwds.update(self.colorbar_kwds)
        plt.colorbar(chart, colorbar_kwds)

    def _update_ticks(self, fig, ax, xs, ys, chart_kwds):
        tick_kwds = dict(axis='both', which='both', length=0, color='gray')
        tick_kwds.update(self.tick_kwds)
        ax.tick_params(**tick_kwds)

        if self.chart.startswith('bar'):
            tick_labels = chart_kwds.pop('tick_label').ravel()
            if self.chart == 'bar':
                ax.set_xticks(xs)
                ax.set_xticklabels(tick_labels)
            elif self.chart == 'barh':
                ax.set_yticks(xs)
                ax.set_yticklabels(tick_labels)
        else:
            if not self.x_is_datetime:
                xformatter = FormatStrFormatter(
                    f'%{self._get_base_format(self.xmin)}')
                ax.xaxis.set_major_formatter(xformatter)
            else:
                fig.autofmt_xdate()

            if not self.y_is_datetime:
                yformatter = FormatStrFormatter(
                    f'%{self._get_base_format(self.ymin)}')
                ax.yaxis.set_major_formatter(yformatter)
            else:
                fig.autofmt_ydate()

    @staticmethod
    def _update_logo(ax):
        ax.text(
            0.995, -0.1, 'Animated using Ahlive',
            ha='right', va='bottom', transform=ax.transAxes,
            fontsize=SIZES['small'], alpha=0.28)

    def _apply_hooks(self):
        hooks = [self.hooks] if callable(self.hooks) else self.hooks
        for hook in self.hooks:
            if not callable(hook):
                continue
            hook(fig, ax)

    def _buffer_frame(self):
        buf = BytesIO()
        frame_kwds = dict(format='png')
        frame_kwds.update(**self.frame_kwds)
        plt.savefig(buf, **frame_kwds)
        buf.seek(0)
        plt.close()
        return buf

    @dask.delayed()
    def _draw_frame(self, ds_state):
        fig = self._prep_figure()
        ax = self._prep_axes(ds_state)
        self._update_grid(ax)
        self._update_margins(ax)

        if self.logo:
            self._update_logo(ax)

        has_colors = 'c' in ds_state
        base_kwds = self._prep_kwds(ds_state, has_colors)
        for label, ds_overlay in ds_state.groupby('label'):
            chart_kwds = self._update_kwds(base_kwds, ds_overlay, label)
            xs = chart_kwds.pop('x')
            ys = chart_kwds.pop('y')
            chart = self._plot_chart(ax, xs, ys, chart_kwds)

            if 'state_label' in chart_kwds:
                self._add_state_labels(ax, chart_kwds)

            if 'inline_label' in chart_kwds:
                self._add_inline_labels(ax, xs, ys, chart, chart_kwds)

            self._update_ticks(fig, ax, xs, ys, chart_kwds)

        legend_labels = ax.get_legend_handles_labels()[1]
        if legend_labels:
            self._update_legend(ax, legend_labels)

        if has_colors:
            self._update_colorbar(chart)

        if self.hooks:
            self._apply_hooks(fig, ax)

        buf = self._buffer_frame()
        return buf

    def animate(self, ds):
        self.chart = self.chart or 'line'
        if self.style is None:
            self.style = 'minimal' if self.chart == 'scatter' else 'graph'

        if self.num_states > self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = self.num_states

        with dask.diagnostics.ProgressBar(minimum=2):
            buf_list = dask.compute([
                self._draw_frame(
                    ds.isel(**{'state': [state]})
                    if self.chart != 'line'
                    else ds.isel(**{'state': slice(None, state + 1)})
                ) for state in ds['state'].values
            ], scheduler='processes', num_workers=num_workers)[0]

        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        duration = ds['duration'].values.tolist()
        gif_kwds = dict(
            format='gif', mode='I', loop=loop,
            duration=duration, subrectangles=True)
        gif_kwds.update(self.gif_kwds)

        with imageio.get_writer(self.out_fp, **gif_kwds) as writer:
            for buf in buf_list:
                if buf is None:
                    continue
                image = imageio.imread(buf)
                writer.append_data(image)
                buf.close()
        optimize(self.out_fp)
