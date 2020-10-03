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


STATE_VARS = [
    'duration', 'label', 'x0_limit', 'x1_limit', 'y0_limit', 'y1_limit']

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
    out_fp = param.Path(default='untitled.gif')
    style = param.ObjectSelector(
        default=None, objects=['graph', 'minimal', 'bare'])
    margins = param.ClassSelector(
        default=None, class_=(tuple, dict, int, float)
    )

    figsize = param.Tuple(default=(10, 8))
    title = param.String(default=None)
    xlabel = param.String(default='x')
    ylabel = param.String(default='y')

    fig_kwds = param.Dict(default={})
    axes_kwds = param.Dict(default={})
    chart_kwds = param.Dict(default={})
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
            num /= 1e9 / 3600  # nanoseconds to seconds
            if num < 1:  # 1 second
                return '%S.%f'
            elif num < 60:  # 1 minute
                return '%M:%S'
            elif num < 3600:  # 1 hour
                return '%H:%M'
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
            xlabel = axes_kwds.pop('xlabel')
            ylabel = axes_kwds.pop('ylabel')
            axes_kwds['xticks'] = []
            axes_kwds['yticks'] = []
        return axes_kwds

    def _add_state_labels(self, ax, formatter, state_label):
        state_kwds = dict(
            ha='right', va='top', transform=ax.transAxes,
            fontsize=SIZES['super'], alpha=0.5)
        state_kwds.update(self.state_kwds)
        ax.text(0.975, 0.925, f'{state_label:{formatter}}', **state_kwds)
        return ax

    def _add_inline_labels(self, ax, xs, ys, formatter, inline_labels):
        inline_kwds = dict(
            xycoords='data', xytext=(1.5, 1.5), ha='left', va='bottom',
            textcoords='offset points')
        inline_kwds.update(**self.inline_kwds)
        for i, inline_label in enumerate(inline_labels):
            ax.annotate(
                f'{inline_label:{formatter}}', xy=(xs[i][-1], ys[i][-1]),
                **inline_kwds)
        return ax

    @staticmethod
    def _add_margins(ax, margins):
        if isinstance(margins, dict):
            ax.margins(**margins)
        elif isinstance(margins, tuple):
            ax.margins(*margins)
        else:
            ax.margins(margins)
        return ax

    @dask.delayed()
    def _draw_frame(self, ds_state):
        if len(ds_state['state']) == 0:
            return

        fig_kwds = ds_state.attrs
        fig_kwds['figsize'] = self.figsize
        fig_kwds.update(self.fig_kwds)

        limits = {
            var: ds_state[var].values[-1]
            for var in ds_state.data_vars
            if var.endswith('_limit')
        }
        axes_kwds = {
            'title': self.title,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
        }
        if 'x0_limit' in limits or 'x1_limit' in limits:
            axes_kwds['xlim'] = util.try_to_pydatetime(
                limits.get('x0_limit'), limits.get('x1_limit'))
        if 'y0_limit' in limits or 'y1_limit' in limits:
            axes_kwds['ylim'] = util.try_to_pydatetime(
                limits.get('y0_limit'), limits.get('y1_limit'))
        axes_kwds = self._update_style(ds_state, axes_kwds)
        axes_kwds.update(self.axes_kwds)

        fig = plt.figure(**fig_kwds)
        ax = plt.axes(**axes_kwds)

        color = 'darkgray' if len(np.unique(ds_state['label'])) == 1 else None
        base_kwds = {'color': color}
        has_colors = 'c' in ds_state
        if has_colors:
            base_kwds['cmap'] = 'RdBu_r'
            base_kwds['vmin'] = self.vmin.values
            base_kwds['vmax'] = self.vmax.values

        for label, ds_overlay in ds_state.groupby('label'):
            chart_kwds = base_kwds.copy()
            chart_kwds['label'] = label
            chart_kwds.update({
                var: ds_overlay[var].values
                for var in ds_overlay if var not in STATE_VARS
            })

            if 'alpha' in chart_kwds:
                chart_kwds['alpha'] = chart_kwds['alpha'][-1][-1]
            chart_kwds.update(self.chart_kwds)

            xs = chart_kwds.pop('x')
            ys = chart_kwds.pop('y')

            if 'state_label' in chart_kwds:
                state_formatter = self._get_base_format(
                    self.state_label_step)
                state_label = util.try_to_pydatetime(
                    chart_kwds.pop('state_label')[-1])
                ax = self._add_state_labels(ax, state_formatter, state_label)

            if 'inline_label' in chart_kwds:
                inline_formatter = self._get_base_format(
                    self.inline_label_step)
                inline_labels = chart_kwds.pop('inline_label')[:, -1]
                if self.chart == 'barh':
                    ax = self._add_inline_labels(
                        ax, ys, xs, inline_formatter, inline_labels)
                else:
                    ax = self._add_inline_labels(
                        ax, xs, ys, inline_formatter, inline_labels)

            if self.chart == 'scatter':
                image = ax.scatter(xs, ys, **chart_kwds)
            elif self.chart == 'line':
                for x, y in zip(xs, ys): # plot each line separately
                    image = ax.scatter(x[-1], y[-1], **chart_kwds)
                    chart_kwds.pop('s', '')
                    chart_kwds.pop('label', '')
                    _ = ax.plot(x, y, **chart_kwds)
            elif self.chart.startswith('bar'):
                image = getattr(ax, self.chart)(
                    xs.ravel(), ys.ravel(), **chart_kwds)

        plt.box(False)
        tick_kwds = dict(axis='both', which='both', length=0, color='gray')
        tick_kwds.update(self.tick_kwds)
        ax.tick_params(**tick_kwds)

        if not self.chart.startswith('bar'):
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

        if self.margins is not None:
            ax = self._add_margins(ax, self.margins)

        grid_kwds = dict()
        grid_kwds.update(self.grid_kwds)
        if grid_kwds or self.style != 'bare':
            ax.grid(**grid_kwds)

        legend_labels = ax.get_legend_handles_labels()[1]
        if legend_labels:
            ncol = int(len(legend_labels) / 5)
            if ncol == 0:
                ncol += 1
            legend_kwds = dict(
                loc='upper left', ncol=ncol, framealpha=0,
                bbox_to_anchor=(0.025, 0.95))
            legend_kwds.update(self.legend_kwds)
            legend = ax.legend(**legend_kwds)
            legend.get_frame().set_linewidth(0)

        if has_colors:
            colorbar_kwds = {}
            colorbar_kwds.update(self.colorbar_kwds)
            plt.colorbar(image, colorbar_kwds)

        if self.hooks:
            hooks = [self.hooks] if callable(self.hooks) else self.hooks
            for hook in self.hooks:
                if not callable(hook):
                    continue
                hook(fig, ax)

        buf = BytesIO()
        frame_kwds = dict(format='png')
        frame_kwds.update(**self.frame_kwds)
        plt.savefig(buf, **frame_kwds)
        buf.seek(0)
        plt.close()
        return buf

    def save(self, ds):
        self.chart = self.chart or 'line'
        if self.style is None:
            self.style = 'minimal' if self.chart == 'scatter' else 'graph'

        if not self.grid_kwds:
            if self.chart == 'barh':
                self.grid_kwds['axis'] = 'x'
            elif self.chart == 'bar':
                self.grid_kwds['axis'] = 'y'
            else:
                self.grid_kwds['axis'] = 'both'

        if self.num_states > self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = self.num_states

        with dask.diagnostics.ProgressBar(minimum=2):
            buf_list = dask.compute([
                self._draw_frame(
                    ds.isel(**{'state': [state]})
                    if self.chart != 'line'
                    else ds.isel(**{'state': slice(None, state)})
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
        optimize(self.out_fp)
