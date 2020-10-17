from io import BytesIO
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

from . import config, util


class Animation(param.Parameterized):

    chart = param.ObjectSelector(
        default=None, objects=['scatter', 'line', 'barh', 'bar', 'plot'])
    trail_chart = param.ObjectSelector(
        default=None, objects=['scatter', 'line', 'plot'])
    out_fp = param.String(default='untitled.gif')
    style = param.ObjectSelector(
        default=None, objects=['graph', 'minimal', 'bare'])

    figsize = param.Tuple(default=config.defaults['fig_kwds']['figsize'])
    title = param.String(default='')
    xlabel = param.String(default=None)
    ylabel = param.String(default=None)
    clabel = param.String(default=None)
    watermark = param.String(default='Animated using Ahlive')
    legend = param.Boolean(default=None)
    colorbar = param.Boolean(default=None)

    fig_kwds = param.Dict(default=None)
    axes_kwds = param.Dict(default=None)
    plot_kwds = param.Dict(default=None)
    chart_kwds = param.Dict(default=None)
    trail_kwds = param.Dict(default=None)
    annotation_kwds = param.Dict(default=None)
    grid_kwds = param.Dict(default=None)
    margins_kwds = param.Dict(default=None)
    xlabel_kwds = param.Dict(default=None)
    ylabel_kwds = param.Dict(default=None)
    clabel_kwds = param.Dict(default=None)
    title_kwds = param.Dict(default=None)
    xtick_kwds = param.Dict(default=None)
    ytick_kwds = param.Dict(default=None)
    ctick_kwds = param.Dict(default=None)
    state_kwds = param.Dict(default=None)
    inline_kwds = param.Dict(default=None)
    legend_kwds = param.Dict(default=None)
    colorbar_kwds = param.Dict(default=None)
    watermark_kwds = param.Dict(default=None)
    frame_kwds = param.Dict(default=None)
    animate_kwds = param.Dict(default=None)

    num_workers = param.Integer(default=8, bounds=(1, None))
    hooks = param.List(default=None)

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._path_effects = [
            withStroke(linewidth=3, alpha=0.5, foreground='white')]

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

    def _update_text(self, kwds, label_key, num=None, apply_format=True):
        label = kwds.get(label_key, None)
        if isinstance(label, Iterable) and not isinstance(label, str):
            labels = []
            for i, label in enumerate(kwds[label_key]):
                sub_kwds = kwds.copy()
                sub_kwds['labels'] = label
                sub_kwds = self._update_text(
                    sub_kwds, label_key, num=num, apply_format=apply_format)
                format_ = sub_kwds['format']
                labels.append(sub_kwds[label_key])
            kwds[label_key] = labels
            kwds['format'] = format_
            kwds = {
                key: val for key, val in kwds.items()
                if key in sub_kwds.keys()}
            return kwds

        format_ = kwds.pop('format', 'auto')
        if num is not None and format_ == 'auto':
            format_ = self._get_base_format(num)

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
        fig_kwds = config._load(
            'fig_kwds', self.fig_kwds, figsize=self.figsize)
        fig = plt.figure(**fig_kwds)
        return fig

    def _update_style(self, ds_state, axes_kwds):
        xlabel = self.xlabel
        ylabel = self.ylabel
        if self.style == 'minimal':
            axes_kwds['xticks'] = util.try_to_pydatetime(
                float(ds_state['x'].min()),
                float(ds_state['x'].max()),
            )
            axes_kwds['yticks'] = util.try_to_pydatetime(
                float(ds_state['y'].min()),
                float(ds_state['y'].max()),
            )
            if xlabel:
                xlabel = f'Higher {xlabel} ➜'
            if ylabel:
                ylabel = f'Higher {ylabel} ➜'
        elif self.style == 'bare':
            axes_kwds['xticks'] = []
            axes_kwds['yticks'] = []
        return axes_kwds, xlabel, ylabel

    def _prep_axes(self, ds_state):
        limits = {
            var: util.pop(ds_state, var)[-1]
            for var in list(ds_state.data_vars)
            if var[1:4] == 'lim'}

        axes_kwds = {}
        if 'xlim0' in limits or 'xlim1' in limits:
            axes_kwds['xlim'] = util.try_to_pydatetime(
                limits.get('xlim0'), limits.get('xlim1'))
        if 'ylim0' in limits or 'ylim1' in limits:
            axes_kwds['ylim'] = util.try_to_pydatetime(
                limits.get('ylim0'), limits.get('ylim1'))

        axes_kwds, xlabel, ylabel = self._update_style(ds_state, axes_kwds)
        axes_kwds = config._load('axes_kwds', self.axes_kwds, **axes_kwds)

        xlabel_kwds = config._load(
            'xlabel_kwds', self.xlabel_kwds, xlabel=xlabel)
        xlabel_kwds = self._update_text(xlabel_kwds, 'xlabel')

        ylabel_kwds = config._load(
            'ylabel_kwds', self.ylabel_kwds, ylabel=ylabel)
        ylabel_kwds = self._update_text(ylabel_kwds, 'ylabel')

        title_kwds = config._load(
            'title_kwds', self.title_kwds, label=self.title)
        title_kwds = self._update_text(title_kwds, 'label')

        ax = plt.axes(**axes_kwds)
        ax.set_xlabel(**xlabel_kwds)
        ax.set_ylabel(**ylabel_kwds)
        ax.set_title(**title_kwds)
        return ax

    def _update_grid(self, ax):
        grid_kwds = {}
        if self.style == 'bare':
            grid_kwds['b'] = False
        if self.chart == 'barh':
            grid_kwds['axis'] = 'x'
        elif self.chart == 'bar':
            grid_kwds['axis'] = 'y'
        else:
            grid_kwds['axis'] = 'both'
        grid_kwds = config._load('grid_kwds', self.grid_kwds, **grid_kwds)
        ax.grid(**grid_kwds)

    def _update_margins(self, ax):
        margins_kwds = config._load('margins_kwds', self.margins_kwds)
        ax.margins(**margins_kwds)

    def _update_watermark(self, fig):
        watermark_kwds = config._load(
            'watermark_kwds', self.watermark_kwds, s=self.watermark)
        fig.text(.995, .005, **watermark_kwds)

    def _prep_kwds(self, ds_state, has_colors):
        color = 'darkgray' if len(np.unique(ds_state['label'])) == 1 else None
        base_kwds = {'color': color}
        if has_colors:
            base_kwds['cmap'] = 'RdBu_r'
            base_kwds['vmin'] = self.vmin
            base_kwds['vmax'] = self.vmax
        return base_kwds

    def _update_kwds(self, base_kwds, ds_overlay):
        plot_kwds = base_kwds.copy()
        plot_kwds.update({
            var: util.pop(ds_overlay, var)[-1]
            for var in list(ds_overlay.data_vars)
        })
        plot_kwds = config._load('plot_kwds', self.plot_kwds, **plot_kwds)
        return plot_kwds

    def _add_state_labels(self, ax, state_label):
        state_label = util.try_to_pydatetime(state_label)
        state_kwds = dict(s=state_label)
        state_kwds = config._load('state_kwds', self.state_kwds, **state_kwds)
        state_kwds = self._update_text(state_kwds, 's', num=self.state_step)
        ax.annotate(**state_kwds)

    def _update_legend(self, ax, legend_labels):
        ncol = int(len(legend_labels) / 5)
        if ncol == 0:
            ncol += 1
        legend_kwds = dict(labels=legend_labels, ncol=ncol)
        legend_kwds = config._load(
            'legend_kwds', self.legend_kwds, **legend_kwds)
        show = legend_kwds.pop('show')
        if show:
            legend = ax.legend(**{
                key: val for key, val in legend_kwds.items()
                if key not in ['replacements', 'casing', 'format']
            })
            for legend_label in legend.get_texts():
                legend_label.set_path_effects(self._path_effects)
            legend.get_frame().set_linewidth(0)

    @staticmethod
    def _get_color(chart):
        try:
            color = chart.get_edgecolor()
        except AttributeError:
            color = chart[0].get_facecolor()
        return color

    def _plot_chart(self, ax, xs, ys, plot_kwds):
        s = plot_kwds.pop('s', None)

        if self.chart == 'scatter':
            # select last state
            chart = ax.scatter(xs[-1], ys[-1], s=s, **plot_kwds)
        elif self.chart == 'line':
            # select last state
            chart = ax.scatter(xs[-1], ys[-1], s=s, **plot_kwds)
            plot_kwds.pop('label', '')
            # squeeze out item
            _ = ax.plot(xs, ys, **plot_kwds)
        elif self.chart.startswith('bar'):
            chart = getattr(ax, self.chart)(
                xs[-1], ys[-1], **plot_kwds)
        return chart

    def _plot_trails(self, ax, x_trails, y_trails, plot_kwds, color):
        trail_kwds = plot_kwds.copy()
        trail_kwds['color'] = color
        trail_kwds = config._load(
            'trail_kwds', self.trail_kwds, **trail_kwds)
        trail_kwds['label'] = '_nolegend_'
        expire = trail_kwds.pop('expire', None)
        stride = trail_kwds.pop('stride', None)
        indices = np.where(~np.isnan(x_trails))
        x_trails = x_trails[indices][:-expire:-stride]
        y_trails = y_trails[indices][:-expire:-stride]
        if self.trail_chart in ['line', 'plot']:
            x_trails = np.concatenate([xs[-1:], x_trails])
            y_trails = np.concatenate([ys[-1:], y_trails])
            self.trail_chart = 'plot'
        getattr(ax, self.trail_chart)(x_trails, y_trails, **trail_kwds)

    def _plot_diff(self, ax, x_centers, y_centers, y_diffs, color):
        chart_kwds = dict(color=color, chart=self.chart)
        chart_kwds = config._load(
            'chart_kwds', self.chart_kwds, **chart_kwds)
        chart_kwds.pop('kind')
        ax.errorbar(x_centers, y_centers, y_diffs, **chart_kwds)

        inline_kwds = dict(
            s=y_diffs[-1] * 2,
            xy=(x_centers[-1], y_centers[-1] + y_diffs[-1]),
            ha='center', va='bottom',
            color=color, xytext=(0, 1.5),
            path_effects=self._path_effects
        )
        inline_kwds = config._load(
            'inline_kwds', self.inline_kwds, **inline_kwds)
        inline_kwds = self._update_text(inline_kwds, 's', num=self.inline_step)
        ax.annotate(**inline_kwds)

    def _add_inline_labels(self, ax, xs, ys, color, inline_labels):
        ha = 'center'
        va = 'center'
        xytext = (0, 1.5)
        if self.chart == 'barh':
            ha = 'left'
            xytext = xytext[::-1]
        elif self.chart == 'bar':
            va = 'bottom'
        elif self.chart in ['line', 'scatter']:
            ha = 'left'
            va = 'bottom'

        inline_kwds = dict(
            s=inline_labels[-1],
            xy=(xs[-1], ys[-1]), ha=ha, va=va,
            color=color, xytext=xytext,
            path_effects=self._path_effects
        )
        inline_kwds = config._load(
            'inline_kwds', self.inline_kwds, **inline_kwds)
        inline_kwds = self._update_text(
            inline_kwds, 's', num=self.inline_step)
        ax.annotate(**inline_kwds)
        return ax

    def _add_annotations(self, ax, xs, ys, color, annotations):
        for x, y, annotation in zip(xs, ys, annotations):
            if annotation  == '':
                continue
            annotation = util.to_num(annotation)
            annotation_kwds = dict(
                s=annotation, xy=(x, y),
                color=color, path_effects=self._path_effects
            )
            annotation_kwds = config._load(
                'annotation_kwds', self.annotation_kwds, **annotation_kwds)
            annotation_kwds = self._update_text(
                annotation_kwds, 's', num=annotation)
            ax.annotate(**annotation_kwds)
            ax.scatter(x, y, color=color)

    def _update_ticks(self, fig, ax, ds_state, tick_labels):
        xtick_kwds = config._load(
            'xtick_kwds', self.xtick_kwds, labels=tick_labels)
        xtick_kwds = self._update_text(
            xtick_kwds, 'labels', num=self.xmin, apply_format=False)
        xformat = xtick_kwds.pop('format')
        xtick_labels = xtick_kwds.pop('labels')

        ytick_kwds = config._load('ytick_kwds', self.ytick_kwds,
            labels=tick_labels)
        ytick_kwds = self._update_text(
            ytick_kwds, 'labels', num=self.ymin, apply_format=False)
        yformat = ytick_kwds.pop('format')
        ytick_labels = ytick_kwds.pop('labels')

        if self.chart.startswith('bar'):
            xs = util.pop(ds_state, 'x')
            if self.chart == 'bar':
                ax.set_xticks(xs)
                ax.set_xticklabels(xtick_labels)
            elif self.chart == 'barh':
                ax.set_yticks(xs)
                ax.set_yticklabels(ytick_labels)
        else:
            if not self.x_is_datetime:
                xformatter = FormatStrFormatter(f'%{xformat}')
                ax.xaxis.set_major_formatter(xformatter)
            else:
                fig.autofmt_xdate()

            if not self.y_is_datetime:
                yformatter = FormatStrFormatter(f'%{yformat}')
                ax.yaxis.set_major_formatter(yformatter)
            else:
                fig.autofmt_ydate()

        ax.tick_params(**xtick_kwds)
        ax.tick_params(**ytick_kwds)

    def _update_colorbar(self, ax, chart):  # TODO: add defaults
        colorbar_kwds = {'ax': ax}
        colorbar_kwds = config._load(
            'colorbar_kwds', self.colorbar_kwds, **colorbar_kwds)
        colorbar = plt.colorbar(chart, **colorbar_kwds)
        cax = colorbar.ax

        clabel_kwds = config._load(
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
        ctick_kwds = config._load('ctick_kwds', self.ctick_kwds,
            labels=tick_labels, **ctick_kwds)
        ctick_kwds = self._update_text(
            ctick_kwds, 'labels', num=self.cmin, apply_format=False)
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

    def _apply_hooks(self, fig, ax):
        hooks = [self.hooks] if callable(self.hooks) else self.hooks
        for hook in self.hooks:
            if not callable(hook):
                continue
            hook(fig, ax)

    def _buffer_frame(self):
        buf = BytesIO()
        frame_kwds = config._load('frame_kwds', self.frame_kwds)
        plt.savefig(buf, **frame_kwds)
        buf.seek(0)
        plt.close()
        return buf

    @dask.delayed()
    def _draw_frame(self, ds_state):
        if self.chart.startswith('bar'):
            ds_state = ds_state.isel(state=-1).squeeze()

        fig = self._prep_figure()
        ax = self._prep_axes(ds_state)
        self._update_grid(ax)
        self._update_margins(ax)

        if self.watermark:
            self._update_watermark(fig)

        has_colors = 'c' in ds_state
        base_kwds = self._prep_kwds(ds_state, has_colors)

        state_label = util.pop(ds_state, 'state_label')[-1]
        if state_label is not None:
            self._add_state_labels(ax, state_label)

        tick_labels = util.pop(ds_state, 'tick_label')
        for item in np.atleast_1d(ds_state['item']):
            ds_overlay = ds_state.sel(item=[item])

            xs = util.pop(ds_overlay, 'x')
            ys = util.pop(ds_overlay, 'y')

            x_trails = util.pop(ds_overlay, 'x_trail')
            y_trails = util.pop(ds_overlay, 'y_trail')

            x_centers = util.pop(ds_overlay, 'x_center')
            y_centers = util.pop(ds_overlay, 'y_center')
            y_diffs = util.pop(ds_overlay, 'y_diff')

            annotations = util.pop(ds_overlay, 'annotation')

            inline_labels = util.pop(ds_overlay, 'inline_label')

            plot_kwds = self._update_kwds(base_kwds, ds_overlay)
            chart = self._plot_chart(ax, xs, ys, plot_kwds)

            if 'cmap' in plot_kwds.keys():
                color = 'black'
            else:
                color = self._get_color(chart)

            if x_trails is not None and y_trails is not None:
                self._plot_trails(ax, x_trails, y_trails, plot_kwds, color)
            if x_centers is not None and y_diffs is not None:
                self._plot_diff(ax, x_centers, y_centers, y_diffs, color)
            if inline_labels is not None:
                self._add_inline_labels(ax, xs, ys, color, inline_labels)
            if annotations is not None:
                self._add_annotations(ax, xs, ys, color, annotations)

        self._update_ticks(fig, ax, ds_state, tick_labels)

        legend_labels = ax.get_legend_handles_labels()[1]
        if legend_labels and self.legend:
            self._update_legend(ax, legend_labels)

        if has_colors and self.colorbar:
            self._update_colorbar(ax, chart)

        if self.hooks:
            self._apply_hooks(fig, ax)

        buf = self._buffer_frame()
        return buf

    def animate(self):
        ds = self.final_ds.copy()
        delays = util.pop(ds, 'delay').tolist()

        self.chart = self.chart or 'line'
        if self.legend is None:
            self.legend = True
        if self.colorbar is None:
            self.colorbar = True
        if self.style is None:
            self.style = 'minimal' if self.chart == 'scatter' else 'graph'

        if self.num_states > self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = self.num_states

        with dask.diagnostics.ProgressBar(minimum=2):
            buf_list = dask.compute([
                self._draw_frame(
                    ds.isel(**{'state': slice(None, state + 1)})
                ) for state in ds['state'].values
            ], scheduler='processes', num_workers=num_workers)[0]

        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        animate_kwds = dict(loop=loop, duration=delays)
        animate_kwds = config._load(
            'animate_kwds', self.animate_kwds, **animate_kwds)
        with imageio.get_writer(self.out_fp, **animate_kwds) as writer:
            for buf in buf_list:
                if buf is None:
                    continue
                image = imageio.imread(buf)
                writer.append_data(image)
                buf.close()
        optimize(self.out_fp)
