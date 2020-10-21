import warnings
from copy import deepcopy
from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from . import config, easing, animation, util


OPTIONS = {
    'limit': ['fixed', 'follow']
}


class Data(easing.Easing, animation.Animation):

    xs = param.ClassSelector(class_=(Iterable,), allow_None=False)
    ys = param.ClassSelector(class_=(Iterable,), allow_None=False)

    chart = param.ObjectSelector(
        default=None, objects=['scatter', 'line', 'barh', 'bar', 'plot'])
    label = param.String()
    figsize = param.NumericTuple(
        default=config.defaults['fig_kwds']['figsize'], length=2)
    title = param.String(default='')
    xlabel = param.String(default=None)
    ylabel = param.String(default=None)
    watermark = param.String(default='Animated using Ahlive')
    legend = param.Boolean(default=None)
    style = param.ObjectSelector(
        default=None, objects=['graph', 'minimal', 'bare'])
    delays = param.ClassSelector(
        default=None, class_=(Iterable, int, float))

    fig_kwds = param.Dict(default=None)
    axes_kwds = param.Dict(default=None)
    plot_kwds = param.Dict(default=None)
    chart_kwds = param.Dict(default=None)
    grid_kwds = param.Dict(default=None)
    margins_kwds = param.Dict(default=None)
    title_kwds = param.Dict(default=None)
    xlabel_kwds = param.Dict(default=None)
    ylabel_kwds = param.Dict(default=None)
    watermark_kwds = param.Dict(default=None)
    colorbar_kwds = param.Dict(default=None)
    legend_kwds = param.Dict(default=None)
    xtick_kwds = param.Dict(default=None)
    ytick_kwds = param.Dict(default=None)
    annotation_kwds = param.Dict(default=None)
    delays_kwds = param.Dict(default=None)

    xlim0s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    xlim1s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    ylim0s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    ylim1s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))

    subplot = param.Integer(default=1, bounds=(1, None))

    def __init__(self, xs, ys, **kwds):
        self_attrs = set(dir(self))
        input_data_vars = {
            key: kwds.pop(key) for key in kwds
            if key not in self_attrs
        }
        super().__init__(**kwds)
        for key in list(input_data_vars.keys()):
            key_and_s = key + 's'
            key_strip = key.rstrip('s')
            expected_key = None
            if key_and_s in self_attrs and key_and_s != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_and_s}!')
                expected_key = key_and_s
            elif key_strip in self_attrs and key_strip != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_strip}!')
                expected_key = key_strip
            if expected_key:
                setattr(self, expected_key, input_data_vars.pop(key))

        num_states = len(xs)
        if self.chart is None:
            chart = 'scatter' if num_states <= 5 else 'line'
        else:
            chart = self.chart

        label = '' if self.label is None else self.label

        fig_kwds = config._load(
            'fig_kwds', self.fig_kwds, figsize=self.figsize)
        axes_kwds = config._load('axes_kwds', self.axes_kwds)
        plot_kwds = config._load('plot_kwds', self.plot_kwds)
        chart_kwds = config._load(
            'chart_kwds', self.chart_kwds, chart=self.chart)
        margins_kwds = config._load('margins_kwds', self.margins_kwds)
        grid_kwds = config._load('grid_kwds', self.grid_kwds)
        title_kwds = config._load(
            'title_kwds', self.title_kwds, label=self.title)
        xlabel_kwds = config._load(
            'xlabel_kwds', self.xlabel_kwds, xlabel=self.xlabel)
        ylabel_kwds = config._load(
            'ylabel_kwds', self.ylabel_kwds, ylabel=self.ylabel)
        watermark_kwds = config._load(
            'watermark_kwds', self.watermark_kwds, s=self.watermark)
        legend_kwds = config._load(
            'legend_kwds', self.legend_kwds, show=self.legend)
        xtick_kwds = config._load('xtick_kwds', self.xtick_kwds)
        ytick_kwds = config._load('ytick_kwds', self.ytick_kwds)
        annotation_kwds = config._load(
            'annotation_kwds', self.annotation_kwds)
        delays_kwds = config._load(
            'delays_kwds', self.delays_kwds, delays=self.delays)

        data_vars = {'x': xs, 'y': ys}
        data_vars.update({
            key: val for key, val in input_data_vars.items()
            if val is not None
        })
        for var in list(data_vars.keys()):
            val = np.array(data_vars.pop(var))
            dims = ('item', 'state')
            if util.is_scalar(val):
                val = [val] * num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = dims, val

        coords = {
            'item': [1],
            'chart': ('item', [chart]),
            'label': ('item', [label]),
            'state': np.arange(num_states),
        }

        attrs = {
            'fig_kwds': fig_kwds,
            'axes_kwds': axes_kwds,
            'plot_kwds': plot_kwds,
            'chart_kwds': chart_kwds,
            'grid_kwds': grid_kwds,
            'margins_kwds': margins_kwds,
            'title_kwds': title_kwds,
            'xlabel_kwds': xlabel_kwds,
            'ylabel_kwds': ylabel_kwds,
            'watermark_kwds': watermark_kwds,
            'legend_kwds': legend_kwds,
            'xtick_kwds': xtick_kwds,
            'ytick_kwds': ytick_kwds,
            'annotation_kwds': annotation_kwds,
            'delays_kwds': delays_kwds,
            'xlim0s': self.xlim0s,
            'xlim1s': self.xlim1s,
            'ylim0s': self.ylim0s,
            'ylim1s': self.ylim1s
        }
        ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        self.data = {self.subplot: ds}

    def __str__(self):
        return ' '.join(
            f'<Subplot {subplot}> {repr(ds)}\n\n'
            for subplot, ds in self.data.items()
        )

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        self_copy = deepcopy(self)
        subplots = set(self.data) | set(other.data)
        for subplot in subplots:
            self_ds = self.data.get(subplot)
            other_ds = other.data.get(subplot)

            if self_ds is not None and other_ds is not None:
                has_same_items = len(
                    set(self_ds['item'].values) ^
                    set(other_ds['item'].values)
                ) == 0
                if has_same_items:
                    other_ds = other_ds.copy()
                    other_ds['item'] = (
                        other_ds['item'] + self_ds['item'].max())
                self_copy.data[subplot] = xr.combine_by_coords([
                    self_ds, other_ds])
            elif self_ds is not None:
                self_copy.data[subplot] = self_ds
            elif other_ds is not None:
                self_copy.data[subplot] = other_ds
        return self_copy

    def __rmul__(self, other):
        return self.__mul__(self, other)


class Array(Data):

    state_labels = param.ClassSelector(class_=(Iterable,))
    inline_labels = param.ClassSelector(class_=(Iterable,))
    colorbar = param.Boolean(default=None)
    clabel = param.String(default=None)

    state_kwds = param.Dict(default=None)
    inline_kwds = param.Dict(default=None)
    colorbar_kwds = param.Dict(default=None)
    clabel_kwds = param.Dict(default=None)
    ctick_kwds = param.Dict(default=None)

    def __init__(self, xs, ys, **kwds):
        super().__init__(xs, ys, **kwds)

        ds = self.data[self.subplot]
        if self.state_labels is not None:
            ds['state_label'] = ('state', self.state_labels)

        if self.inline_labels is not None:
            ds['inline_label'] = ('state', self.inline_labels)

        state_kwds = config._load('state_kwds', self.state_kwds)
        inline_kwds = config._load('inline_kwds', self.inline_kwds)
        colorbar_kwds = config._load(
            'colorbar_kwds', self.colorbar_kwds, show=self.colorbar)
        clabel_kwds = config._load(
            'clabel_kwds', self.clabel_kwds, clabel=self.clabel)
        ctick_kwds = config._load('ctick_kwds', self.ctick_kwds)
        ds.attrs.update({
            'state_kwds': state_kwds,
            'inline_kwds': inline_kwds,
            'colorbar_kwds': colorbar_kwds,
            'clabel_kwds': clabel_kwds,
            'ctick_kwds': ctick_kwds
        })
