import operator
import warnings
from copy import copy, deepcopy
from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from . import easing, animation, util



class Data(easing.Easing, animation.Animation):

    xs = param.ClassSelector(class_=(Iterable,))
    ys = param.ClassSelector(class_=(Iterable,))

    chart = param.ObjectSelector(
        objects=['scatter', 'line', 'barh', 'bar', 'plot'])
    style = param.ObjectSelector(objects=['graph', 'minimal', 'bare'])
    label = param.String()

    xlim0s = param.ClassSelector(class_=(Iterable, int, float))
    xlim1s = param.ClassSelector(class_=(Iterable, int, float))
    ylim0s = param.ClassSelector(class_=(Iterable, int, float))
    ylim1s = param.ClassSelector(class_=(Iterable, int, float))

    title = param.String(default='')
    xlabel = param.String()
    ylabel = param.String()
    legend = param.Boolean(default=True)
    hooks = param.HookList(default=None)

    axes_kwds = param.Dict()
    plot_kwds = param.Dict()
    chart_kwds = param.Dict()
    grid_kwds = param.Dict()
    margins_kwds = param.Dict()
    title_kwds = param.Dict()
    xlabel_kwds = param.Dict()
    ylabel_kwds = param.Dict()
    legend_kwds = param.Dict()
    xtick_kwds = param.Dict()
    ytick_kwds = param.Dict()
    annotation_kwds = param.Dict()
    rowcol = param.NumericTuple(default=(1, 1), length=2)

    _inputs = []
    _num_states = 0
    data = {}

    def __init__(self, xs, ys, **kwds):
        self._inputs = [key for key in dir(self) if not key.startswith('_')]
        self._num_states = len(xs)

        input_vars = {
            key: kwds.pop(key) for key in list(kwds)
            if key not in self._inputs}
        super().__init__(**kwds)
        input_vars = self._amend_input_vars(input_vars, )
        data_vars = self._load_data_vars(xs, ys, input_vars)
        coords = self._load_coords()
        attrs = self._load_attrs()
        ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        self.data = {self.rowcol: ds}

    def _amend_input_vars(self, input_vars):
        for key in list(input_vars.keys()):
            key_and_s = key + 's'
            key_strip = key.rstrip('s')
            expected_key = None
            if key_and_s in self._inputs and key_and_s != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_and_s}!')
                expected_key = key_and_s
            elif key_strip in self._inputs and key_strip != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_strip}!')
                expected_key = key_strip
            if expected_key:
                setattr(self, expected_key, input_vars.pop(key))
        return input_vars

    def _load_data_vars(self, xs, ys, input_vars):
        if self.chart is None:
            chart = 'scatter' if self._num_states <= 5 else 'line'
        else:
            chart = self.chart
        label = '' if self.label is None else self.label

        data_vars = {'x': xs, 'y': ys}
        data_vars.update({
            key: val for key, val in input_vars.items()
            if val is not None
        })
        for var in list(data_vars.keys()):
            val = np.array(data_vars.pop(var))
            dims = ('item', 'state')
            if util.is_scalar(val):
                val = [val] * self._num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = dims, val
        data_vars['label'] = 'item', [label]
        data_vars['chart'] = 'item', [chart]
        return data_vars

    def _load_coords(self):
        coords = {
            'item': [1],
            'state': np.arange(self._num_states),
            'root': ('state', [1] * self._num_states)
        }
        return coords

    def _load_attrs(self):
        kwds_inputs = {
            param: getattr(self, param) for param in self._inputs
            if param.endswith('kwds')}

        attrs = {}
        for key, val in kwds_inputs.items():
            attrs[key] = getattr(self, key) or {}
            if key == 'chart_kwds':
                attrs[key]['chart'] = self.chart
            elif key == 'axes_kwds':
                attrs[key]['style'] = self.style
            elif key == 'title_kwds':
                attrs[key]['label'] = self.title
            elif key == 'xlabel_kwds':
                attrs[key]['xlabel'] = self.xlabel
            elif key == 'ylabel_kwds':
                attrs[key]['ylabel'] = self.ylabel
            elif key == 'legend_kwds':
                attrs[key]['show'] = self.legend
        attrs.update({
            'xlim0s': self.xlim0s,
            'xlim1s': self.xlim1s,
            'ylim0s': self.ylim0s,
            'ylim1s': self.ylim1s
        })
        return attrs

    def __str__(self):
        strings = []
        for rowcol, ds in self.data.items():
            dims = ', '.join(f'{key}: {val}' for key, val in ds.dims.items())
            data = repr(ds.data_vars)
            strings.append(
                f'Subplot:{" ":9}{rowcol}\n'
                f'Dimensions:{" ":6}({dims})\n'
                f'{data}\n\n'
            )
        return '<ahlive.Data>\n' + ''.join(strings)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        self_copy = deepcopy(self)
        rowcols = set(self.data) | set(other.data)
        for rowcol in rowcols:
            self_ds = self.data.get(rowcol)
            other_ds = other.data.get(rowcol)

            if self_ds is not None and other_ds is not None:
                has_same_items = len(
                    set(self_ds['item'].values) |
                    set(other_ds['item'].values)
                ) > 0
                if has_same_items:
                    other_ds['item'] = other_ds['item'].copy()
                    other_ds['item'] = (
                        other_ds['item'] + self_ds['item'].max())
                self_copy.data[rowcol] = xr.combine_by_coords([
                    self_ds, other_ds])
                self_copy.data[rowcol]['state_label'] = (
                    self_copy.data[rowcol]['state_label'].isel(item=0))
            elif self_ds is not None:
                self_copy.data[rowcol] = self_ds
            elif other_ds is not None:
                self_copy.data[rowcol] = other_ds
        return self_copy

    def __rmul__(self, other):
        return other * self

    def __floordiv__(self, other):
        self_copy = deepcopy(self)
        self_rows = max(self_copy.data)[0]
        for rowcol, ds in other.data.items():
            if rowcol[0] <= self_rows:
                rowcol_shifted = (rowcol[0] + self_rows, rowcol[1])
                self_copy.data[rowcol_shifted] = ds
            else:
                self_copy.data[rowcol] = ds
        return self_copy

    def __truediv__(self, other):
        return self.__floordiv__(other)

    def __add__(self, other):
        self_copy = deepcopy(self)
        self_cols = max(self_copy.data, key=operator.itemgetter(1))[1]
        for rowcol, ds in other.data.items():
            if rowcol[0] <= self_cols:
                rowcol_shifted = (rowcol[0], rowcol[1] + self_cols)
                self_copy.data[rowcol_shifted] = ds
            else:
                self_copy.data[rowcol] = ds
        return self_copy

    def __radd__(self, other):
        return other + self

    def cols(self, ncols):
        self_copy = deepcopy(self)
        for iplot, rowcol in enumerate(list(self_copy.data)):
            row = (iplot) // ncols + 1
            col = (iplot) % ncols + 1
            self_copy.data[(row, col)] = self_copy.data.pop(rowcol)
        return self_copy

    def interpolate(self):
        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            ds = ds.reset_coords().map(super().interpolate, keep_attrs=True)
            self_copy.data[rowcol] = ds.set_coords('root')
        self_copy._num_states = len(ds['root'])
        return self_copy


class Array(Data):

    state_labels = param.ClassSelector(class_=(Iterable,))
    inline_labels = param.ClassSelector(class_=(Iterable,))
    colorbar = param.Boolean(default=False)
    clabel = param.String()

    state_kwds = param.Dict()
    inline_kwds = param.Dict()
    colorbar_kwds = param.Dict()
    clabel_kwds = param.Dict()
    ctick_kwds = param.Dict()

    def __init__(self, xs, ys, **kwds):
        super().__init__(xs, ys, **kwds)

        ds = self.data[self.rowcol]
        if self.state_labels is not None:
            ds['state_label'] = ('state', self.state_labels)

        if self.inline_labels is not None:
            ds['inline_label'] = (('item', 'state'), self.inline_labels)

        ds.attrs.update({
            'state_kwds': self.state_kwds or {},
            'inline_kwds': self.inline_kwds or {},
            'colorbar_kwds': self.colorbar_kwds or {},
            'clabel_kwds': self.clabel_kwds or {},
            'ctick_kwds': self.ctick_kwds or {}
        })


class DataFrame(Array):

    df = param.DataFrame()

    join = param.ObjectSelector(
        default='overlay', objects=['overlay', 'layout'])

    def __init__(self, df, xs, ys, label, **kwds):
        join = kwds.pop('join', 'overlay')

        arrays = []
        self._inputs = [key for key in dir(self) if not key.startswith('_')]
        for label, df_item in df.groupby(label):
            kwds_updated = kwds.copy()
            kwds_updated.update({
                key: df_item[val] if val in df_item else val
                for key, val in kwds.items()})
            if 'xlabel' not in kwds_updated:
                kwds_updated['xlabel'] = xs
            if 'ylabel' not in kwds_updated:
                kwds_updated['ylabel'] = ys
            if 'title' not in kwds_updated and join == 'layout':
                kwds_updated['title'] = label
            arrays.append(
                Array(df_item[xs], df_item[ys], label=label, **kwds_updated))
        self._num_states = arrays[0]._num_states
        self.data = getattr(util, join)(arrays).data
