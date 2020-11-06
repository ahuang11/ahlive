import operator
import warnings
from copy import copy, deepcopy
from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from . import easing, animation, util


DIMS = {
    'vars': ('item', 'state'),
    'refs': ('ref_item', 'state')
}


class Data(easing.Easing, animation.Animation):

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
    ref_plot_kwds = param.Dict()
    ref_inline_kwds = param.Dict()
    rowcol = param.NumericTuple(default=(1, 1), length=2)

    _parameters = []
    data = {}

    def __init__(self, xs, ys, **kwds):
        self._parameters = [key for key in dir(self)
                            if not key.startswith('_')]
        self._num_states = len(xs)

        input_vars = {
            key: kwds.pop(key) for key in list(kwds)
            if key not in self._parameters}
        super().__init__(**kwds)
        input_vars = self._amend_input_vars(input_vars)
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
            if key_and_s in self._parameters and key_and_s != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_and_s}!')
                expected_key = key_and_s
            elif key_strip in self._parameters and key_strip != key:
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
        label = self.label or ''

        data_vars = {'x': xs, 'y': ys}
        data_vars.update({
            key: val for key, val in input_vars.items()
            if val is not None
        })
        for var in list(data_vars.keys()):
            val = np.array(data_vars.pop(var))
            if util.is_scalar(val):
                val = [val] * self._num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = DIMS['vars'], val
        data_vars['label'] = 'item', [label]
        data_vars['chart'] = 'item', [chart]
        return data_vars

    def _load_coords(self):
        coords = {
            'item': [1],
            'state': np.arange(1, self._num_states + 1),
            'root': ('state', [1] * self._num_states)
        }
        return coords

    def _load_attrs(self):
        kwds_parameters = {
            param: getattr(self, param) for param in self._parameters
            if param.endswith('kwds')}

        attrs = {}
        for key, val in kwds_parameters.items():
            attrs[key] = getattr(self, key) or {}
            if key == 'chart_kwds':
                attrs[key]['base_chart'] = self.chart
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
                joined_ds = xr.combine_by_coords([self_ds, other_ds])
                joined_ds['state_label'] = (
                    joined_ds['state_label'].isel(item=0))
                self_copy.data[rowcol] = joined_ds
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
        return self + other

    def __sub__(self, other):
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
                    other_ds['state'] = (
                        other_ds['state'] + self_ds['state'].max())
                joined_ds = xr.concat(
                    [self_ds, other_ds], 'state'
                ).map(util.transpose, keep_attrs=True)
                joined_ds['label'] = joined_ds['label'].max('state')
                joined_ds['chart'] = joined_ds['chart'].max('state')
                self_copy._num_states = len(joined_ds['state'])
                self_copy.data[rowcol] = joined_ds
                for param in self._parameters:
                    self_param = getattr(self, param)
                    other_param = getattr(other, param)
                    if self_param is None and other_param is not None:
                        setattr(self_copy, param, other_param)
            elif self_ds is not None:
                self_copy.data[rowcol] = self_ds
            elif other_ds is not None:
                self_copy.data[rowcol] = other_ds
        return self_copy

    def __rsub__(self, other):
        return self - other

    def cols(self, ncols):
        self_copy = deepcopy(self)
        for iplot, rowcol in enumerate(list(self_copy.data)):
            row = (iplot) // ncols + 1
            col = (iplot) % ncols + 1
            self_copy.data[(row, col)] = self_copy.data.pop(rowcol)
        return self_copy


class Array(Data):

    xs = param.ClassSelector(class_=(Iterable,))
    ys = param.ClassSelector(class_=(Iterable,))

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
            ds['inline_label'] = ('state', self.inline_labels)

        ds.attrs.update({
            'state_kwds': self.state_kwds or {},
            'inline_kwds': self.inline_kwds or {},
            'colorbar_kwds': self.colorbar_kwds or {},
            'clabel_kwds': self.clabel_kwds or {},
            'ctick_kwds': self.ctick_kwds or {}
        })

    def add_annotations(self, xs=None, ys=None, state_labels=None,
                        inline_labels=None, condition=None,
                        annotations=None, delays=None, rowcols=None):
        args = (xs, ys, state_labels, inline_labels, condition)
        args_none = sum([1 for arg in args if arg is None])
        if args_none == len(args):
            raise ValueError(
                'Must supply either xs, ys, state_labels, '
                'inline_labels, or condition!')
        elif args_none != len(args) - 1:
            raise ValueError(
                'Must supply only one of xs, ys, state_labels, '
                'inline_labels, or condition!')

        if delays is None and annotations is None:
            raise ValueError(
                'Must supply at least annotations or delays!')

        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            if xs is not None:
                condition = ds['x'].isin(xs)
            elif ys is not None:
                condition = ds['y'].isin(ys)
            elif state_labels is not None:
                condition = ds['state_label'].isin(state_labels)
            elif inline_labels is not None:
                condition = ds['inline_label'].isin(inline_labels)

            if annotations is not None:
                if 'annotation' not in ds:
                    ds['annotation'] = (
                        DIMS['vars'],
                        np.full((len(ds['item']), self._num_states), '')
                    )
                if isinstance(annotations, str):
                    if annotations in ds.data_vars:
                        annotations = ds[annotations]
                ds['annotation'] = xr.where(
                    condition, annotations, ds['annotation']
                ).transpose(DIMS['vars'])

            if delays is not None:
                if 'delay' not in ds:
                    ds['delay'] = (
                        'state', np.repeat(0, len(ds['state'])))
                ds['delay'] = xr.where(
                    condition, delays, ds['delay']
                )
                if 'item' in ds['delay'].dims:
                    ds['delay'] = ds['delay'].max('item')
            self_copy.data[rowcol] = ds
        return self_copy


    def add_references(self, x0s=None, x1s=None, y0s=None, y1s=None,
                       label=None, inline_labels=None, inline_loc=None,
                       rowcols=None):
        args = {'ref_x0': x0s, 'ref_x1': x1s, 'ref_y0': y0s, 'ref_y1': y1s}
        has_args = {key: val is not None for key, val in args.items()}
        if not any(has_args.values()):
            raise ValueError('Must provide either x0s, x1s, y0s, y1s!')

        has_x0 = has_args['ref_x0']
        has_x1 = has_args['ref_x1']
        has_y0 = has_args['ref_y0']
        has_y1 = has_args['ref_y1']
        if has_x0 and has_x1 and has_y0 and has_y1:
            chart = 'rectangle'
            loc_axis = 'x'
        elif has_x0 and has_x1:
            chart = 'axvspan'
            loc_axis = 'y'
        elif has_y0 and has_y1:
            chart = 'axhspan'
            loc_axis = 'x'
        elif has_x0:
            chart = 'axvline'
            loc_axis = 'y'
        elif has_y0:
            chart = 'axhline'
            loc_axis = 'x'
        else:
            raise NotImplementedError()

        label = label or ''

        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            ref_vars = [
                var for var in ds.data_vars if 'ref_item' in ds[var].dims]
            base_ds = ds.drop_vars(ref_vars, errors='ignore')
            if len(ref_vars) > 0:
                past_ds = ds[ref_vars]
            else:
                past_ds = None

            for key, arg in args.items():
                if isinstance(arg, str):
                    axis = key[4]
                    arg = getattr(ds[axis], arg)('item')
                if util.is_scalar(arg):
                    arg = [arg] * self._num_states
                ds[key] = (DIMS['refs'], np.array(arg).reshape(1, -1))

            ds['ref_label'] = 'ref_item', [label]
            ds['ref_chart'] = 'ref_item', [chart]
            if inline_labels is None and inline_labels != False:
                if chart == 'axvspan':
                    inline_labels = ds['ref_x0'].isel(ref_item=[0])
                if chart == 'axhspan':
                    inline_labels = ds['ref_y0'].isel(ref_item=[0])
                if chart == 'axvline':
                    inline_labels = ds['ref_x0'].isel(ref_item=[0])
                elif chart == 'axhline':
                    inline_labels = ds['ref_y0'].isel(ref_item=[0])

            if inline_labels is not None:
                if isinstance(inline_labels, str):
                    if inline_labels in ds.data_vars:
                        inline_labels = ds[inline_labels].isel(item=[0])
                ds['ref_inline_label'] = DIMS['refs'], inline_labels

                if inline_loc is None:
                    if chart in ['rectangle', 'axvspan', 'axhspan']:
                        inline_loc = 'center'
                    elif chart == 'axvline':
                        inline_loc = 'bottom'
                    elif chart == 'axhline':
                        inline_loc = 'left'

                if isinstance(inline_loc, str):
                    if inline_loc == 'left':
                        inline_loc = [base_ds['x'].values.min()]
                    elif inline_loc == 'right':
                        inline_loc = [base_ds['x'].values.max()]
                    elif inline_loc == 'bottom':
                        inline_loc = [base_ds['y'].values.min()]
                    elif inline_loc == 'top':
                        inline_loc = [base_ds['y'].values.max()]
                    elif inline_loc == 'center':
                        inline_loc = [base_ds[loc_axis].values.mean()]

                ds['ref_inline_loc'] = 'ref_item', inline_loc

            if past_ds is not None:
                ds = xr.concat([past_ds, ds[ref_vars]], 'ref_item')
            else:
                ds = ds.drop_vars(base_ds.data_vars)
            ds = xr.merge([base_ds, ds], combine_attrs='override')
            self_copy.data[rowcol] = ds
        return self_copy


class DataFrame(Array):

    df = param.DataFrame()

    join = param.ObjectSelector(
        default='overlay', objects=['overlay', 'layout'])

    def __init__(self, df, xs, ys, label, **kwds):
        join = kwds.pop('join', 'overlay')

        arrays = []
        for label, df_item in df.groupby(label):
            kwds_updated = kwds.copy()

            kwds_updated.update({
                key: df_item[val].values
                if not isinstance(val, dict) and val in df_item else val
                for key, val in kwds.items()})
            if 'xlabel' not in kwds_updated:
                kwds_updated['xlabel'] = xs
            if 'ylabel' not in kwds_updated:
                kwds_updated['ylabel'] = ys
            if 'title' not in kwds_updated and join == 'layout':
                kwds_updated['title'] = label
            super().__init__(
                df_item[xs], df_item[ys], label=label, **kwds_updated)
            arrays.append(deepcopy(self))
        self.data = getattr(util, join)(arrays).data
