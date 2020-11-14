import operator
import warnings
from copy import copy, deepcopy
from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from .easing import Easing
from .animation import Animation
from .util import is_scalar, transpose, srange
from .join import _get_rowcols, layout, cascade, overlay


CHARTS = {
    'vars': ['scatter', 'line', 'barh', 'bar', 'plot'],
    'refs': ['axvspan', 'axhspan', 'axvline', 'axhline']
}
DIMS = {
    'vars': ('item', 'state'),
    'refs': ('REF_item', 'state')
}
VARS = {
    'item': ['label', 'chart', 'REF_label', 'REF_chart', 'state_label']
}
NULL_VALS = [(), {}, [], None, '']


class Data(Easing, Animation):

    chart = param.ObjectSelector(objects=CHARTS['vars'])
    style = param.ObjectSelector(objects=['graph', 'minimal', 'bare'])
    label = param.String()
    group = param.String()

    state_labels = param.ClassSelector(class_=(Iterable,))
    inline_labels = param.ClassSelector(class_=(Iterable,))

    xlim0s = param.ClassSelector(class_=(Iterable, int, float))
    xlim1s = param.ClassSelector(class_=(Iterable, int, float))
    ylim0s = param.ClassSelector(class_=(Iterable, int, float))
    ylim1s = param.ClassSelector(class_=(Iterable, int, float))

    title = param.String(default='')
    xlabel = param.String()
    ylabel = param.String()

    legend = param.Boolean(default=True)
    grid = param.Boolean(default=True)
    batch = param.Boolean(default=False)
    hooks = param.HookList()

    crs = param.String()
    projection = param.String()
    borders = param.Boolean(default=None)
    coastline = param.Boolean(default=True)
    land = param.Boolean(default=None)
    lakes = param.Boolean(default=None)
    ocean = param.Boolean(default=None)
    rivers = param.Boolean(default=None)
    states = param.Boolean(default=None)
    worldwide = param.Boolean(default=None)

    axes_kwds = param.Dict()
    plot_kwds = param.Dict()
    chart_kwds = param.Dict()
    state_kwds = param.Dict()
    inline_kwds = param.Dict()

    grid_kwds = param.Dict()
    margins_kwds = param.Dict()
    title_kwds = param.Dict()
    xlabel_kwds = param.Dict()
    ylabel_kwds = param.Dict()
    legend_kwds = param.Dict()
    xtick_kwds = param.Dict()
    ytick_kwds = param.Dict()

    crs_kwds = param.Dict()
    projection_kwds = param.Dict()
    borders_kwds = param.Dict()
    coastline_kwds = param.Dict()
    land_kwds = param.Dict()
    lakes_kwds = param.Dict()
    ocean_kwds = param.Dict()
    rivers_kwds = param.Dict()
    states_kwds = param.Dict()

    annotation_kwds = param.Dict()
    REF_plot_kwds = param.Dict()
    REF_inline_kwds = param.Dict()
    rowcol = param.NumericTuple(default=(1, 1), length=2)

    _parameters = []
    data = {}

    def __init__(self, num_states, **kwds):
        self._parameters = [
            key for key in dir(self) if not key.startswith('_')]
        self._num_states = num_states
        input_vars = {
            key: kwds.pop(key) for key in list(kwds)
            if key not in self._parameters}
        super().__init__(**kwds)
        input_vars = self._amend_input_vars(input_vars)
        data_vars = self._load_data_vars(input_vars)
        coords = self._load_coords()
        attrs = self._load_attrs()
        ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        self.data = {self.rowcol: ds}

    def _adapt_input(self, val, reshape=True):
        val = np.array(val)
        if is_scalar(val):
            val = np.repeat(val, self._num_states)
        if reshape:
            val = val.reshape(1, -1)
        return val

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

    def _load_data_vars(self, input_vars):
        if self.chart is None:
            chart = 'scatter' if self._num_states <= 5 else 'line'
        else:
            chart = self.chart
        label = self.label or ''
        group = self.group or ''
        batch = self.batch or ''

        data_vars = {
            key: val for key, val in input_vars.items()
            if val is not None}
        for var in list(data_vars.keys()):
            val = data_vars.pop(var)
            val = self._adapt_input(val)
            dims = DIMS['refs'] if var.startswith('ref') else DIMS['vars']
            data_vars[var] = dims, val

        if self.state_labels is not None:
            state_labels = self._adapt_input(
                self.state_labels, reshape=False)
            data_vars['state_label'] = ('state', state_labels)

        if self.inline_labels is not None:
            inline_labels = self._adapt_input(
                self.inline_labels, reshape=False)
            data_vars['inline_label'] = ('state', inline_labels)

        data_vars['chart'] = 'item', [chart]
        data_vars['label'] = 'item', [label]
        data_vars['group'] = 'item', [group]
        data_vars['batch'] = 'item', [batch]
        return data_vars

    def _load_coords(self):
        coords = {
            'item': [1],
            'state': srange(self._num_states)
        }
        return coords

    def _load_attrs(self):
        kwds_parameters = {
            param: getattr(self, param) for param in self._parameters
            if param.endswith('kwds')}

        attrs = {}
        for key, val in kwds_parameters.items():
            attrs[key] = val or {}
            if key == 'chart_kwds':
                attrs[key]['base_chart'] = self.chart
            elif key == 'grid_kwds':
                attrs[key]['grid'] = self.grid
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
            elif key == 'crs_kwds':
                attrs[key]['crs'] = self.crs
            elif key == 'projection_kwds':
                attrs[key]['projection'] = self.projection
            elif key == 'borders_kwds':
                attrs[key]['borders'] = self.borders
            elif key == 'coastline_kwds':
                attrs[key]['coastline'] = self.coastline
            elif key == 'land_kwds':
                attrs[key]['land'] = self.land
            elif key == 'lakes_kwds':
                attrs[key]['lakes'] = self.lakes
            elif key == 'ocean_kwds':
                attrs[key]['ocean'] = self.ocean
            elif key == 'rivers_kwds':
                attrs[key]['rivers'] = self.rivers
            elif key == 'states_kwds':
                attrs[key]['states'] = self.states

        attrs['finalize_kwds'] = {
            'xlim0s': self.xlim0s,
            'xlim1s': self.xlim1s,
            'ylim0s': self.ylim0s,
            'ylim1s': self.ylim1s,
            'worldwide': self.worldwide,
            'hooks': self.hooks
        }
        return attrs

    def _propagate_params(self, self_copy, other):
        for param in self._parameters:
            self_param = getattr(self, param)
            other_param = getattr(other, param)

            try:
                self_null = self_param in NULL_VALS
            except ValueError:
                self_null = False

            try:
                other_null = other_param in NULL_VALS
            except ValueError:
                other_null = False

            if self_null and not other_null:
                setattr(self_copy, param, other_param)
        return self_copy

    @staticmethod
    def _shift_items(self_ds, other_ds):
        for item in ['item', 'REF_item']:
            if not (item in self_ds.dims and item in other_ds.dims):
                continue
            has_same_items = len(
                set(self_ds[item].values) |
                set(other_ds[item].values)
            ) > 0
            if has_same_items:
                other_ds[item] = other_ds[item].copy()
                other_ds[item] = (
                    other_ds[item] + self_ds[item].max())
        return other_ds

    @staticmethod
    def _drop_state(joined_ds):
        for var in VARS['item']:
            if var in joined_ds:
                if 'state' in joined_ds[var].dims:
                    joined_ds[var] = joined_ds[var].max('state')
        return joined_ds

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
        rowcols = _get_rowcols([self, other])
        for rowcol in rowcols:
            self_ds = self.data.get(rowcol)
            other_ds = other.data.get(rowcol)

            if self_ds is None:
                self_copy.data[rowcol] = other_ds
            elif other_ds is None:
                self_copy.data[rowcol] = self_ds
            else:
                other_ds = self._shift_items(self_ds, other_ds)
                joined_ds = xr.combine_by_coords(
                    [self_ds, other_ds], combine_attrs='override')
                joined_ds = self._drop_state(joined_ds)
                self_copy.data[rowcol] = joined_ds
        self_copy = self._propagate_params(self_copy, other)
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
        self_copy = self._propagate_params(self_copy, other)
        return self_copy

    def __truediv__(self, other):
        return self / other

    def __add__(self, other):
        self_copy = deepcopy(self)
        self_cols = max(self_copy.data, key=operator.itemgetter(1))[1]
        for rowcol, ds in other.data.items():
            if rowcol[0] <= self_cols:
                rowcol_shifted = (rowcol[0], rowcol[1] + self_cols)
                self_copy.data[rowcol_shifted] = ds
            else:
                self_copy.data[rowcol] = ds
        self_copy = self._propagate_params(self_copy, other)
        return self_copy

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        self_copy = deepcopy(self)
        rowcols = _get_rowcols([self, other])
        for rowcol in rowcols:
            self_ds = self.data.get(rowcol)
            other_ds = other.data.get(rowcol)

            if self_ds is None:
                self_copy.data[rowcol] = other_ds
            elif other_ds is None:
                self_copy.data[rowcol] = self_ds
            else:
                other_ds = self._shift_items(self_ds, other_ds)
                joined_ds = xr.concat(
                    [self_ds, other_ds], 'state'
                ).map(transpose, keep_attrs=True)
                joined_ds = self._drop_state(joined_ds)
                self_copy.data[rowcol] = joined_ds
                self_copy._num_states = len(joined_ds['state'])
        self_copy = self._propagate_params(self_copy, other)
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

    colorbar = param.Boolean(default=False)
    clabel = param.String()

    colorbar_kwds = param.Dict()
    clabel_kwds = param.Dict()
    ctick_kwds = param.Dict()

    def __init__(self, xs, ys, **kwds):
        num_states = len(xs)
        super().__init__(num_states, **kwds)
        ds = self.data[self.rowcol]
        ds['x'] = DIMS['vars'], self._adapt_input(xs)
        ds['y'] = DIMS['vars'], self._adapt_input(ys)
        ds.attrs.update({
            'colorbar_kwds': self.colorbar_kwds or {},
            'clabel_kwds': self.clabel_kwds or {},
            'ctick_kwds': self.ctick_kwds or {}
        })

    def add_annotations(self, annotations=None, delays=None, condition=None,
                        xs=None, ys=None, state_labels=None,
                        inline_labels=None, rowcols=None):
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
            else:
                condition = np.array(condition)

            if annotations is not None:
                if 'annotation' not in ds:
                    ds['annotation'] = (
                        DIMS['vars'],
                        np.full((len(ds['item']), self_copy._num_states), '')
                    )
                if isinstance(annotations, str):
                    if annotations in ds.data_vars:
                        annotations = ds[annotations].values.astype(str)
                ds['annotation'] = xr.where(
                    condition, annotations, ds['annotation']
                ).transpose(*DIMS['vars'])

            if delays is not None:
                if 'delay' not in ds:
                    ds['delay'] = 'state', self._adapt_input(
                        np.zeros_like(ds['state']), reshape=False)
                ds['delay'] = xr.where(
                    condition, delays, ds['delay']
                )
                if 'item' in ds['delay'].dims:
                    ds['delay'] = ds['delay'].max('item')

            self_copy.data[rowcol] = ds
        return self_copy


class DataFrame(Array):

    df = param.DataFrame()

    join = param.ObjectSelector(
        default='overlay', objects=['overlay', 'layout', 'cascade'])

    def __init__(self, df, xs, ys, **kwds):
        join = kwds.pop('join', 'overlay')

        group_key = kwds.pop('group', None)
        label_key = kwds.pop('label', None)

        df_cols = df.columns
        for key in [group_key, label_key]:
            if key and key not in df_cols:
                raise ValueError(f'{key} not found in {df_cols}!')

        arrays = []
        for group, group_df in self._groupby_key(df, group_key):
            for label, label_df in self._groupby_key(group_df, label_key):
                kwds_updated = kwds.copy()
                kwds_updated.update({
                    key: label_df[val].values
                    if not isinstance(val, dict) and val in label_df else val
                    for key, val in kwds.items()})

                if 'xlabel' not in kwds_updated:
                    kwds_updated['xlabel'] = xs
                if 'ylabel' not in kwds_updated:
                    kwds_updated['ylabel'] = ys
                if 'clabel' not in kwds_updated and 'c' in kwds:
                    kwds_updated['clabel'] = kwds['c']
                if 'title' not in kwds_updated and join == 'layout':
                    kwds_updated['title'] = label

                super().__init__(
                    label_df[xs], label_df[ys],
                    group=group, label=label,
                    **kwds_updated
                )
                arrays.append(deepcopy(self))

        if join == 'overlay':
            self.data = overlay(arrays, quick=True).data
        elif join == 'layout':
            self.data = layout(arrays, quick=True).data
        elif join == 'cascade':
            self.data = cascade(arrays, quick=True).data


class Reference(Data):

    chart = param.ObjectSelector(objects=CHARTS['refs'])

    x0s = param.ClassSelector(class_=(Iterable,))
    x1s = param.ClassSelector(class_=(Iterable,))
    y0s = param.ClassSelector(class_=(Iterable,))
    y1s = param.ClassSelector(class_=(Iterable,))
    inline_loc = param.ClassSelector(class_=(Iterable,))

    def __init__(self, x0s=None, x1s=None, y0s=None, y1s=None, **kwds):
        args = {
            'REF_x0': x0s,
            'REF_x1': x1s,
            'REF_y0': y0s,
            'REF_y1': y1s
        }

        has_args = {key: val is not None for key, val in args.items()}
        if not any(has_args.values()):
            raise ValueError('Must provide either x0s, x1s, y0s, y1s!')
        elif sum(has_args.values()) > 2:
            raise ValueError('At most two values can be provided!')

        for arg in args.values():
            if arg is not None:
                num_states = len(np.atleast_1d(arg))
                break

        has_x0 = has_args['REF_x0']
        has_x1 = has_args['REF_x1']
        has_y0 = has_args['REF_y0']
        has_y1 = has_args['REF_y1']
        if has_x0 and has_x1:
            kwds['chart'] = 'axvspan'
            loc_axis = 'y'
        elif has_y0 and has_y1:
            kwds['chart'] = 'axhspan'
            loc_axis = 'x'
        elif has_x0:
            kwds['chart'] = 'axvline'
            loc_axis = 'y'
        elif has_y0:
            kwds['chart'] = 'axhline'
            loc_axis = 'x'
        else:
            raise NotImplementedError()

        super().__init__(num_states, **kwds)
        ds = self.data[self.rowcol]
        ds = ds.rename({
            var: f'REF_{var}' for var in list(ds.data_vars) + ['item']
        })

        label = self.label or ''

        for key, arg in args.items():
            if isinstance(arg, str):
                axis = key[4]
                arg = getattr(ds[axis], arg)('item')
            arg = self._adapt_input(arg)
            ds[key] = (DIMS['refs'], np.array(arg).reshape(1, -1))

        inline_labels = self.inline_labels
        if isinstance(inline_labels, str):
            if inline_labels in ds.data_vars:
                inline_labels = ds[inline_labels].isel(item=[0])

        if inline_labels is not None:
            inline_labels = self._adapt_input(inline_labels)
            ds['REF_inline_label'] = DIMS['refs'], inline_labels

            inline_loc = self.inline_loc
            if inline_loc is None:
                raise ValueError(
                    'Must provide an inline location '
                    'if inline_labels is not None!')
                ds['REF_inline_loc'] = 'REF_item', [inline_loc]

        self.data[self.rowcol] = ds
