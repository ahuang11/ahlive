import operator
import warnings
from copy import copy, deepcopy
from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from .easing import Easing
from .animation import Animation, CHARTS, DIMS, LIMS, VARS
from .util import is_scalar, is_datetime, srange, to_1d, to_scalar, ffill
from .join import _get_rowcols, _combine, layout, cascade, overlay


NULL_VALS = [(), {}, [], None, '']


class Data(Easing, Animation):

    chart = param.ObjectSelector(objects=CHARTS['base'])
    chart_type = param.ObjectSelector(objects=CHARTS['type'])
    style = param.ObjectSelector(objects=['graph', 'minimal', 'bare'])
    label = param.String()
    group = param.String()

    state_labels = param.ClassSelector(class_=(Iterable,))
    inline_labels = param.ClassSelector(class_=(Iterable,))

    xlims = param.ClassSelector(class_=(Iterable))
    ylims = param.ClassSelector(class_=(Iterable))
    xlim0s = param.ClassSelector(class_=(Iterable, int, float))
    xlim1s = param.ClassSelector(class_=(Iterable, int, float))
    ylim0s = param.ClassSelector(class_=(Iterable, int, float))
    ylim1s = param.ClassSelector(class_=(Iterable, int, float))

    title = param.String()
    subtitle = param.String()
    xlabel = param.String()
    ylabel = param.String()
    note = param.String()
    caption = param.String()

    xticks = param.ClassSelector(class_=(Iterable,))
    yticks = param.ClassSelector(class_=(Iterable,))

    legend = param.Boolean(default=True)
    grid = param.Boolean(default=True)
    hooks = param.HookList()

    crs = param.String()
    projection = param.String()
    central_lon = param.ClassSelector(class_=(Iterable, int, float))
    borders = param.Boolean(default=None)
    coastline = param.Boolean(default=True)
    land = param.Boolean(default=None)
    ocean = param.Boolean(default=None)
    lakes = param.Boolean(default=None)
    rivers = param.Boolean(default=None)
    states = param.Boolean(default=None)
    worldwide = param.Boolean(default=None)

    rowcol = param.NumericTuple(default=(1, 1), length=2)

    _parameters = []
    data = {}

    def __init__(self, num_states, **kwds):
        self._parameters = [
            key for key in dir(self) if not key.startswith('_')]
        input_vars = {
            key: kwds.pop(key) for key in list(kwds)
            if key not in self._parameters}
        super().__init__(**kwds)
        self.num_states = num_states
        input_vars = self._amend_input_vars(input_vars)
        data_vars = self._load_data_vars(input_vars)
        coords = self._load_coords()
        attrs = self._load_attrs()
        ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        self.data = {self.rowcol: ds}

    @property
    def num_states(self):
        return self._num_states

    @num_states.setter
    def num_states(self, num_states):
        self._num_states = num_states

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        for ds in data.values():
            self.num_states = len(ds['state'])
            break
        self._data = data

    def _adapt_input(self, val, reshape=True, shape=None):
        val = np.array(val)
        if is_scalar(val):
            val = np.repeat(val, self.num_states)
        if reshape:
            if shape is None:
                val = val.reshape(-1, self.num_states)
            else:
                val = val.reshape(-1, self.num_states, *shape)
        return val

    def _amend_input_vars(self, input_vars):
        for key in list(input_vars.keys()):
            key_and_s = key + 's'
            key_strip = key.rstrip('s')
            expected_key = None
            if key_and_s in self._parameters and key_and_s != key:
                warnings.warn(f'Replacing unexpected {key} as {key_and_s}!')
                expected_key = key_and_s
            elif key_strip in self._parameters and key_strip != key:
                warnings.warn(f'Replacing unexpected {key} as {key_strip}!')
                expected_key = key_strip
            if expected_key:
                setattr(self, expected_key, input_vars.pop(key))

        for lim in LIMS:
            if lim in input_vars:
                warnings.warn(f'Replacing unexpected {key} as {key_strip}!')
                if 'x' in lim and '0' in lim:
                    expected_key = 'xlim0s'
                elif 'x' in lim and '1' in lim:
                    expected_key = 'xlim1s'
                elif 'y' in lim and '0' in lim:
                    expected_key = 'ylim0s'
                elif 'y' in lim and '1' in lim:
                    expected_key = 'ylim1s'
                setattr(self, expected_key, input_vars.pop(key))

        return input_vars

    def _load_data_vars(self, input_vars):
        if self.chart is None:
            if self.num_states <= 5 or 's' in input_vars:
                chart = 'scatter'
            else:
                chart = 'line'
        else:
            chart = self.chart
        label = self.label or ''
        group = self.group or ''

        data_vars = {
            key: val for key, val in input_vars.items()
            if val is not None}
        for var in list(data_vars.keys()):
            val = data_vars.pop(var)
            val = self._adapt_input(val)
            dims = DIMS['refs'] if var.startswith('ref') else DIMS['base']
            data_vars[var] = dims, val

        if self.state_labels is not None:
            state_labels = self._adapt_input(
                self.state_labels, reshape=False)
            data_vars['state_label'] = ('state', state_labels)

        if self.inline_labels is not None:
            inline_labels = self._adapt_input(
                self.inline_labels)
            data_vars['inline_label'] = DIMS['base'], inline_labels

        data_vars['chart'] = 'item', [chart]
        data_vars['label'] = 'item', [label]
        data_vars['group'] = 'item', [group]
        return data_vars

    def _load_coords(self):
        coords = {
            'item': [1],
            'state': srange(self.num_states)
        }
        return coords

    def _load_attrs(self):
        for axis in ['x', 'y']:
            axis_lim = getattr(self, f'{axis}lims')
            if axis_lim is None:
                continue
            elif not isinstance(axis_lim, str) and len(axis_lim) != 2:
                raise ValueError(
                    f'`{axis_lim}` must be a string or tuple, got '
                    f'{axis_lim}; for moving limits, set `{axis}lim0s` '
                    f'and `{axis}lim1s` instead!')

        attrs = {}
        attrs['settings'] = {
            'xlims': self.xlims,
            'ylims': self.ylims,
            'xlim0s': self.xlim0s,
            'xlim1s': self.xlim1s,
            'ylim0s': self.ylim0s,
            'ylim1s': self.ylim1s,
            'worldwide': self.worldwide,
            'hooks': self.hooks
        }
        return attrs

    @staticmethod
    def _match_states(self_ds, other_ds):
        other_num_states = len(other_ds['state'])
        self_num_states = len(self_ds['state'])
        if other_num_states != self_num_states:
            warnings.warn(
                f'The latter dataset has {other_num_states} state(s) while '
                f'the former has {self_num_states} state(s); '
                f'reindexing the latter dataset to match the former.'
            )
            other_ds = other_ds.reindex(state=self_ds['state']).map(
                ffill, keep_attrs=True)
        return other_ds

    @staticmethod
    def _shift_items(self_ds, other_ds):
        for item in DIMS['item']:
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
        for var in VARS['stateless']:
            if var in joined_ds:
                if 'state' in joined_ds[var].dims:
                    joined_ds[var] = joined_ds[var].max('state')
        return joined_ds

    def _propagate_params(self, self_copy, other):
        for param in self._parameters:
            try:
                self_param = getattr(self, param)
                other_param = getattr(other, param)
            except AttributeError:
                continue

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

    def __getitem__(self, key):
        return self.data[key]

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
        other_copy = deepcopy(other)
        rowcols = _get_rowcols([self, other])
        data = {}
        for rowcol in rowcols:
            self_ds = self.data.get(rowcol)
            other_ds = other_copy.data.get(rowcol)
            other_ds = self._match_states(self_ds, other_ds)

            if self_ds is None:
                data[rowcol] = other_ds
            elif other_ds is None:
                data[rowcol] = self_ds
            else:
                other_ds = self._shift_items(self_ds, other_ds)
                joined_ds = _combine([self_ds, other_ds], method='merge')
                joined_ds = self._drop_state(joined_ds)
                data[rowcol] = joined_ds
        self_copy.data = data
        self_copy = self._propagate_params(self_copy, other_copy)
        return self_copy

    def __rmul__(self, other):
        return other * self

    def __floordiv__(self, other):
        self_copy = deepcopy(self)
        self_rows = max(self_copy.data)[0]
        data = {}
        for rowcol, ds in other.data.items():
            if rowcol[0] <= self_rows:
                rowcol_shifted = (rowcol[0] + self_rows, rowcol[1])
                data[rowcol_shifted] = ds
            else:
                data[rowcol] = ds
        self_copy.data = data
        self_copy = self._propagate_params(self_copy, other)
        return self_copy

    def __truediv__(self, other):
        return self / other

    def __add__(self, other):
        self_copy = deepcopy(self)
        other_copy = deepcopy(other)
        self_cols = max(self_copy.data, key=operator.itemgetter(1))[1]
        rowcols = _get_rowcols([self, other])
        data = {}
        for rowcol in rowcols:
            self_ds = self.data.get(rowcol)
            other_ds = other_copy.data.get(rowcol)
            other_ds = self._match_states(self_ds, other_ds)

            if rowcol[0] <= self_cols:
                rowcol_shifted = (rowcol[0], rowcol[1] + self_cols)
                data[rowcol_shifted] = other_ds
            else:
                data[rowcol] = other_ds

        self_copy.data.update(data)
        self_copy = self._propagate_params(self_copy, other)
        return self_copy

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        self_copy = deepcopy(self)
        rowcols = _get_rowcols([self, other])
        data = {}
        for rowcol in rowcols:
            self_ds = self.data.get(rowcol)
            other_ds = other.data.get(rowcol)

            if self_ds is None:
                data[rowcol] = other_ds
            elif other_ds is None:
                data[rowcol] = self_ds
            else:
                other_ds = self._shift_items(self_ds, other_ds)
                other_ds['state'] = other_ds['state'] + self_ds['state'].max()
                joined_ds = _combine(
                    [self_ds, other_ds], method='merge')
                joined_ds = self._drop_state(joined_ds)
                data[rowcol] = joined_ds
        self_copy.data = data
        self_copy = self._propagate_params(self_copy, other)
        return self_copy

    def __rsub__(self, other):
        return self - other

    def cols(self, ncols):
        self_copy = deepcopy(self)
        data = {}
        for iplot, rowcol in enumerate(self_copy.data.copy()):
            row = (iplot) // ncols + 1
            col = (iplot) % ncols + 1
            data[(row, col)] = self_copy.data.pop(rowcol)
        self_copy.data = data
        return self_copy

    def config(self, axes=None, plot=None, chart=None,
               state=None, inline=None, grid=None,
               title=None, subtitle=None,
               xlabel=None, ylabel=None,
               note=None, caption=None, legend=None,
               xticks=None, yticks=None,
               colorbar=None, clabel=None, cticks=None,
               grid_plot=None, grid_inline=None,
               ref_plot=None, ref_inline=None,
               remark_inline=None, remark=None,
               crs=None, projection=None,
               borders=None, coastline=None, land=None, ocean=None,
               lakes=None, rivers=None, states=None, margins=None,
               figure=None, animate=None, compute=None,
               suptitle=None, watermark=None,
               spacing=None, durations=None, frame=None):
        # TODO: find a better way to implement this as a class
        attrs = {}
        for key, val in locals().items():
            if key in ['self', 'attrs']:
                continue
            if val is None:
                attrs[key] = {}
            elif not isinstance(val, dict):
                raise ValueError(f'{key} value, {val} must be a dictionary!')
            else:
                attrs[key] = val

            if key == 'chart':
                attrs[key]['chart_type'] = self.chart_type
            elif key == 'grid':
                attrs[key]['grid'] = self.grid
            elif key == 'axes':
                attrs[key]['style'] = self.style
            elif key == 'title':
                attrs[key]['label'] = self.title
            elif key == 'subtitle':
                attrs[key]['label'] = self.subtitle
            elif key == 'xlabel':
                attrs[key]['xlabel'] = self.xlabel
            elif key == 'ylabel':
                attrs[key]['ylabel'] = self.ylabel
            elif key == 'note':
                attrs[key]['s'] = self.note
            elif key == 'caption':
                attrs[key]['s'] = self.caption
            elif key == 'legend':
                attrs[key]['show'] = self.legend
            elif key == 'xticks':
                attrs[key]['ticks'] = self.xticks
            elif key == 'yticks':
                attrs[key]['ticks'] = self.yticks
            elif key == 'crs':
                attrs[key]['crs'] = self.crs
            elif key == 'projection':
                attrs[key]['projection'] = self.projection
                attrs[key]['central_longitude'] = self.central_lon
            elif key == 'borders':
                attrs[key]['borders'] = self.borders
            elif key == 'coastline':
                attrs[key]['coastline'] = self.coastline
            elif key == 'land':
                attrs[key]['land'] = self.land
            elif key == 'lakes':
                attrs[key]['lakes'] = self.lakes
            elif key == 'ocean':
                attrs[key]['ocean'] = self.ocean
            elif key == 'rivers':
                attrs[key]['rivers'] = self.rivers
            elif key == 'states':
                attrs[key]['states'] = self.states
            elif key == 'clabel':
                try:  # TODO: refactor this
                    attrs[key]['text'] = self.clabel
                except AttributeError:
                    pass
            elif key == 'colorbar':
                try:
                    attrs[key]['show'] = self.colorbar
                except AttributeError:
                    pass
            elif key == 'cticks':
                try:
                    attrs[key]['ticks'] = self.cticks
                    attrs[key]['tick_labels'] = self.ctick_labels
                except AttributeError:
                    pass
        self_copy = deepcopy(self)
        data = {}
        for rowcol, ds in self_copy.data.items():
            ds.attrs.update(**attrs)
            ds.attrs['configured'] = True
            data[rowcol] = ds
        self_copy.data = data
        return self_copy


class ReferenceArray(param.Parameterized):

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def reference(self, x0s=None, x1s=None, y0s=None, y1s=None,
                  label=None, inline_labels=None, inline_loc=None,
                  rowcols=None, **kwds):
        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            kwds.update({
                'x0s': x0s, 'x1s': x1s, 'y0s': y0s, 'y1s': y1s, 'label': label,
                'inline_labels': inline_labels, 'inline_loc': inline_loc})
            has_kwds = {key: val is not None for key, val in kwds.items()}
            if has_kwds['x0s'] and has_kwds['x1s']:
                loc_axis = 'x'
            elif has_kwds['y0s'] and has_kwds['y1s']:
                loc_axis = 'y'
            elif has_kwds['x0s']:
                loc_axis = 'x'
            elif has_kwds['y0s']:
                loc_axis = 'y'

            for key in list(kwds):
                val = kwds[key]
                if isinstance(val, str):
                    if hasattr(ds, val):
                        kwds[key] = getattr(ds[loc_axis], val)('item')

            self_copy *= Reference(**kwds)

        return self_copy


class ColorArray(param.Parameterized):

    cs = param.ClassSelector(class_=(Iterable,))

    cticks = param.ClassSelector(class_=(Iterable,))
    ctick_labels = param.ClassSelector(class_=(Iterable,))
    colorbar = param.Boolean(default=True)
    clabel = param.String()

    def __init__(self, **kwds):
        super().__init__(**kwds)


class RemarkArray(param.Parameterized):

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def remark(self, remarks=None, durations=None, condition=None,
               xs=None, ys=None, cs=None, state_labels=None,
               inline_labels=None, first=False, rtol=1e-05, atol=1e-08,
               rowcols=None):
        args = (xs, ys, cs, state_labels, inline_labels, condition)
        args_none = sum([1 for arg in args if arg is None])
        if args_none == len(args):
            raise ValueError(
                'Must supply either xs, ys, cs, state_labels, '
                'inline_labels, or condition!')
        elif args_none != len(args) - 1:
            raise ValueError(
                'Must supply only one of xs, ys, cs, state_labels, '
                'inline_labels, or condition!')

        if durations is None and remarks is None:
            raise ValueError(
                'Must supply at least remarks or durations!')

        if rowcols is None:
            rowcols = self.data.keys()

        self_copy = deepcopy(self)
        data = {}
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            if xs is not None:
                condition = self._match_values(
                    ds['x'], xs, first, rtol, atol)
            elif ys is not None:
                condition = self._match_values(
                    ds['y'], ys, first, rtol, atol)
            elif cs is not None:
                condition = self._match_values(
                    ds['c'], cs, first, rtol, atol)
            elif state_labels is not None:
                condition = self._match_values(
                    ds['state_label'], state_labels, first, rtol, atol)
            elif inline_labels is not None:
                condition = self._match_values(
                    ds['inline_label'], inline_labels, first, rtol, atol)
            else:
                condition = np.array(condition)

            condition = condition.broadcast_like(ds)
            if remarks is not None:
                if 'remark' not in ds:
                    ds['remark'] = (
                        DIMS['base'],
                        np.full((len(ds['item']), self_copy._num_states), '')
                    )
                if isinstance(remarks, str):
                    if remarks in ds.data_vars:
                        remarks = ds[remarks].values.astype(str)
                ds['remark'] = xr.where(
                    condition, remarks, ds['remark'])

            if durations is not None:
                if 'duration' not in ds:
                    ds['duration'] = 'state', self._adapt_input(
                        np.zeros_like(ds['state']), reshape=False)
                ds['duration'] = xr.where(
                    condition, durations, ds['duration']
                )
                if 'item' in ds['duration'].dims:
                    ds['duration'] = ds['duration'].max('item')

            data[rowcol] = ds
        self_copy.data = data
        return self_copy


class Array(Data, ReferenceArray, ColorArray, RemarkArray):

    xs = param.ClassSelector(class_=(Iterable,))
    ys = param.ClassSelector(class_=(Iterable,))

    def __init__(self, xs, ys, **kwds):
        num_states = len(xs)
        super().__init__(num_states, **kwds)
        ds = self.data[self.rowcol]
        ds = ds.assign(**{
            'x': (DIMS['base'], self._adapt_input(xs)),
            'y': (DIMS['base'], self._adapt_input(ys))})
        self.data = {self.rowcol: ds}

    @staticmethod
    def _match_values(da, values, first, rtol, atol):
        if is_datetime(da):
            values = pd.to_datetime(values)
        if first:
            return xr.concat((
                da['state'] == (da >= value).argmax()
                for value in values), 'stack'
            ).sum('stack')
        try:
            return xr.concat((
                np.isclose(da, value, rtol=rtol, atol=atol)
                for value in to_1d(values)), 'stack'
            ).sum('stack')
        except TypeError:
            return da.isin(values)

    def invert(self):
        data = {}
        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            attrs = ds.attrs
            df = ds.to_dataframe().rename_axis(DIMS['base'][::-1])
            ds = df.to_xarray().assign_attrs(attrs).transpose(*DIMS['base'])
            data[rowcol] = ds
        self_copy.data = data
        return self_copy


class Array2D(Data, ReferenceArray, ColorArray, RemarkArray):

    chart = param.ObjectSelector(
        objects=CHARTS['grid'], default=CHARTS['grid'][0])

    inline_xs = param.ClassSelector(class_=(Iterable, int, float))
    inline_ys = param.ClassSelector(class_=(Iterable, int, float))

    def __init__(self, xs, ys, cs, **kwds):
        shape = cs.shape[-2:]
        if cs.ndim > 2:
            num_states = len(cs)
            if shape[0] != len(ys):
                cs = np.swapaxes(cs, -1, -2)
                shape = shape[::-1]  # TODO: auto figure out time dimension
        else:
            num_states = 1
        super().__init__(num_states, **kwds)

        ds = self.data[self.rowcol]
        ds = ds.assign_coords({
            'x': xs.values,
            'y': ys.values,
        }).assign({
            'c': (DIMS['grid'], self._adapt_input(cs, shape=shape))
        })

        inline_labels = self.inline_labels
        if isinstance(inline_labels, str):
            if inline_labels in ds.data_vars:
                inline_labels = ds[inline_labels].isel(item=[0])

        if inline_labels is not None:
            inline_xs = self.inline_xs
            inline_ys = self.inline_ys
            if inline_xs is None or inline_ys is None:
                raise ValueError(
                    'Must provide an inline x and y '
                    'if inline_labels is not None!')
            else:
                ds['inline_x'] = (
                    DIMS['base'], self._adapt_input(inline_xs))
                ds['inline_y'] = (
                    DIMS['base'], self._adapt_input(inline_ys))

        grid_vars = list(ds.data_vars) + ['x', 'y', 'item']
        ds = ds.rename({
            var: f'grid_{var}' for var in grid_vars
            if ds[var].dims != ('state',)})

        self.data = {self.rowcol: ds}


class DataFrame(Array):

    df = param.DataFrame()

    join = param.ObjectSelector(
        default='overlay', objects=['overlay', 'layout', 'cascade'])

    def __init__(self, df, xs, ys, join='overlay', **kwds):
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
                for key, val in kwds.items():
                    if isinstance(val, dict):
                        continue
                    elif isinstance(val, str):
                        if val in label_df.columns:
                            val = label_df[val].values
                    kwds_updated[key] = val

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


class Dataset(Array2D):
    def __init__(self):
        pass


class Reference(Data):

    chart = param.ObjectSelector(objects=CHARTS['refs'])

    x0s = param.ClassSelector(class_=(Iterable,))
    x1s = param.ClassSelector(class_=(Iterable,))
    y0s = param.ClassSelector(class_=(Iterable,))
    y1s = param.ClassSelector(class_=(Iterable,))
    inline_locs = param.ClassSelector(class_=(Iterable, int, float))

    def __init__(self, x0s=None, x1s=None, y0s=None, y1s=None, **kwds):
        ref_kwds = {
            'x0': x0s,
            'x1': x1s,
            'y0': y0s,
            'y1': y1s,
        }
        has_kwds = {key: val is not None for key, val in ref_kwds.items()}
        if not any(has_kwds.values()):
            raise ValueError('Must provide either x0s, x1s, y0s, y1s!')

        for key in list(ref_kwds):
            val = ref_kwds[key]
            if val is not None:
                num_states = to_1d(val, flat=False).shape[-1]
            else:
                ref_kwds.pop(key)

        has_xs = has_kwds['x0'] and has_kwds['x1']
        has_ys = has_kwds['y0'] and has_kwds['y1']
        if has_xs and has_ys:
            kwds['chart'] = 'rectangle'
        elif has_kwds['x0'] and has_kwds['y0']:
            kwds['chart'] = 'scatter'
        elif has_kwds['x0'] and has_kwds['x1']:
            kwds['chart'] = 'axvspan'
        elif has_kwds['y0'] and has_kwds['y1']:
            kwds['chart'] = 'axhspan'
        elif has_kwds['x0']:
            kwds['chart'] = 'axvline'
        elif has_kwds['y0']:
            kwds['chart'] = 'axhline'
        else:
            raise NotImplementedError(
                'One of the following combinations must be provided: '
                'x0+x1, y0+y1, x0+y0, x0, y0')

        super().__init__(num_states, **kwds)

        ds = self.data[self.rowcol]

        for key, val in ref_kwds.items():
            val = self._adapt_input(val)
            if val is not None:
                ds[key] = DIMS['refs'], val

        inline_labels = self.inline_labels
        if isinstance(inline_labels, str):
            if inline_labels in ds.data_vars:
                inline_labels = ds[inline_labels].isel(item=[0])

        if inline_labels is not None:
            inline_locs = self.inline_locs
            if inline_locs is None:
                raise ValueError(
                    'Must provide an inline location '
                    'if inline_labels is not None!')
            else:
                ds['inline_loc'] = (
                    DIMS['refs'], self._adapt_input(inline_locs))

        ds = ds.rename({
            var: f'ref_{var}' for var in list(ds.data_vars) + ['item']
            if ds[var].dims != ('state',)})

        self.data[self.rowcol] = ds
