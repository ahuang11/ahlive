from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from .easing import Easing
from .animation import Animation

OPTIONS = {
    'limit': ['fixed', 'follow']
}


class Ahlive(Easing, Animation):

    x0_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    x1_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    y0_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    y1_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    durations = param.ClassSelector(
        default=None, class_=(Iterable, int, float))

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _add_data(self, label=None, state_labels=None, **input_data_vars):
        if all(key in input_data_vars for key in ['x', 'y']):
            item_type = None
            self.num_states = len(input_data_vars['x'])
            saved_ds = self.ds
        else:
            if all(key in input_data_vars for key in ['x0', 'x1', 'y0', 'y1']):
                item_type = 'box'
                self.num_states = len(input_data_vars['x0'])
            elif all(key in input_data_vars for key in ['y0', 'x0', 'x1']):
                item_type = 'finite_hline'
                self.num_states = len(input_data_vars['y0'])
            elif all(key in input_data_vars for key in ['x0', 'y0', 'y1']):
                item_type = 'finite_vline'
                self.num_states = len(input_data_vars['x0'])
            elif 'y0' in input_data_vars:
                item_type = 'hline'
                self.num_states = len(input_data_vars['y0'])
            elif 'x0' in input_data_vars:
                item_type = 'vline'
                self.num_states = len(input_data_vars['x0'])
            saved_ds = self.ref_ds

        item_num = len(saved_ds['item']) if saved_ds else 0
        coords = {'item': [item_num], 'state': range(self.num_states)}
        if item_type is not None:
            coords['item_type'] = ('item', [item_type])

        if label is not None:
            if not isinstance(label, str):
                if len(label) > 0:
                    label = label[0]
            coords['label'] = ('item', [label])
            if item_num > 0 and 'label' not in saved_ds:
                saved_ds.coords['label'] = ('item', [label])
        elif 'label' in saved_ds:
            coords['label'] = saved_ds['label'].values[0]

        if state_labels is not None:
            coords['state_label'] = ('state', state_labels)

        data_vars = {
            key: val for key, val in input_data_vars.items()
            if val is not None}
        for var in list(data_vars.keys()):
            val = np.array(data_vars.pop(var))
            dims = ('item', 'state')
            if len(np.atleast_1d(val)) == 1:
                val = [val] * self.num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = dims, val
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        if not saved_ds:
            saved_ds = ds
        else:
            saved_ds = xr.concat([saved_ds, ds], 'item')

        return saved_ds

    def add_arrays(self, xs, ys, label=None, state_labels=None,
                   inline_labels=None, **input_data_vars):
        self.ds = self._add_data(
            x=xs, y=ys, label=label, state_labels=state_labels,
            inline_label=inline_labels, **input_data_vars)

    def add_references(self, x0s=None, x1s=None, y0s=None, y1s=None,
                       label=None, state_labels=None, inline_labels=None,
                       **input_data_vars):
        if all(var is None for var in [x0s, x1s, y0s, y1s]):
            raise ValueError('Must provide either x0s, x1s, y0s, y1s!')
        elif x0s is None and x1s is not None:
            x0s, x1s = x1s, x0s
        elif y0s is None and y1s is not None:
            y0s, y1s = y1s, y0s

        x0s = x0s or np.nan
        x1s = x1s or np.nan
        y0s = y0s or np.nan
        y1s = y1s or np.nan

        self.ds_ref = self._add_data(
            x0=np.atleast_1d(x0s), x1=np.atleast_1d(x1s),
            y0=np.atleast_1d(y0s), y1=np.atleast_1d(y1s),
            label=label, state_labels=state_labels, inline_label=inline_labels,
            **input_data_vars
        )

    def add_dataframe(self, df, xs, ys, labels=None, state_labels=None,
                      inline_labels=None, **input_data_vars):
        if 'label' in input_data_vars:
            labels = input_data_vars.pop('label')

        if labels is None:
            df_iter = zip([None], [df])
        else:
            df_iter = df.groupby(labels)

        for label, df_item in df_iter:
            input_kwds = {
                key: df_item[val] if val in df_item.columns else val
                for key, val in input_data_vars.items()}
            self.add_arrays(
                df_item[xs], df_item[ys], label=label,
                state_labels=df_item.get(state_labels, None),
                inline_labels=df_item.get(inline_labels, None),
                **input_kwds
            )

        if self.chart == 'barh':
            self.xlabel = ys
            self.ylabel = xs
        else:
            self.xlabel = xs
            self.ylabel = ys

    def _add_xy01_limits(self):
        limits = {
            'x0_limit': self.x0_limits,
            'x1_limit': self.x1_limits,
            'y0_limit': self.y0_limits,
            'y1_limit': self.y1_limits
        }

        for key, limit in limits.items():
            axis = key[0]
            left = int(key[1]) == 0

            in_axes_kwds = f'{axis}lim' in self.axes_kwds
            is_scatter = self.chart == 'scatter'
            is_bar_y = self.chart.startswith('bar') and axis == 'y'
            if limit is None and not in_axes_kwds and (is_scatter or is_bar_y):
                limit = 'fixed'
            elif isinstance(limit, str) and limit not in OPTIONS['limit']:
                raise ValueError(
                    f"Got {limit} for {key}; must be either "
                    f"from {OPTIONS['limit']} or numeric values!"
                )

            if isinstance(limit, str):
                input_ = limit
                stat = 'min' if left or limit == 'min' else 'max'
                dims = 'item' if limit == 'follow' else None
                limit = getattr(self.ds[axis], stat)(dims)
                if len(np.atleast_1d(limit)) == 1:
                    limit = np.repeat(limit.values, len(self.ds['state']))

            if limit is not None:
                if self.chart == 'barh':
                    axis = 'x' if axis == 'y' else 'y'
                    key = axis + key[1:]
                if len(np.atleast_1d(limit)) == 1:  # TODO: make util
                    limit = [limit] * self.num_states
                self.ds[key] = ('state', limit)

    def _add_durations(self):
        if self.durations is None:
            durations = 0.5 if len(self.ds['state']) < 10 else 1 / 30
        else:
            durations = self.durations

        if isinstance(durations, (int, float)):
            durations = np.repeat(durations, len(self.ds['state']))
        self.ds['duration'] = ('state', durations)

    def save(self):
        if self.chart is None:
            self.chart = 'scatter' if len(self.ds['state']) <= 5 else 'line'
        elif self.chart.startswith('bar'):
            self.ds = xr.concat([
                ds_group.unstack().assign(**{
                    'item': [item], 'state': range(len(ds_group['state']))})
                for item, (group, ds_group) in enumerate(self.ds.groupby('x'))
            ], 'item')
        self._add_xy01_limits()
        self._add_durations()
        if 'label' not in self.ds:
            self.ds['label'] = ('item', np.repeat('', len(self.ds['item'])))
        ds = self.ds.reset_coords()
        if self.loop is None:
            self.loop = 0 if self.chart == 'line' else 'boomerang'

        # initialize
        self.xmin = ds['x'].min()
        self.ymin = ds['y'].min()
        try:
            self.state_label_step = np.abs(np.diff(ds.get(
                'state_label', np.array([0, 1])
            )).min())
        except TypeError:
            self.state_label_step = ds.get(
                'state_label', np.array([0, 1])
            ).min()
        try:
            self.inline_label_step = np.abs(np.diff(ds.get(
                'inline_label', np.array([0, 1])
            )).min())
        except TypeError:
            self.inline_label_step = ds.get(
                'inline_label', np.array([0, 1])
            ).min()

        self.x_is_datetime = np.issubdtype(ds['x'].values.dtype, np.datetime64)
        self.y_is_datetime = np.issubdtype(ds['y'].values.dtype, np.datetime64)

        if 'c' in ds:
            self.vmin = ds['c'].min()
            self.vmax = ds['c'].max()

        ds = ds.apply(self.interpolate)
        print(ds)
        super().save(ds)
