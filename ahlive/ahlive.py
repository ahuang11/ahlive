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
        default=None, class_=(Iterable, int, float), allow_None=True
    )
    x1_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float), allow_None=True
    )
    y0_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float), allow_None=True
    )
    y1_limits = param.ClassSelector(
        default=None, class_=(Iterable, int, float), allow_None=True
    )
    durations = param.ClassSelector(
        default=None, class_=(Iterable, int, float),
        allow_None=True
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def add_arrays(self, xs, ys, label=None, state_labels=None,
                   inline_labels=None, **input_data_vars):
        num_states = len(xs)

        item_num = len(self.ds['item']) if self.ds else 0
        coords = {'item': [item_num], 'state': range(num_states)}

        if label is not None:
            if not isinstance(label, str):
                if len(label) > 0:
                    label = label[0]
            coords['label'] = ('item', [label])
            if item_num > 0 and 'label' not in self.ds:
                self.ds.coords['label'] = ('item', [label])
        elif 'label' in self.ds:
            coords['label'] = self.ds['label'].values[0]

        if state_labels is not None:
            coords['state_label'] = ('state', state_labels)

        data_vars = input_data_vars.copy()
        data_vars.update({'x': xs, 'y': ys})
        if inline_labels is not None:
            data_vars['inline_label'] = inline_labels

        for var in list(data_vars.keys()):
            val = np.array(data_vars.pop(var))
            dims = ('item', 'state')
            if len(np.atleast_1d(val)) == 1:
                val = [val] * num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = dims, val
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        if not self.ds:
            self.ds = ds
        else:
            self.ds = xr.concat([self.ds, ds], 'item')

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

        if self.plot == 'barh':
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
            is_scatter = self.plot == 'scatter'
            is_bar_y = self.plot.startswith('bar') and axis == 'y'
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
                if self.plot == 'barh':
                    axis = 'x' if axis == 'y' else 'y'
                    key = axis + key[1:]
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
        if self.plot is None:
            self.plot = 'scatter' if len(ds['state']) <= 5 else 'line'
        elif self.plot.startswith('bar'):
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
            self.loop = 0 if self.plot == 'line' else 'boomerang'
        ds = ds.apply(self.interpolate)
        print(ds)
        super().save(ds)
