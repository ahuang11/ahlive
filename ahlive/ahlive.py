from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from .easing import Easing
from .animation import Animation

OPTIONS = {
    'limit': ['min', 'max', 'follow', 'explore']
}


class Ahlive(Easing, Animation):

    state_labels = param.List(default=None, allow_None=True)
    x0_limits = param.ClassSelector(
        default='min', class_=(Iterable, int, float)
    )
    x1_limits = param.ClassSelector(
        default='explore', class_=(Iterable, int, float)
    )
    y0_limits = param.ClassSelector(
        default='min', class_=(Iterable, int, float)
    )
    y1_limits = param.ClassSelector(
        default='explore', class_=(Iterable, int, float)
    )
    durations = param.ClassSelector(
        default=1 / 60, class_=(Iterable, int, float)
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def add_arrays(self, xs, ys, inline_labels=None, label=None,
                   **input_data_vars):
        num_states = len(xs)
        if inline_labels is None:
            inline_labels = np.repeat('', num_states)
        data_vars = input_data_vars.copy()
        data_vars.update({'x': xs, 'y': ys, 'inline_label': inline_labels})
        item_num = len(self.ds['item']) if self.ds else 0
        coords = {'item': [item_num], 'state': range(num_states),
                  'label': ('item', [label])}
        for var in list(data_vars.keys()):
            val = data_vars.pop(var)
            dims = ('item', 'state')
            if len(np.atleast_1d(val)) == 1:
                val = [val] * num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = dims, val
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        if self.ds is None:
            self.ds = ds
        else:
            self.ds = xr.concat([self.ds, ds], 'item')

    def _add_state_labels(self):
        if self.state_labels:
            self.ds['state_label'] = ('state', self.state_labels)

    def _add_xy01_limits(self):
        limits = {
            'x0_limit': self.x0_limits,
            'x1_limit': self.x1_limits,
            'y0_limit': self.y0_limits,
            'y1_limit': self.y1_limits
        }

        paddings = {}
        for key, limit in limits.items():
            if isinstance(limit, str):
                if limit not in OPTIONS['limit']:
                    raise ValueError(
                        f'Select from {OPTIONS["limit"]} if using a string; '
                        f'got {limit} for {key}!')
            axis = key[0]
            left = int(key[1]) == 0
            if limit is None:
                limit = 'min' if left else 'follow'

            if isinstance(limit, str):
                input_ = limit
                if axis not in paddings:
                    paddings[axis] = (
                        np.abs(self.ds[axis].diff('state')).mean().values)
                padding = paddings[axis]

                if limit == 'explore':
                    if left:
                        limit = self.ds[axis].min()
                    else:
                        limit = np.maximum.accumulate(
                            self.ds[axis].max('item').values)
                else:
                    dims = 'item' if limit == 'follow' else None
                    stat = 'min' if left or limit == 'min' else 'max'
                    limit = getattr(self.ds[axis], stat)(dims)

                limit = limit - padding if left else limit + padding
                if len(np.atleast_1d(limit)) == 1:
                    limit = np.repeat(limit.values, len(self.ds['state']))
            self.ds[key] = ('state', limit)

    def _add_durations(self):
        if isinstance(self.durations, (int, float)):
            durations = np.repeat(self.durations, len(self.ds['state']))
        else:
            durations = self.durations
        self.ds['duration'] = ('state', durations)

    def save(self):
        self._add_state_labels()
        self._add_xy01_limits()
        self._add_durations()
        ds = self.ds.reset_coords()
        ds = ds.apply(self.interp)
        print(ds)
        super().save(ds)
