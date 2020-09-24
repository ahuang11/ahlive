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

    def add_dataframe(self, df, xs, ys, label=None, state_labels=None,
                      inline_labels=None, **input_data_vars):
        if inline_labels is None:
            df_iter = zip('', df)
        else:
            df_iter = df.groupby(inline_labels)
        for item, df_item in df_iter:
            input_kwds = {
                key: df_item[val] if val in df_item.columns else val
                for key, val in input_data_vars.items()}
            self.add_arrays(
                df_item[xs], df_item[ys], label=label,
                state_labels=df_item[state_labels],
                inline_labels=item, **input_kwds)
        self.xlabel = xs
        self.ylabel = ys

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
                stat = 'min' if left or limit == 'min' else 'max'
                if axis not in paddings:
                    paddings[axis] = (
                        getattr(self.ds[axis], stat)() / 5
                    ).values
                padding = paddings[axis]

                if limit == 'explore':
                    if left:
                        limit = self.ds[axis].min()
                    else:
                        limit = np.maximum.accumulate(
                            self.ds[axis].max('item').values)
                else:
                    dims = 'item' if limit == 'follow' else None
                    limit = getattr(self.ds[axis], stat)(dims)

                limit = limit - padding if left else limit + padding
                if len(np.atleast_1d(limit)) == 1:
                    limit = np.repeat(limit.values, len(self.ds['state']))
            self.ds[key] = ('state', limit)

    def _add_durations(self):
        if self.durations is None:
            durations = 2.8 if len(self.ds['state']) < 10 else 1 / 30
        else:
            durations = self.durations

        if isinstance(durations, (int, float)):
            durations = np.repeat(durations, len(self.ds['state']))
        self.ds['duration'] = ('state', durations)

    def save(self):
        self._add_xy01_limits()
        self._add_durations()
        ds = self.ds.reset_coords()
        ds = ds.apply(self.interp)
        print(ds)
        super().save(ds)
