import dask
import param
import imageio
import numpy as np
import xarray as xr
from pygifsicle import optimize

from .easing import Easing
from .animation import Animation

LABELS = ['item_label', 'state_label']
LIMITS = ['xlim0', 'xlim1', 'ylim0', 'ylim1']


class Ahlive(Easing, Animation):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def add_arrays(self, x, y, **input_data_vars):
        num_states = len(x)
        data_vars = input_data_vars.copy()
        data_vars.update({'x': x, 'y': y})
        item_num = len(self.ds['item']) if self.ds else 0
        coords = {'item': [item_num], 'state': range(num_states)}
        for var in list(data_vars.keys()):
            val = data_vars.pop(var)
            if var in LABELS:
                raise ValueError(f'Use add_labels method to add {var}')
            elif var in LIMITS:
                raise ValueError(f'Use add_limits method to add {var}')
            else:
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

    def add_labels(self, state_labels=None, item_labels=None):
        if state_labels is None and item_labels is None:
            raise ValueError('Provide either state_labels or item_labels!')
        elif not self.ds:
            raise ValueError('First use add_arrays method at least once!')

        if state_labels:
            self.ds['state_label'] = ('state', state_labels)

        if item_labels:
            item_labels = np.reshape(item_labels, (1, -1)).transpose()
            item_labels = np.tile(item_labels, len(self.ds['state']))
            self.ds['item_label'] = (('item', 'state'), item_labels)

    def add_limits(
            self,
            x0_limits=None,
            x1_limits=None,
            y0_limits=None,
            y1_limits=None
        ):
        if not self.ds:
            raise ValueError('First use add_arrays method at least once!')

        limits = {
            'x0_limit': x0_limits,
            'x1_limit': x1_limits,
            'y0_limit': y0_limits,
            'y1_limit': y1_limits
        }

        paddings = {}
        for key, limit in limits.items():
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

    def add_durations(self, durations):
        self.ds['duration'] = ('state', durations)

    def save(self):
        ds_eased = self.ds.reset_coords().apply(self.interp)
        duration = ds_eased['duration'].values.tolist()
        ds_eased = ds_eased.drop('duration')

        with dask.diagnostics.ProgressBar(minimum=3):
            buf_list = dask.compute([
                self._draw(ds_eased, state)
                for state in ds_eased['state'].values
            ], scheduler='processes', num_workers=self.num_workers)[0]

        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        with imageio.get_writer(
            self.out_fp, format='gif', mode='I',
            subrectangles=True, duration=duration, loop=loop
        ) as writer:
            for buf in buf_list:
                image = imageio.imread(buf)
                writer.append_data(image)
        optimize(self.out_fp)
        gif = imageio.get_reader(self.out_fp)
        return gif
