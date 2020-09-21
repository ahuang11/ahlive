import param
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, rgb2hex

METHODS = [
    'linear',
    'quadratic',
    'cubic',
    'quartic',
    'quintic',
    'sine',
    'circular',
    'exponential',
    'elastic',
    'back',
    'bounce'
]
ALIASES = {
    'quad': 'quadratic',
    'quart': 'quartic',
    'quint': 'quintic',
    'circ': 'circular',
    'expo': 'exponential'
}
TIMINGS = ['in', 'out', 'in_out']


class Easing(param.Parameterized):

    method = param.ObjectSelector(
        default='cubic', objects=METHODS + list(ALIASES),
        doc='Interpolation method')
    timing = param.ObjectSelector(
        default='in_out', objects=TIMINGS,
        doc='Timing of when easing is applied')
    frames = param.Number(
        default=30, bounds=(1, None),
        doc='Number of frames per transition to next state')
    loop = param.ObjectSelector(
        default='boomerang',
        objects=['boomerang', 'traceback'] + list(range(0, 999)),
        doc='Number of times the animation plays; '
            'select 0, boomerang, or traceback to play indefinitely with '
            'boomerang finding the shortest path to the initial state and '
            'traceback backtracks the original path to the initial state.'
        )

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.method = ALIASES.get(self.method, self.method)

    def interp(self, da, name=''):
        is_xarray = isinstance(da, xr.DataArray)
        if is_xarray: name = da.name

        array = np.array(da)
        if array.ndim == 1:
            array = array.reshape(-1, len(array))
        if self.loop == 'boomerang':
            array = np.hstack([array, array[:, :1]])
        len_items, len_states = array.shape

        steps = np.linspace(0, 1, self.frames)
        len_steps = len(steps)
        if name == 'duration':
            len_result = (len_states - 1) * len_steps
            result = np.zeros(len_result) + 1 / 60.
            indices = np.arange(len_states) * len_steps
            indices[-1] -= 1
            result[indices] = array[0]
            result = result.reshape(1, -1)
            print(result)
        elif np.issubdtype(array.dtype, np.number):
            init = np.repeat(array[:, :-1], len_steps).reshape(len_items, -1)
            stop = np.repeat(array[:, 1:], len_steps).reshape(len_items, -1)
            tiled_steps = np.tile(
                steps, (len_states - 1) * len_items
            ).reshape(len_items, -1)
            weights = getattr(self, f'_{self.method}')(tiled_steps)
            result = stop * weights + init * (1 - weights)
        elif 'label' in name:
            result = np.repeat(array[:, 1:], len_steps).reshape(len_items, -1)
        elif name.startswith('c'):
            results = []
            for colors in array:
                cmap = LinearSegmentedColormap.from_list('eased', colors)
                results.append(
                    [rgb2hex(rgb) for rgb in cmap(range(len_steps))])
            result = np.array(results)
        else:
            raise NotImplementedError

        if self.loop == 'traceback':
            result = np.hstack([result, result[:, ::-1]])
        print(result.shape)

        if is_xarray:
            if len(da.dims) > 1:
                result = result.reshape(len_items, -1)
            else:
                result = result.squeeze()
            da_result = xr.DataArray(result, dims=da.dims, name=da.name)
            return da_result
        else:
            return result

    def _linear(self, ts):
        return ts

    def _quadratic(self, ts):
        if self.timing == 'in':
            ts = ts * ts
        elif self.timing == 'out':
            ts = -(ts * (ts - 2))
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = 2 * ts[index] * ts[index]
            ts[~index] = (-2 * ts[~index] * ts[~index]) + (4 * ts[~index]) - 1
        return ts

    def _cubic(self, ts):
        if self.timing == 'in':
            ts = ts * ts * ts
        elif self.timing == 'out':
            ts = (ts - 1) * (ts - 1) * (ts - 1) + 1
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = 4 * ts[index] * ts[index] * ts[index]
            ts[~index] = 2 * ts[~index] - 2
            ts[~index] = (
                0.5 * ts[~index] * ts[~index] * ts[~index] + 1)
        return ts

    def _quartic(self, ts):
        if self.timing == 'in':
            ts = ts * ts * ts * ts
        elif self.timing == 'out':
            ts = (ts - 1) * (ts - 1) * (ts - 1) * (1 - ts) + 1
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = 8 * ts[index] * ts[index] * ts[index] * ts[index]
            ts[~index] = ts[~index] - 1
            ts[~index] = (
                -8 * ts[~index] * ts[~index] * ts[~index] * ts[~index] + 1)
        return ts

    def _quintic(self, ts):
        if self.timing == 'in':
            ts = ts * ts * ts * ts * ts
        elif self.timing == 'out':
            ts = (ts - 1) * (ts - 1) * (ts - 1) * (ts - 1) * (ts - 1) + 1
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = (
                16 * ts[index] * ts[index] *
                ts[index] * ts[index] * ts[index])
            ts[~index] = (2 * ts[~index]) - 2
            ts[~index] = (
                0.5 * ts[~index] * ts[~index] *
                ts[~index] * ts[~index] * ts[~index] + 1)
        return ts

    def _sine(self, ts):
        if self.timing == 'in':
            ts = np.sin((ts - 1) * np.pi / 2) + 1
        elif self.timing == 'out':
            ts = np.sin(ts * np.pi / 2)
        elif self.timing == 'in_out':
            ts = 0.5 * (1 - np.cos(ts * np.pi))
        return ts

    def _circular(self, ts):
        if self.timing == 'in':
            ts = 1 - np.sqrt(1 - (ts * ts))
        elif self.timing == 'out':
            ts = np.sqrt((2 - ts) * ts)
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = 0.5 * (1 - np.sqrt(1 - 4 * (ts[index] * ts[index])))
            ts[~index] = 0.5 * (
                np.sqrt(-((2 * ts[~index]) - 3) * ((2 * ts[~index]) - 1)) + 1)
        return ts

    def _exponential(self, ts):
        if self.timing == 'in':
            index = ts != 0
            ts[~index] = 0
            ts[index] = np.power(2, 10 * (ts[index] - 1))
        elif self.timing == 'out':
            index = ts != 1
            ts[~index] = 1
            ts[index] = 1 - np.power(2, -10 * ts[index])
        elif self.timing == 'in_out':
            index0 = (ts != 0) & (ts < 0.5) & (ts != 1)
            index1 = (ts != 0) & (ts >= 0.5) & (ts != 1)
            ts[index0] = 0.5 * np.power(2, (20 * ts[index0]) - 10)
            ts[index1] = -0.5 * np.power(2, (-20 * ts[index1]) + 10) + 1
        return ts

    def _elastic(self, ts):
        if self.timing == 'in':
            ts = (
                np.sin(13 * np.pi / 2 * ts) * np.power(2, 10 * (ts - 1)))
        elif self.timing == 'out':
            ts = (
                np.sin(-13 * np.pi / 2 * (ts + 1)) *
                np.power(2, -10 * ts) + 1)
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = (
                0.5 * np.sin(13 * np.pi / 2 * (2 * ts[index])) *
                np.power(2, 10 * ((2 * ts[index]) - 1)))
            ts[~index] = 0.5 * (
                np.sin(-13 * np.pi / 2 * ((2 * ts[~index] - 1) + 1)) *
                np.power(2, -10 * (2 * ts[~index] - 1)) + 2)
        return ts

    def _back(self, ts):
        if self.timing == 'in':
            ts = ts * ts * ts - ts * np.sin(ts * np.pi)
        elif self.timing == 'out':
            ts = 1 - ts
            ts = 1 - (ts * ts * ts - ts * np.sin(ts * np.pi))
        elif self.timing == 'in_out':
            index = ts < 0.5
            ts[index] = 2 * ts[index]
            ts[index] = 0.5 * (
                ts[index] * ts[index] * ts[index] -
                ts[index] * np.sin(ts[index] * np.pi)
            )
            ts[~index] = 1 - (2 * ts[~index] - 1)
            ts[~index] = 0.5 * (1 - (
                ts[~index] * ts[~index] * ts[~index] -
                ts[~index] * np.sin(ts[~index] * np.pi))
            ) + 0.5
        return ts

    def _bounce(self, ts):
        index = ts < 0.5
        if self.timing == 'in':
            ts = 1 - ts
        elif self.timing == 'in_out':
            ts[index] = 1 - (ts[index] * 2)
            ts[~index] = ts[~index] * 2 - 1
        index0 = ts < 4 / 11
        index1 = (ts < 8 / 11) & ~index0
        index2 = (ts < 9 / 10) & ~index1 & ~index0
        index3 = ts >= 9 / 10
        ts[index0] = 121 * ts[index0] * ts[index0] / 16
        ts[index1] = (
            363 / 40.0 * ts[index1] * ts[index1]
        ) - (99 / 10.0 * ts[index1]) + 17 / 5.0
        ts[index2] = (
            (4356 / 361.0 * ts[index2] * ts[index2]) -
            (35442 / 1805.0 * ts[index2]) + 16061 / 1805.0
        )
        ts[index3] = (
            (54 / 5.0 * ts[index3] * ts[index3]) -
            (513 / 25.0 * ts[index3]) + 268 / 25.0
        )
        if self.timing == 'in':
            ts = 1 - ts
        elif self.timing == 'out':
            pass
        elif self.timing == 'in_out':
            ts[index] = 0.5 * (1 - ts[index])
            ts[~index] = 0.5 * ts[~index] + 0.5
        return ts
