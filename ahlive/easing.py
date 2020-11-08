import param
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, rgb2hex

INTERPS = [
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
EASES = ['in', 'out', 'in_out']


class Easing(param.Parameterized):

    interp = param.ObjectSelector(
        default=None, objects=INTERPS + list(ALIASES),
        doc='Interpolation method')
    ease = param.ObjectSelector(
        default='in_out', objects=EASES,
        doc='Type of easing')
    frames = param.Integer(
        default=None, bounds=(1, None),
        doc='Number of frames per transition to next state')
    loop = param.ObjectSelector(
        default=None,
        objects=['boomerang', 'traceback', 'rollback'] + list(range(0, 999)),
        doc='Number of times the animation plays; '
            'select 0, boomerang, or traceback to play indefinitely with '
            'boomerang finding the shortest path to the initial state and '
            'traceback backtracks the original path to the initial state.'
        )

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def interpolate(self, da, name=''):
        is_xarray = isinstance(da, xr.DataArray)
        if is_xarray:
            name = da.name
            if 'state' not in da.dims:
                return da

        array = np.array(da)
        if array.ndim == 1:
            array = array.reshape(-1, len(array))
        if self.loop == 'boomerang':
            array = np.hstack([array, array[:, :1]])

        num_items, num_states = array.shape
        new_shape = (num_items, -1)

        if self.frames is None:
            if num_states < 10:
                num_steps = int(np.ceil(80 / num_states))
            else:
                num_steps = int(np.ceil(350 / num_states))
        else:
            num_steps = self.frames

        if num_steps == 1 and isinstance(self.loop, int):
            return da

        steps = np.linspace(0, 1, num_steps)

        if self.interp is None:
            interp = 'cubic' if num_states < 10 else 'linear'
        else:
            interp = ALIASES.get(self.interp, self.interp)

        num_result = (num_states - 1) * num_steps
        if name in ['delay', 'root']:
            result = np.full(num_result, 0.)
            indices = np.arange(num_states) * num_steps
            indices[-1] -= 1
            result[indices] = array[0]  # (1, num_states)
            result = result.reshape(1, -1)
        elif 'trail' in name:
            indices = np.arange(num_states * num_steps - num_steps)
            result = pd.DataFrame(
                array, columns=np.arange(0, num_states * num_steps, num_steps)
            ).T.reindex(indices).T.values
        elif 'annotation' in name:
            indices = np.arange(num_states * num_steps - num_steps)
            result = pd.DataFrame(
                array, columns=np.arange(0, num_states * num_steps, num_steps)
            ).T.reindex(indices).T.fillna('').values
        elif np.issubdtype(array.dtype, np.datetime64):
            array = array.astype(float)
            init = np.repeat(array[:, :-1], num_steps, axis=1)
            stop = np.repeat(array[:, 1:], num_steps, axis=1)
            tiled_steps = np.tile(
                steps, (num_states - 1) * num_items
            ).reshape(*new_shape)
            weights = getattr(self, f'_{interp}')(tiled_steps)
            result = stop * weights + init * (1 - weights)
            result = result.astype(np.datetime64)
        elif np.issubdtype(array.dtype, np.number):
            init = np.repeat(array[:, :-1], num_steps, axis=1)
            stop = np.repeat(array[:, 1:], num_steps, axis=1)
            tiled_steps = np.tile(
                steps, (num_states - 1) * num_items
            ).reshape(*new_shape)
            weights = getattr(self, f'_{interp}')(tiled_steps)
            result = stop * weights + init * (1 - weights)
        elif name in ['c', 'color']:
            results = []
            for colors in array:
                cmap = LinearSegmentedColormap.from_list('eased', colors)
                results.append(
                    [rgb2hex(rgb) for rgb in cmap(np.arange(num_steps))])
            result = np.array(results)
        else:
            result = np.repeat(array, num_steps, axis=1)
            if name == 'state_label':
                num_roll = -int(np.ceil(num_steps / num_states * 2))
                result = np.roll(result, num_roll, axis=-1)
            result = result[:, :num_result]

        if self.loop in ['traceback', 'rollback']:
            result_back = result[:, ::-2]
            if name == 'delay' and self.loop == 'rollback':
                result_back = np.repeat(
                    0, result_back.shape[1]
                ).reshape(1, -1)
            result = np.hstack([result, result_back])

        if is_xarray:
            if len(da.dims) == 1:
                result = result.squeeze()
            da_result = xr.DataArray(
                result, dims=da.dims, name=da.name, attrs=da.attrs)
            return da_result
        else:
            return result

    def _linear(self, ts):
        return ts

    def _quadratic(self, ts):
        if self.ease == 'in':
            ts = ts * ts
        elif self.ease == 'out':
            ts = -(ts * (ts - 2))
        elif self.ease == 'in_out':
            index = ts < 0.5
            ts[index] = 2 * ts[index] * ts[index]
            ts[~index] = (-2 * ts[~index] * ts[~index]) + (4 * ts[~index]) - 1
        return ts

    def _cubic(self, ts):
        if self.ease == 'in':
            ts = ts * ts * ts
        elif self.ease == 'out':
            ts = (ts - 1) * (ts - 1) * (ts - 1) + 1
        elif self.ease == 'in_out':
            index = ts < 0.5
            ts[index] = 4 * ts[index] * ts[index] * ts[index]
            ts[~index] = 2 * ts[~index] - 2
            ts[~index] = (
                0.5 * ts[~index] * ts[~index] * ts[~index] + 1)
        return ts

    def _quartic(self, ts):
        if self.ease == 'in':
            ts = ts * ts * ts * ts
        elif self.ease == 'out':
            ts = (ts - 1) * (ts - 1) * (ts - 1) * (1 - ts) + 1
        elif self.ease == 'in_out':
            index = ts < 0.5
            ts[index] = 8 * ts[index] * ts[index] * ts[index] * ts[index]
            ts[~index] = ts[~index] - 1
            ts[~index] = (
                -8 * ts[~index] * ts[~index] * ts[~index] * ts[~index] + 1)
        return ts

    def _quintic(self, ts):
        if self.ease == 'in':
            ts = ts * ts * ts * ts * ts
        elif self.ease == 'out':
            ts = (ts - 1) * (ts - 1) * (ts - 1) * (ts - 1) * (ts - 1) + 1
        elif self.ease == 'in_out':
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
        if self.ease == 'in':
            ts = np.sin((ts - 1) * np.pi / 2) + 1
        elif self.ease == 'out':
            ts = np.sin(ts * np.pi / 2)
        elif self.ease == 'in_out':
            ts = 0.5 * (1 - np.cos(ts * np.pi))
        return ts

    def _circular(self, ts):
        if self.ease == 'in':
            ts = 1 - np.sqrt(1 - (ts * ts))
        elif self.ease == 'out':
            ts = np.sqrt((2 - ts) * ts)
        elif self.ease == 'in_out':
            index = ts < 0.5
            ts[index] = 0.5 * (1 - np.sqrt(1 - 4 * (ts[index] * ts[index])))
            ts[~index] = 0.5 * (
                np.sqrt(-((2 * ts[~index]) - 3) * ((2 * ts[~index]) - 1)) + 1)
        return ts

    def _exponential(self, ts):
        if self.ease == 'in':
            index = ts != 0
            ts[~index] = 0
            ts[index] = np.power(2, 10 * (ts[index] - 1))
        elif self.ease == 'out':
            index = ts != 1
            ts[~index] = 1
            ts[index] = 1 - np.power(2, -10 * ts[index])
        elif self.ease == 'in_out':
            index0 = (ts != 0) & (ts < 0.5) & (ts != 1)
            index1 = (ts != 0) & (ts >= 0.5) & (ts != 1)
            ts[index0] = 0.5 * np.power(2, (20 * ts[index0]) - 10)
            ts[index1] = -0.5 * np.power(2, (-20 * ts[index1]) + 10) + 1
        return ts

    def _elastic(self, ts):
        if self.ease == 'in':
            ts = (
                np.sin(13 * np.pi / 2 * ts) * np.power(2, 10 * (ts - 1)))
        elif self.ease == 'out':
            ts = (
                np.sin(-13 * np.pi / 2 * (ts + 1)) *
                np.power(2, -10 * ts) + 1)
        elif self.ease == 'in_out':
            index = ts < 0.5
            ts[index] = (
                0.5 * np.sin(13 * np.pi / 2 * (2 * ts[index])) *
                np.power(2, 10 * ((2 * ts[index]) - 1)))
            ts[~index] = 0.5 * (
                np.sin(-13 * np.pi / 2 * ((2 * ts[~index] - 1) + 1)) *
                np.power(2, -10 * (2 * ts[~index] - 1)) + 2)
        return ts

    def _back(self, ts):
        if self.ease == 'in':
            ts = ts * ts * ts - ts * np.sin(ts * np.pi)
        elif self.ease == 'out':
            ts = 1 - ts
            ts = 1 - (ts * ts * ts - ts * np.sin(ts * np.pi))
        elif self.ease == 'in_out':
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
        if self.ease == 'in':
            ts = 1 - ts
        elif self.ease == 'in_out':
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
        if self.ease == 'in':
            ts = 1 - ts
        elif self.ease == 'out':
            pass
        elif self.ease == 'in_out':
            ts[index] = 0.5 * (1 - ts[index])
            ts[~index] = 0.5 * ts[~index] + 0.5
        return ts
