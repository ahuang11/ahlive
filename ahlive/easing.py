from collections.abc import Iterable

import numpy as np
import pandas as pd
import param
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, rgb2hex

INTERPS = [
    "fill",
    "linear",
    "cubic",
    "exponential",
    "quadratic",
    "quartic",
    "quintic",
    "circular",
    "sine",
    "bounce",
    "elastic",
    "back",
]
EASES = ["in_out", "in", "out"]

REVERTS = ["boomerang", "traceback", "rollback"]


class Easing(param.Parameterized):

    interp = param.ClassSelector(
        default=None,
        class_=Iterable,
        doc=f"Interpolation method; {INTERPS}",
    )
    ease = param.ClassSelector(
        default="in_out", class_=Iterable, doc=f"Type of easing; {EASES}"
    )
    frames = param.Integer(
        default=None,
        bounds=(1, None),
        doc="Number of frames between each base state",
    )
    revert = param.ObjectSelector(
        default=None,
        objects=REVERTS,
        doc="Method for reverting to the initial state; "
        "boomerang finds the shortest path to the initial state, "
        "traceback backtracks the original path to the initial state, and "
        "rollback is like traceback, but disregards the "
        "original's path durations",
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def interpolate(self, da, name=""):
        interp = self.interp
        if interp is None:
            interp = "cubic"
        ease = self.ease
        frames = self.frames
        revert = self.revert
        is_xarray = isinstance(da, xr.DataArray)
        da_origin = da.copy()
        if is_xarray:
            name = da.name
            if "state" not in da.dims:
                return da_origin

            for item_dim in da.dims:
                if "item" in item_dim:
                    da = da.transpose(item_dim, "state", ...)
                    break

            frames = da.attrs.get("frames", None)
            revert = da.attrs.get("revert", None)
            interp = da.attrs.get("interp", None)
            ease = da.attrs.get("ease", None)

            if len(da.dims) > 2:  # more than (item, state)
                da = da.stack({"stacked": ["grid_item", "grid_y", "grid_x"]})
                da = da.transpose("stacked", "state")
            coords = da.drop_vars("state", errors="ignore").coords

        array = np.array(da)
        if array.ndim == 1:
            array = array.reshape(-1, len(array))
        if revert == "boomerang":
            array = np.hstack([array, array[:, :1]])

        num_items, num_states = array.shape
        if frames is None:
            if num_states < 10:
                num_steps = int(np.ceil(60 / num_states))
            else:
                num_steps = int(np.ceil(100 / num_states))
        else:
            num_steps = frames
        self._num_steps = num_steps

        new_shape = (num_items, -1)
        has_revert = isinstance(revert, int) or revert is not None
        if num_steps == 1 and not has_revert:
            return da_origin

        steps = np.linspace(0, 1, num_steps)
        num_result = (num_states - 1) * num_steps
        if name in ["duration", "root"]:
            result = np.full(num_result, 0.0)
            indices = np.arange(num_states) * num_steps
            indices[-1] -= 1
            result[indices] = array[0]  # (1, num_states)
            result = result.reshape(1, -1)
        elif interp == "fill" or name.endswith("discrete_trail"):
            result = self._fill(array, num_states, num_steps)
            if interp == "fill":
                result = result.ffill(axis=1)
                result.iloc[:, -1] = array[:, -1]
            result = result.values
        elif "remark" in name:
            result = self._fill(array, num_states, num_steps).fillna("").values
            result[:, -1] = array[:, -1]
        elif np.issubdtype(array.dtype, np.datetime64):
            array = array.astype(float)
            result = self._interp(
                array,
                steps,
                interp,
                ease,
                num_states,
                num_steps,
                num_items,
                new_shape,
            )
            result = pd.to_datetime(result.ravel()).values
            result = result.reshape(new_shape)
        elif np.issubdtype(array.dtype, np.timedelta64):
            array = array.astype(float)
            result = self._interp(
                array,
                steps,
                interp,
                ease,
                num_states,
                num_steps,
                num_items,
                new_shape,
            )
            result = pd.to_timedelta(result.ravel()).values
            result = result.reshape(new_shape)
        elif np.issubdtype(array.dtype, np.number):
            if name == "central_longitude":
                interp = "linear"
            result = self._interp(
                array,
                steps,
                interp,
                ease,
                num_states,
                num_steps,
                num_items,
                new_shape,
            )
        elif name in ["c", "color"]:
            results = []
            for colors in array:
                cmap = LinearSegmentedColormap.from_list("eased", colors)
                results.append([rgb2hex(rgb) for rgb in cmap(np.arange(num_steps))])
            result = np.array(results)
        else:
            result = np.repeat(array, num_steps, axis=1)
            num_roll = -int(np.ceil(num_steps / num_states * 2))
            if num_states > 2:
                result = np.roll(result, num_roll, axis=-1)
                result = result[:, :num_result]
            else:
                half_way = int(num_result / 2)
                result = result[:, half_way:-half_way]
                if num_steps % 2 != 0:
                    result = result[:, :-1]

        if revert in ["traceback", "rollback"]:
            if result.ndim == 1:
                result_back = result[::-1]
            else:
                result_back = result[:, ::-1]
            if name == "duration" and revert == "rollback":
                result_back = np.repeat(1 / 60, result_back.shape[1]).reshape(1, -1)
            result = np.hstack([result, result_back])

        if is_xarray:
            if len(da.dims) == 1:
                result = result.squeeze()
            da_result = xr.DataArray(
                result,
                dims=da.dims,
                coords=coords,
                name=da.name,
                attrs=da.attrs,
            )
            if "stacked" in da_result.dims:
                da_result = da_result.unstack()
            return da_result
        else:
            return result

    def _fill(self, array, num_states, num_steps):
        indices = np.arange(num_states * num_steps - num_steps)
        return (
            pd.DataFrame(
                array,
                columns=np.arange(0, num_states * num_steps, num_steps),
            )
            .T.reindex(indices)
            .T
        )

    def _interp(
        self,
        array,
        steps,
        interp,
        ease,
        num_states,
        num_steps,
        num_items,
        new_shape,
    ):
        init = np.repeat(array[:, :-1], num_steps, axis=1)
        init_nans = np.isnan(init)
        init[init_nans] = 0  # temporarily fill the nans
        stop = np.repeat(array[:, 1:], num_steps, axis=1)
        stop_nans = np.isnan(stop)
        tiled_steps = np.tile(steps, (num_states - 1) * num_items).reshape(*new_shape)
        weights = getattr(self, f"_{interp.lower()}")(tiled_steps, ease)
        result = stop * weights + init * (1 - weights)
        result[init_nans | stop_nans] = np.nan  # replace nans
        return result

    def _linear(self, ts, ease):
        return ts

    def _quadratic(self, ts, ease):
        if ease == "in":
            ts = ts * ts
        elif ease == "out":
            ts = -(ts * (ts - 2))
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = 2 * ts[index] * ts[index]
            ts[~index] = (-2 * ts[~index] * ts[~index]) + (4 * ts[~index]) - 1
        return ts

    def _cubic(self, ts, ease):
        if ease == "in":
            ts = ts * ts * ts
        elif ease == "out":
            ts = (ts - 1) * (ts - 1) * (ts - 1) + 1
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = 4 * ts[index] * ts[index] * ts[index]
            ts[~index] = 2 * ts[~index] - 2
            ts[~index] = 0.5 * ts[~index] * ts[~index] * ts[~index] + 1
        return ts

    def _quartic(self, ts, ease):
        if ease == "in":
            ts = ts * ts * ts * ts
        elif ease == "out":
            ts = (ts - 1) * (ts - 1) * (ts - 1) * (1 - ts) + 1
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = 8 * ts[index] * ts[index] * ts[index] * ts[index]
            ts[~index] = ts[~index] - 1
            ts[~index] = -8 * ts[~index] * ts[~index] * ts[~index] * ts[~index] + 1
        return ts

    def _quintic(self, ts, ease):
        if ease == "in":
            ts = ts * ts * ts * ts * ts
        elif ease == "out":
            ts = (ts - 1) * (ts - 1) * (ts - 1) * (ts - 1) * (ts - 1) + 1
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = 16 * ts[index] * ts[index] * ts[index] * ts[index] * ts[index]
            ts[~index] = (2 * ts[~index]) - 2
            ts[~index] = (
                0.5 * ts[~index] * ts[~index] * ts[~index] * ts[~index] * ts[~index] + 1
            )
        return ts

    def _sine(self, ts, ease):
        if ease == "in":
            ts = np.sin((ts - 1) * np.pi / 2) + 1
        elif ease == "out":
            ts = np.sin(ts * np.pi / 2)
        elif ease == "in_out":
            ts = 0.5 * (1 - np.cos(ts * np.pi))
        return ts

    def _circular(self, ts, ease):
        if ease == "in":
            ts = 1 - np.sqrt(1 - (ts * ts))
        elif ease == "out":
            ts = np.sqrt((2 - ts) * ts)
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = 0.5 * (1 - np.sqrt(1 - 4 * (ts[index] * ts[index])))
            ts[~index] = 0.5 * (
                np.sqrt(-((2 * ts[~index]) - 3) * ((2 * ts[~index]) - 1)) + 1
            )
        return ts

    def _exponential(self, ts, ease):
        if ease == "in":
            index = ts != 0
            ts[~index] = 0
            ts[index] = np.power(2, 10 * (ts[index] - 1))
        elif ease == "out":
            index = ts != 1
            ts[~index] = 1
            ts[index] = 1 - np.power(2, -10 * ts[index])
        elif ease == "in_out":
            index0 = (ts != 0) & (ts < 0.5) & (ts != 1)
            index1 = (ts != 0) & (ts >= 0.5) & (ts != 1)
            ts[index0] = 0.5 * np.power(2, (20 * ts[index0]) - 10)
            ts[index1] = -0.5 * np.power(2, (-20 * ts[index1]) + 10) + 1
        return ts

    def _elastic(self, ts, ease):
        if ease == "in":
            ts = np.sin(13 * np.pi / 2 * ts) * np.power(2, 10 * (ts - 1))
        elif ease == "out":
            ts = np.sin(-13 * np.pi / 2 * (ts + 1)) * np.power(2, -10 * ts) + 1
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = (
                0.5
                * np.sin(13 * np.pi / 2 * (2 * ts[index]))
                * np.power(2, 10 * ((2 * ts[index]) - 1))
            )
            ts[~index] = 0.5 * (
                np.sin(-13 * np.pi / 2 * ((2 * ts[~index] - 1) + 1))
                * np.power(2, -10 * (2 * ts[~index] - 1))
                + 2
            )
        return ts

    def _back(self, ts, ease):
        if ease == "in":
            ts = ts * ts * ts - ts * np.sin(ts * np.pi)
        elif ease == "out":
            ts = 1 - ts
            ts = 1 - (ts * ts * ts - ts * np.sin(ts * np.pi))
        elif ease == "in_out":
            index = ts < 0.5
            ts[index] = 2 * ts[index]
            ts[index] = 0.5 * (
                ts[index] * ts[index] * ts[index]
                - ts[index] * np.sin(ts[index] * np.pi)
            )
            ts[~index] = 1 - (2 * ts[~index] - 1)
            ts[~index] = (
                0.5
                * (
                    1
                    - (
                        ts[~index] * ts[~index] * ts[~index]
                        - ts[~index] * np.sin(ts[~index] * np.pi)
                    )
                )
                + 0.5
            )
        return ts

    def _bounce(self, ts, ease):
        index = ts < 0.5
        if ease == "in":
            ts = 1 - ts
        elif ease == "in_out":
            ts[index] = 1 - (ts[index] * 2)
            ts[~index] = ts[~index] * 2 - 1
        index0 = ts < 4 / 11
        index1 = (ts < 8 / 11) & ~index0
        index2 = (ts < 9 / 10) & ~index1 & ~index0
        index3 = ts >= 9 / 10
        ts[index0] = 121 * ts[index0] * ts[index0] / 16
        ts[index1] = (
            (363 / 40.0 * ts[index1] * ts[index1]) - (99 / 10.0 * ts[index1]) + 17 / 5.0
        )
        ts[index2] = (
            (4356 / 361.0 * ts[index2] * ts[index2])
            - (35442 / 1805.0 * ts[index2])
            + 16061 / 1805.0
        )
        ts[index3] = (
            (54 / 5.0 * ts[index3] * ts[index3])
            - (513 / 25.0 * ts[index3])
            + 268 / 25.0
        )
        if ease == "in":
            ts = 1 - ts
        elif ease == "out":
            pass
        elif ease == "in_out":
            ts[index] = 0.5 * (1 - ts[index])
            ts[~index] = 0.5 * ts[~index] + 0.5
        return ts
