from collections.abc import Iterable

import numpy as np
import pandas as pd
import param
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, rgb2hex

from .configuration import DEFAULTS, EASES, INTERPS, PRECEDENCES, REVERTS
from .util import is_str, length


class Easing(param.Parameterized):

    interp = param.ClassSelector(
        default=None,
        class_=Iterable,
        doc=f"Interpolation method; {INTERPS}",
        precedence=PRECEDENCES["interp"],
    )
    ease = param.ClassSelector(
        default="in_out",
        class_=Iterable,
        doc=f"Type of easing; {EASES}",
        precedence=PRECEDENCES["interp"],
    )
    frames = param.Integer(
        default=None,
        bounds=(1, None),
        doc="Number of frames between each base state",
        precedence=PRECEDENCES["interp"],
    )
    revert = param.ObjectSelector(
        default=None,
        objects=REVERTS,
        doc="Method for reverting to the initial state; "
        "boomerang finds the shortest path to the initial state, "
        "traceback backtracks the original path to the initial state, and "
        "rollback is like traceback, but disregards the "
        "original's path durations",
        precedence=PRECEDENCES["interp"],
    )

    num_states = param.Integer(doc="Number of states", **DEFAULTS["num_kwds"])
    num_steps = param.Integer(
        doc="Number of frames between each base state", **DEFAULTS["num_kwds"]
    )

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def interpolate(self, da, name=""):
        interp = self.interp
        if interp is None:
            if length(da) > 4:
                interp = "linear"
            else:
                interp = "cubic"
        ease = self.ease

        da_origin = da.copy()

        is_xarray = isinstance(da, xr.DataArray)
        is_bar = False
        if is_xarray:
            if "state" not in da.dims:
                return da_origin
            da, name, dims, coords, is_bar, is_errorbar_morph = self._prep_xarray(da)

        array = self._prep_array(da)

        num_items, num_states, num_steps, num_result = self._calc_shapes(array)
        if (num_steps == 1 or num_states == 1) and self.revert is None:
            return da_origin

        steps = np.linspace(0, 1, num_steps)
        interp_args = (steps, interp, ease, num_states, num_steps, num_items)
        array_dtype = array.dtype
        if name in ["duration", "remark", "xerr", "yerr"] and not is_errorbar_morph:
            result = self._interp_first(
                array, num_states, num_steps, num_items, num_result, name
            )
        elif interp == "fill" or name.endswith(
            ("zoom", "discrete_trail", "morph_trail", "tick_label", "bar_label")
        ):
            result = self._interp_fill(array, num_states, num_steps, name)
        elif np.issubdtype(array_dtype, np.datetime64):
            result = self._interp_time(array, pd.to_datetime, *interp_args)
        elif np.issubdtype(array_dtype, np.timedelta64):
            result = self._interp_time(array, pd.to_timedelta, *interp_args)
        elif np.issubdtype(array_dtype, np.number) and not is_bar:
            if name == "central_longitude":
                interp = "linear"
            result = self._interp_numeric(array, *interp_args)
        elif name in "c":  # must be after number
            result = self._interp_color(array, num_result)
        elif is_bar:
            result = self._interp_fill(array, num_states, num_steps, name)
        else:  # str
            result = self._interp_text(array, num_states, num_steps, num_result)

        if self.revert in ["traceback", "rollback"]:
            result = self._apply_revert(result, name)

        # twitter ignores the last frame, so this is a hack
        # to make it drop a useless frame rather than the actual
        # final frame with a duration
        result = np.hstack([result, result[:, -1:]])
        if name == "duration":
            result[:, -1] = 0

        if is_xarray:
            result = self._rebuild_da(result, da, dims, coords)

        return result

    def _prep_xarray(self, da):
        name = da.name
        for item_dim in da.dims:
            if "item" in item_dim:
                if "batch" in da.dims:
                    da = da.transpose(item_dim, "batch", "state", ...)
                else:
                    da = da.transpose(item_dim, "state", ...)
                break

        dims = da.dims
        if da.ndim > 2:  # more than (item, state)
            if "grid_item" in dims:
                da = da.stack({"stacked": ["grid_item", "grid_y", "grid_x"]})
            elif "batch" in dims:
                da = da.stack({"stacked": [item_dim, "batch"]})
            da = da.transpose("stacked", "state")
        coords = da.drop_vars("state", errors="ignore").coords
        is_bar = da.attrs.get("is_bar")
        is_errorbar_morph = da.attrs.get("is_errorbar_morph")
        return da, name, dims, coords, is_bar, is_errorbar_morph

    def _prep_array(self, da):
        array = np.array(da)

        if array.ndim == 1:
            array = array[np.newaxis, :]

        if self.revert == "boomerang":
            array = np.hstack([array, array[:, :1]])

        return array

    def _calc_shapes(self, array):
        num_items, num_states = array.shape

        if self.frames is None:
            if num_states < 10:
                num_steps = int(np.ceil(60 / num_states))
            else:
                num_steps = int(np.ceil(120 / num_states))
        else:
            num_steps = self.frames

        with param.edit_constant(self):
            self.num_steps = num_steps

        num_result = (num_states - 1) * num_steps
        return num_items, num_states, num_steps, num_result

    def _apply_revert(self, result, name):
        if result.ndim == 1:
            result_back = result[::-1]
        else:
            result_back = result[:, ::-1]
        if name == "duration" and self.revert == "rollback":
            result_back = np.repeat(1 / 45, result_back.shape[-1])[np.newaxis, :]
        result = np.hstack([result, result_back])
        return result

    def _rebuild_da(self, result, da, dims, coords):
        if len(dims) == 1:
            result = result.squeeze()
        result = xr.DataArray(
            result,
            dims=da.dims,
            coords=coords,
            name=da.name,
            attrs=da.attrs,
        )
        if "stacked" in result.dims:
            result = result.unstack().transpose(*dims)
        return result

    def _interp_first(self, array, num_states, num_steps, num_items, num_result, name):
        if is_str(array):
            fill = ""
            dtype = np.object
        else:
            fill = 0.0
            dtype = None
        result = np.full((num_items, num_result), fill, dtype=dtype)
        indices = np.arange(num_states) * num_steps
        indices[-1] -= 1
        result[:, indices] = array  # (1, num_states)
        return result

    def _interp_fill(self, array, num_states, num_steps, name):
        indices = np.arange(num_states * num_steps - num_steps)
        result = (
            pd.DataFrame(
                array,
                columns=np.arange(0, num_states * num_steps, num_steps),
            )
            .T.reindex(indices)
            .T
        )
        if not name.endswith("discrete_trail"):
            result = result.ffill(axis=1).fillna("").values
            result[:, -1] = array[:, -1]
        else:
            result = result.values
        return result

    def _interp_color(self, array, num_result):
        results = []
        for colors in array:  # item, state
            cmap = LinearSegmentedColormap.from_list("eased", colors, N=num_result)
            results.append([rgb2hex(rgb) for rgb in cmap(np.arange(num_result))])
        result = np.array(results)
        return result

    def _interp_text(self, array, num_states, num_steps, num_result):
        result = np.repeat(array, num_steps, axis=-1)
        num_roll = -int(np.ceil(num_steps / num_states * 2))
        if num_states > 2:
            result = np.roll(result, num_roll, axis=-1)
            result = result[:, :num_result]
        else:
            half_way = int(num_result / 2)
            result = result[:, half_way:-half_way]
            if num_steps % 2 != 0:
                result = result[:, :-1]
        return result

    def _interp_time(
        self, array, conversion, steps, interp, ease, num_states, num_steps, num_items
    ):
        array = array.astype(float)
        result = self._interp_numeric(
            array, steps, interp, ease, num_states, num_steps, num_items
        )
        result = conversion(result.ravel()).values
        result = result.reshape(num_items, -1)
        return result

    def _interp_numeric(
        self, array, steps, interp, ease, num_states, num_steps, num_items
    ):
        init = np.repeat(array[:, :-1], num_steps, axis=-1)
        init_nans = np.isnan(init)
        init[init_nans] = 0  # temporarily fill the nans
        stop = np.repeat(array[:, 1:], num_steps, axis=-1)
        stop_nans = np.isnan(stop)
        tiled_steps = np.tile(steps, (num_states - 1) * num_items).reshape(
            num_items, -1
        )
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
