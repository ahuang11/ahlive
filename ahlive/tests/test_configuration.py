from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import ahlive as ah

CONTAINERS = [list, tuple, np.array, pd.Series, xr.DataArray]

XS = []
XS.append(0)
XS.append([0, 0])
XS.append([0, 1])
XS.append([-1])
XS.append(["1", "2"])
XS.append([datetime(2017, 1, 2), datetime(2017, 1, 3)])
XS.append(pd.date_range("2017-02-01", "2017-02-02"))

YS = []
YS.append(0)
YS.append([0, 1])
YS.append([-1])
YS.append([0.0, np.nan])

LABELS = []
LABELS.append(["x", "x"])
LABELS.append(["x", "y"])

GRID_XS = []
GRID_XS.append([0, 1, 2, 3])
GRID_XS.append([0, 1, np.nan, 3])

GRID_YS = []
GRID_YS.append([0, 1, 2, 3])
GRID_YS.append([0, 1, 2, np.nan])

GRID_CS = []
GRID_CS.append(np.random.rand(2, 4, 4))
GRID_CS.append(GRID_CS[0].copy())
GRID_CS[0][0] = np.nan

GRID_LABELS = []
GRID_LABELS.append(["a", "a"])
GRID_LABELS.append(["a", "b"])

REF_X0S = []
REF_X0S.append(None)
REF_X0S.append(0)
REF_X0S.append([0])
REF_X0S.append(["1", "2", "3"])
REF_X0S.append([0, 1, 2])

REF_Y0S = []
REF_Y0S.append(None)
REF_Y0S.append(0)
REF_Y0S.append([0])
REF_Y0S.append([0, 1, 2])

REF_X1S = []
REF_X1S.append(None)
REF_X1S.append(1)
REF_X1S.append([1])
REF_X1S.append([1, 2, 3])

REF_Y1S = []
REF_Y1S.append(None)
REF_Y1S.append(1)
REF_Y1S.append([1])
REF_Y1S.append([1, 2, 3])

DIRECTIONS = ["forward", "backward"]
JOINS = ["overlay", "cascade", "layout"]

TYPES = {
    # numeric
    "int": 0,
    "float": 0.0,
    "nan": np.nan,
    "inf": np.inf,
    "str": "a",
    "str_int": "0",
    "str_float": "0.",
    "pydatetime": datetime(2021, 1, 1),
    "datetime64": np.datetime64("2021"),
    "pdtimestamp": pd.Timestamp("2021"),
    "pytimedelta": timedelta(0),
    "timedelta64": np.timedelta64(0, "D"),
    "pdtimedelta": pd.Timedelta(0),
}
TYPES_GROUPS = {
    "numeric": ["int", "float", "nan", "inf"],
    "str": ["str", "str_int", "str_float"],
    "datetime": ["pydatetime", "datetime64", "pdtimestamp"],
    "timedelta": ["pytimedelta", "timedelta64", "pdtimedelta"],
}


canvas1_params = dict(durations=1, watermark="watermark1")
canvas2_params = dict(durations=2, watermark="watermark2", revert="boomerang")
subplot1_params = dict(grid=True, legend=True)
subplot2_params = dict(grid=False, legend=False, preset="trail")
geo1_params = dict(worldwide=True, projection="Robinson")
geo2_params = dict(worldwide=False, projection="PlateCarree", crs="PlateCarree")
label1_params = dict(xlabel="xlabel1", ylabel="ylabel1")
label2_params = dict(xlabel="xlabel2", ylabel="ylabel2", title="title2")
ah_array1 = ah.Array([0], [1], label="1")
ah_array2 = ah.Array([0, 1], [2, 3], label="2")
ah_array3 = ah.Array([0, 1, 2], [2, 3, 4], label="3")
