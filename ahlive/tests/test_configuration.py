from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ahlive as ah

TYPES = [list, tuple, np.array, pd.Series, xr.DataArray]

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


@pytest.fixture
def canvas1_params():
    return dict(durations=1, watermark="watermark1")


@pytest.fixture
def canvas2_params():
    return dict(durations=2, watermark="watermark2", revert="boomerang")


@pytest.fixture
def subplot1_params():
    return dict(grid=True, legend=True)


@pytest.fixture
def subplot2_params():
    return dict(grid=False, legend=False, preset="trail")


@pytest.fixture
def geo1_params():
    return dict(worldwide=True, projection="Robinson")


@pytest.fixture
def geo2_params():
    return dict(worldwide=False, projection="PlateCarree", crs="PlateCarree")


@pytest.fixture
def label1_params():
    return dict(xlabel="xlabel1", ylabel="ylabel1")


@pytest.fixture
def label2_params():
    return dict(xlabel="xlabel2", ylabel="ylabel2", title="title2")


@pytest.fixture
def ah_array1():
    return ah.Array([0], [1], label="1")


@pytest.fixture
def ah_array2():
    return ah.Array([0, 1], [2, 3], label="2")
