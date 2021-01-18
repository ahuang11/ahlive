from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TYPES = [list, tuple, np.array, pd.Series, xr.DataArray]
XS = []
XS.append([0, 0])
XS.append([0, 1])
XS.append([-1])
XS.append(["1", "2"])
XS.append([datetime(2017, 1, 2), datetime(2017, 1, 3)])
XS.append(pd.date_range("2017-02-01", "2017-02-02"))
YS = []
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
