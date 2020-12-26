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
LABELS.append(['x', 'x'])
LABELS.append(['x', 'y'])
