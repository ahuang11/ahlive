from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from ahlive.util import is_datetime


def assert_types(ah_obj):
    """assert types are expected"""
    assert isinstance(ah_obj.data, dict)
    assert all(isinstance(value, xr.Dataset) for value in ah_obj.data.values())
    assert all(isinstance(key, tuple) for key in ah_obj.data.keys())


def assert_values(ds, var_dict):
    """assert values are correctly serialized"""
    for var in var_dict:
        actual = np.array(ds[var]).ravel()
        expect = np.array(var_dict[var]).ravel()

        actual = actual[~pd.isnull(actual)]
        expect = expect[~pd.isnull(expect)]

        if len(actual) == 0 and len(expect) == 0:
            return

        if isinstance(expect[0], (datetime, pd.Timestamp)):
            expect = pd.to_datetime(expect).values

        if actual.shape != expect.shape:
            actual = np.unique(actual)
            expect = np.unique(expect)

        if not is_datetime(expect):
            try:
                expect = expect.astype(float)
            except ValueError:
                pass

        try:
            assert np.allclose(actual, expect, equal_nan=True)
        except TypeError:
            assert np.all(actual == expect)


def assert_attrs(ds, configurables):
    """assert all keywords are initiated in attrs"""
    for configurable in configurables:
        for key in configurables[configurable]:
            assert f"{key}_kwds" in ds.attrs
