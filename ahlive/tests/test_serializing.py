from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ahlive as ah

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

        if isinstance(expect[0], (datetime, pd.Timestamp)):
            expect = pd.to_datetime(expect).values

        try:
            assert np.allclose(actual, expect, equal_nan=True)
        except TypeError:
            assert np.all(actual == expect)


def assert_attrs(ds, configurables):
    """assert all keywords are initiated in attrs"""
    for configurable in configurables:
        for key in configurables[configurable]:
            assert f"{key}_kwds" in ds.attrs


@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("x", XS)
@pytest.mark.parametrize("y", YS)
def test_ah_array(type_, x, y):
    x_iterable = type_(x)
    y_iterable = type_(y)
    ah_array = ah.Array(
        x_iterable, y_iterable, s=y_iterable, label="test", frames=2)
    assert_types(ah_array)

    for ds in ah_array.data.values():
        var_dict = {
            "x": x_iterable,
            "y": y_iterable,
            "s": y_iterable,
            "label": "test",
        }
        assert_values(ds, var_dict)

    configurables = ah.CONFIGURABLES.copy()
    configurables.pop("grid")
    assert_attrs(ds, configurables)

    ah_array.render()
