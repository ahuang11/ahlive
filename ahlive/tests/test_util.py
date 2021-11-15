from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ahlive import util
from ahlive.tests.test_configuration import CONTAINERS, TYPES, TYPES_GROUPS


def assert_types(ah_obj):
    """assert types are expected"""
    assert isinstance(ah_obj.data, dict)
    assert all(isinstance(value, xr.Dataset) for value in ah_obj.data.values())
    assert all(isinstance(key, tuple) for key in ah_obj.data.keys())


def assert_values(ds, var_dict):
    """assert values are correctly serialized"""
    for var in var_dict:
        try:
            actual = np.array(ds[var]).ravel()
        except KeyError:
            actual = np.array(ds.attrs["plot_kwds"][var])
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

        if not util.is_datetime(expect):
            try:
                expect = expect.astype(float)
            except ValueError:
                pass

        print(var, actual, expect)
        try:
            assert np.allclose(actual, expect, equal_nan=True)
        except TypeError:
            assert np.all(actual == expect)


def assert_attrs(ds, configurables):
    """assert all keywords are initiated in attrs"""
    for configurable in configurables:
        for key in configurables[configurable]:
            assert f"{key}_kwds" in ds.attrs


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("group", TYPES_GROUPS)
def test_is_dtype(container, group):
    types = TYPES_GROUPS[group]
    is_func = getattr(util, f"is_{group}")
    for type_ in types:
        scalar = TYPES[type_]
        assert is_func(scalar)

        contained = container(util.to_1d(scalar))
        assert is_func(contained)
