import numpy as np
import pytest

import ahlive as ah
from ahlive.configuration import CONFIGURABLES, CONFIGURABLES_KWDS, PARAMS, VARS
from ahlive.join import _drop_state
from ahlive.tests.test_configuration import (  # noqa: F401
    DIRECTIONS,
    JOINS,
    ah_array1,
    ah_array2,
    ah_array3,
    canvas1_params,
    canvas2_params,
    geo1_params,
    geo2_params,
    label1_params,
    label2_params,
    subplot1_params,
    subplot2_params,
)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("join", JOINS)
def test_match_states(direction, join):
    ah_objs = [ah_array1, ah_array2]
    if join == "cascade":
        num_states = 3
    elif direction == "backward":
        ah_objs = ah_objs[::-1]
        num_states = 2
    else:
        num_states = 2

    ah_obj = ah.merge(ah_objs, join=join)
    for ds in ah_obj.data.values():
        assert len(ds["state"]) == num_states


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("join", JOINS[:-1])
def test_shift_items(direction, join):
    ah_objs = [ah_array1, ah_array1]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs, join=join)
    assert len(ah_obj.data[1, 1]["item"]) == 2


def test_drop_state():
    ds = ah_array1.data[1, 1].squeeze("state").expand_dims("state")
    dropped_ds = _drop_state(ds)
    for var in VARS["stateless"]:
        if var in dropped_ds:
            assert "state" not in dropped_ds[var]


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("join", JOINS)
def test_propagate_params(direction, join):
    x = [0, 1]
    y = [2, 3]
    args = x, y
    a = ah.Array(
        *args,
        **canvas1_params,
        **subplot1_params,
        **geo1_params,
        **label1_params,
    )
    b = ah.Reference(
        *args,
        **canvas2_params,
    )
    c = ah.Array(
        *args,
        **canvas2_params,
        **subplot2_params,
        **geo2_params,
        **label2_params,
    )
    ah_objs = [a, b, c]

    all1_params = {
        **canvas1_params,
        **subplot1_params,
        **geo1_params,
        **label1_params,
    }
    all2_params = {
        **canvas2_params,
        **subplot2_params,
        **geo2_params,
        **label2_params,
    }
    if direction == "backward":
        all1_params, all2_params = all2_params, all1_params
        ah_objs = ah_objs[::-1]

    ah_obj = ah.merge(ah_objs, join=join)
    for param in all1_params:
        configurable = PARAMS[param]
        key = f"{configurable}_kwds"
        method_key = CONFIGURABLES_KWDS[configurable][param]
        actual = ah_obj[1, 1].attrs[key][method_key]
        expected = all1_params[param]
        assert actual == expected

    for param in all2_params:
        configurable = PARAMS[param]
        key = f"{configurable}_kwds"
        method_key = CONFIGURABLES_KWDS[configurable][param]
        expected = all2_params[param]
        if join == "layout" and param not in CONFIGURABLES["canvas"]:
            actual = ah_obj[1, 3].attrs[key][method_key]
            assert actual == expected
        else:
            actual = ah_obj[1, 1].attrs[key][method_key]
            if param in all1_params:
                assert actual != expected
            else:
                assert actual == expected


def test_overlay():
    ds_function = ah.overlay([ah_array1, ah_array2, ah_array3]).data[1, 1]
    assert (ds_function["state"] == [1, 2, 3]).all()
    assert (ds_function["item"] == [1, 2, 3]).all()
    assert (ds_function.isel(item=0)["x"] == [0, 0, 0]).all()
    assert (ds_function.isel(item=1)["x"] == [0, 1, 1]).all()
    assert (ds_function.isel(item=2)["x"] == [0, 1, 2]).all()

    assert (ds_function.isel(item=0)["y"] == [1, 1, 1]).all()
    assert (ds_function.isel(item=1)["y"] == [2, 3, 3]).all()
    assert (ds_function.isel(item=2)["y"] == [2, 3, 4]).all()

    ds_operator = (ah_array1 * ah_array2 * ah_array3).data[1, 1]
    ds_method = ah_array1.overlay(ah_array2).overlay(ah_array3).data[1, 1]
    assert ds_function.equals(ds_operator)
    assert ds_function.equals(ds_method)


def test_cascade():
    ds_function = ah.cascade([ah_array1, ah_array2, ah_array3]).data[1, 1]
    assert (ds_function["state"] == [1, 2, 3, 4, 5, 6]).all()
    assert (ds_function["item"] == [1, 2, 3]).all()

    assert (ds_function.isel(item=0)["x"] == [0] * 6).all()
    assert np.isclose(
        ds_function.isel(item=1)["x"], [np.nan, 0, 1, 1, 1, 1], equal_nan=True
    ).all()
    assert np.isclose(
        ds_function.isel(item=2)["x"], [np.nan, np.nan, np.nan, 0, 1, 2], equal_nan=True
    ).all()

    assert (ds_function.isel(item=0)["y"] == [1] * 6).all()
    assert np.isclose(
        ds_function.isel(item=1)["y"], [np.nan, 2, 3, 3, 3, 3], equal_nan=True
    ).all()
    assert np.isclose(
        ds_function.isel(item=2)["y"], [np.nan, np.nan, np.nan, 2, 3, 4], equal_nan=True
    ).all()

    ds_operator = (ah_array1 - ah_array2 - ah_array3).data[1, 1]
    ds_method = ah_array1.cascade(ah_array2).cascade(ah_array3).data[1, 1]
    assert ds_function.equals(ds_operator)
    assert ds_function.equals(ds_method)


def test_layout():
    data_function = ah.layout([ah_array1, ah_array2, ah_array3])

    ds1 = data_function[1, 1]
    assert (ds1["state"] == [1, 2, 3]).all()
    assert (ds1["item"] == [1]).all()
    assert (ds1["x"] == [0, 0, 0]).all()
    assert (ds1["y"] == [1, 1, 1]).all()

    ds2 = data_function[1, 2]
    assert (ds2["state"] == [1, 2, 3]).all()
    assert (ds2["item"] == [1]).all()
    assert (ds2["x"] == [0, 1, 1]).all()
    assert (ds2["y"] == [2, 3, 3]).all()

    ds2 = data_function[1, 3]
    assert (ds2["state"] == [1, 2, 3]).all()
    assert (ds2["item"] == [1]).all()
    assert (ds2["x"] == [0, 1, 2]).all()
    assert (ds2["y"] == [2, 3, 4]).all()

    data_operator = ah_array1 + ah_array2 + ah_array3
    data_method = ah_array1.layout(ah_array2).layout(ah_array3)
    assert data_function.equals(data_operator)
    assert data_function.equals(data_method)


def test_slide():
    ds_function = ah.slide([ah_array1, ah_array2, ah_array3]).data[1, 1]
    assert (ds_function["state"] == [1, 2, 3, 4, 5]).all()
    assert (ds_function["item"] == [1, 2, 3]).all()

    assert (ds_function.isel(item=0)["x"] == [0] * 5).all()
    assert np.isclose(
        ds_function.isel(item=1)["x"], [0.0, 0.0, 1.0, 1.0, 1.0], equal_nan=True
    ).all()
    assert np.isclose(
        ds_function.isel(item=2)["x"], [0.0, 0.0, 0.0, 1.0, 2.0], equal_nan=True
    ).all()

    assert (ds_function.isel(item=0)["y"] == [1] * 5).all()
    assert np.isclose(
        ds_function.isel(item=1)["y"], [2, 2, 3, 3, 3], equal_nan=True
    ).all()
    assert np.isclose(
        ds_function.isel(item=2)["y"], [2, 2, 2, 3, 4], equal_nan=True
    ).all()

    (ah_array1 // ah_array2 // ah_array3).data[1, 1]
    ah_array1.slide(ah_array2).slide(ah_array3).data[1, 1]


def test_stagger():
    ds_function = ah.stagger([ah_array1, ah_array2, ah_array3]).data[1, 1]
    assert (ds_function["state"] == [1, 2, 3, 4, 5, 6]).all()
    assert (ds_function["item"] == [1, 2, 3]).all()

    assert (ds_function.isel(item=0)["x"] == [0] * 6).all()
    assert np.isclose(
        ds_function.isel(item=1)["x"], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], equal_nan=True
    ).all()
    assert np.isclose(
        ds_function.isel(item=2)["x"], [0.0, 0.0, 0.0, 1.0, 1.0, 2.0], equal_nan=True
    ).all()

    assert (ds_function.isel(item=0)["y"] == [1] * 6).all()
    assert np.isclose(
        ds_function.isel(item=1)["y"], [2, 2, 3, 3, 3, 3], equal_nan=True
    ).all()
    assert np.isclose(
        ds_function.isel(item=2)["y"], [2, 2, 2, 3, 3, 4], equal_nan=True
    ).all()

    (ah_array1**ah_array2**ah_array3).data[1, 1]
    ah_array1.stagger(ah_array2).stagger(ah_array3).data[1, 1]
