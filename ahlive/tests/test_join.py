import pytest

import ahlive as ah
from ahlive.configuration import CONFIGURABLES, CONFIGURABLES_KWDS, PARAMS, VARS
from ahlive.tests.test_configuration import (  # noqa: F401
    DIRECTIONS,
    JOINS,
    ah_array1,
    ah_array2,
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
def test_match_states(ah_array1, ah_array2, direction, join):
    ah_objs = [ah_array1, ah_array2]
    if join == "cascade":
        num_states = 3
    elif direction == "backward":
        ah_objs = ah_objs[::-1]
        num_states = 2
    else:
        num_states = 1

    ah_obj = ah.merge(ah_objs, join=join)
    for ds in ah_obj.data.values():
        assert len(ds["state"]) == num_states


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("join", JOINS[:-1])
def test_shift_items(ah_array1, direction, join):
    ah_objs = [ah_array1, ah_array1]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs, join=join)
    assert len(ah_obj.data[1, 1]["item"]) == 2


def test_drop_state(ah_array1):
    ds = ah_array1.data[1, 1].squeeze("state").expand_dims("state")
    dropped_ds = ah.Array._drop_state(ds)
    for var in VARS["stateless"]:
        if var in dropped_ds:
            assert "state" not in dropped_ds[var]


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("join", JOINS)
def test_propagate_params(
    direction,
    join,
    canvas1_params,
    canvas2_params,
    subplot1_params,
    subplot2_params,
    geo1_params,
    geo2_params,
    label1_params,
    label2_params,
):
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


def test_mul(ah_array1, ah_array2):
    ah_obj = ah_array1 * ah_array2
    assert len(ah_obj[1, 1]["item"]) == 2
    assert len(ah_obj[1, 1]["state"]) == 1


def test_rmul(ah_array1, ah_array2):
    ah_obj = ah_array2 * ah_array1
    assert len(ah_obj[1, 1]["item"]) == 2
    assert len(ah_obj[1, 1]["state"]) == 2


def test_floordiv(ah_array1, ah_array2):
    ah_obj = ah_array1 / ah_array2
    assert len(ah_obj[1, 1]["item"]) == 1
    assert len(ah_obj[1, 1]["state"]) == 1
    assert len(ah_obj[2, 1]["item"]) == 1
    assert len(ah_obj[2, 1]["state"]) == 1


def test_truerdiv(ah_array1, ah_array2):
    ah_obj = ah_array1 // ah_array2
    assert len(ah_obj[1, 1]["item"]) == 1
    assert len(ah_obj[1, 1]["state"]) == 1
    assert len(ah_obj[2, 1]["item"]) == 1
    assert len(ah_obj[2, 1]["state"]) == 1


def test_add(ah_array1, ah_array2):
    ah_obj = ah_array1 + ah_array2
    assert len(ah_obj[1, 1]["item"]) == 1
    assert len(ah_obj[1, 1]["state"]) == 1
    assert len(ah_obj[1, 2]["item"]) == 1
    assert len(ah_obj[1, 2]["state"]) == 1


def test_radd(ah_array1, ah_array2):
    ah_obj = ah_array2 + ah_array1
    assert len(ah_obj[1, 1]["item"]) == 1
    assert len(ah_obj[1, 1]["state"]) == 2
    assert len(ah_obj[1, 2]["item"]) == 1
    assert len(ah_obj[1, 2]["state"]) == 2


def test_sub(ah_array1, ah_array2):
    ah_obj = ah_array1 - ah_array2
    assert len(ah_obj[1, 1]["item"]) == 2
    assert len(ah_obj[1, 1]["state"]) == 3


def test_rsub(ah_array1, ah_array2):
    ah_obj = ah_array2 - ah_array1
    assert len(ah_obj[1, 1]["item"]) == 2
    assert len(ah_obj[1, 1]["state"]) == 3
