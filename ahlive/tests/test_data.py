from collections.abc import ItemsView, KeysView, ValuesView

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ahlive as ah
from ahlive.configuration import CONFIGURABLES, DEFAULTS, ITEMS, OPTIONS, VARS
from ahlive.tests.test_configuration import (  # noqa: F401
    CONTAINERS,
    DIRECTIONS,
    GRID_CS,
    GRID_LABELS,
    GRID_XS,
    GRID_YS,
    LABELS,
    REF_X0S,
    REF_X1S,
    REF_Y0S,
    REF_Y1S,
    XS,
    YS,
    ah_array1,
    ah_array2,
)
from ahlive.tests.test_util import assert_attrs, assert_types, assert_values


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("x", XS)
@pytest.mark.parametrize("y", YS)
def test_ah_array(container, x, y):
    x_iterable = container(x) if isinstance(x, list) else x
    y_iterable = container(y) if isinstance(y, list) else y
    ah_array = ah.Array(x_iterable, y_iterable, s=y_iterable, label="test", frames=2)
    assert_types(ah_array)

    for ds in ah_array.data.values():
        var_dict = {
            "x": x_iterable,
            "y": y_iterable,
            "s": y_iterable,
            "label": "test",
        }
        assert_values(ds, var_dict)

        configurables = CONFIGURABLES.copy()
        configurables.pop("grid")
        assert_attrs(ds, configurables)

    ah_array.finalize()


@pytest.mark.parametrize("x", XS)
@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("join", OPTIONS["join"])
def test_ah_dataframe(x, y, label, join):
    df = pd.DataFrame(
        {"x": np.array(x).squeeze(), "y": np.array(y).squeeze(), "label": label}
    )
    ah_df = ah.DataFrame(df, "x", "y", s="y", label="label", join=join, frames=2)
    assert_types(ah_df)

    for ds in ah_df.data.values():
        sub_df = df.loc[df["label"].isin(ds["label"].values.ravel())]
        var_dict = {
            "x": sub_df["x"].values,
            "y": sub_df["y"].values,
            "s": sub_df["y"].values,
            "label": sub_df["label"].values,
        }

        num_labels = len(np.unique(df["label"]))
        if join in ["overlay", "cascade", "slide", "stagger"]:
            assert num_labels == len(ds["item"])
        else:
            assert 1 == len(ds["item"])

        if num_labels > 1 and join == "cascade":
            for i in range(num_labels):
                assert_values(
                    ds.isel(item=i),
                    {key: val[i] for key, val in var_dict.items()},
                )
        else:
            print(ds)
            assert_values(ds, var_dict)

        configurables = CONFIGURABLES.copy()
        configurables.pop("grid")
        assert_attrs(ds, configurables)

    ah_df.finalize()


@pytest.mark.parametrize("grid_x", GRID_XS)
@pytest.mark.parametrize("grid_y", GRID_YS)
@pytest.mark.parametrize("grid_c", GRID_CS)
def test_ah_array2d(grid_x, grid_y, grid_c):
    ah_array2d = ah.Array2D(grid_x, grid_y, grid_c, label="test", frames=2)
    assert_types(ah_array2d)

    for ds in ah_array2d.data.values():
        var_dict = {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_c": grid_c,
            "grid_label": "test",
        }

        assert 1 == len(ds["grid_item"])
        assert_values(ds, var_dict)

        configurables = CONFIGURABLES.copy()
        assert_attrs(ds, configurables)

    ah_array2d.finalize()


@pytest.mark.parametrize("grid_x", GRID_XS)
@pytest.mark.parametrize("grid_y", GRID_YS)
@pytest.mark.parametrize("grid_c", GRID_CS)
@pytest.mark.parametrize("grid_label", GRID_LABELS)
@pytest.mark.parametrize("join", OPTIONS["join"])
def test_ah_dataset(grid_x, grid_y, grid_c, grid_label, join):
    base_ds = xr.Dataset()
    base_ds["c"] = xr.DataArray(
        grid_c,
        dims=("label", "y", "x"),
        coords={"y": grid_y, "x": grid_x, "label": grid_label},
    )
    ah_dataset = ah.Dataset(base_ds, "x", "y", "c", label="label", join=join)
    assert_types(ah_dataset)

    for ds in ah_dataset.data.values():
        sub_ds = base_ds.where(
            base_ds["label"] == np.unique(ds["grid_label"]), drop=True
        )
        assert len(np.unique(sub_ds["label"])) == len(ds["grid_item"])
        configurables = CONFIGURABLES.copy()
        assert_attrs(ds, configurables)

    ah_dataset.finalize()


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("ref_x0", REF_X0S)
@pytest.mark.parametrize("ref_x1", REF_X1S)
@pytest.mark.parametrize("ref_y0", REF_Y0S)
@pytest.mark.parametrize("ref_y1", REF_Y1S)
def test_ah_reference(container, ref_x0, ref_x1, ref_y0, ref_y1):
    refs = [ref_x0, ref_x1, ref_y0, ref_y1]
    iterables = [container(ref) if isinstance(ref, list) else ref for ref in refs]
    if all(ref is None for ref in iterables):
        pytest.skip()
    elif ref_x0 is None and ref_x1 is not None:
        pytest.skip()
    elif ref_y0 is None and ref_y1 is not None:
        pytest.skip()

    ah_ref = ah.Reference(*iterables, frames=2)
    assert_types(ah_ref)

    for ds in ah_ref.data.values():
        var_dict = {
            "ref_x0": iterables[0],
            "ref_x1": iterables[1],
            "ref_y0": iterables[2],
            "ref_y1": iterables[3],
        }
        for var in var_dict.copy():
            if var not in ds:
                var_dict.pop(var)
        assert_values(ds, var_dict)

    ah_ref.finalize()


def test_keys():
    assert isinstance(ah_array1.keys(), KeysView)


def test_values():
    assert isinstance(ah_array1.values(), ValuesView)


def test_items():
    assert isinstance(ah_array1.items(), ItemsView)


def test_data():
    ah_data = ah_array1.data
    assert isinstance(ah_data, dict)
    assert isinstance(list(ah_data.keys())[0], tuple)
    assert isinstance(ah_data[1, 1], xr.Dataset)


def test_attrs():
    ah_attrs = ah_array1.attrs
    assert ah_array1[1, 1].attrs == ah_attrs
    assert isinstance(ah_attrs, dict)


@pytest.mark.parametrize("num_cols", [0, 1, 2, 3])
def test_cols(num_cols):
    ah_obj = ah_array1 + ah_array2
    if num_cols == 0:
        with pytest.raises(ValueError):
            ah_obj.cols(num_cols)
    else:
        ah_obj = ah_obj.cols(num_cols)
        ah_keys = ah_obj.keys()
        if num_cols > 1:
            assert (1, 2) in ah_keys
        else:
            assert (2, 1) in ah_keys


@pytest.mark.parametrize("preset", [None, "stacked", "race", "delta"])
@pytest.mark.parametrize("x_type", [None, "string"])
def test_config_bar_chart(preset, x_type):
    x = [1, 1, 2, 2, 2]
    y = [4, 5, 3, 8, 10]

    if x_type == "string":
        x = np.array(x).astype(str)

    df = pd.DataFrame({"x": x, "y": y})
    ah_df = ah.DataFrame(
        df, "x", "y", label="x", chart="barh", preset=preset, frames=1
    ).finalize()
    ds = ah_df[1, 1]

    if preset == "race":
        actual = ds["x"].values.ravel()
        expected = [2, 1, 1, 1, 2, 2]
        assert (actual == expected).all()
    else:
        actual = ds["x"].values.ravel()
        if preset is None:
            expected = [
                0.83333333,
                0.83333333,
                0.83333333,
                2.16666667,
                2.16666667,
                2.16666667,
            ]
        elif preset in ["stacked", "delta"]:
            expected = [1, 1, 1, 2, 2, 2]
        else:
            expected = [0, 0, 0, 1, 1, 1]
        np.testing.assert_almost_equal(actual, expected)

    for var in ["tick_label", "bar_label"]:
        if preset is None or preset == "stacked" and var == "bar_label":
            continue
        expected = [1, 1, 1, 2, 2, 2]
        actual = ds[var].values.ravel()
        assert (actual == expected).all()

    actual = ds["y"].values.ravel()
    expected = [4, 5, 5, 3, 8, 10]
    np.testing.assert_equal(actual, expected)

    if preset is not None:
        assert ds.attrs["preset_kwds"]["preset"] == preset
    elif preset == "stacked":
        expected = [0, 0, 0, 4, 5, 5]
        assert ds["bar_offset"] == expected


@pytest.mark.parametrize("chart", ["both", "scatter", "line"])
def test_config_trail_chart(chart):
    x = [0, 1, 2]
    y = [3, 4, 5]
    ah_array = (
        ah.Array(x, y, chart="scatter", preset="trail")
        .config("preset", chart=chart)
        .finalize()
    )
    ds = ah_array[1, 1]
    trail_vars = []
    if chart in ["both", "scatter"]:
        trail_vars.extend(["x_discrete_trail", "y_discrete_trail"])
    elif chart in ["both", "line"]:
        trail_vars.extend(["x_trail", "y_trail"])
    for var in trail_vars:
        assert var in ds
        if "x" in var:
            if "discrete" in var:
                assert (x[:-1] == np.unique(ds[var].dropna("state"))).all()
            else:
                assert (ds["x"].values == ds[var].values).all()
        elif "y" in var:
            if "discrete" in var:
                assert (y[:-1] == np.unique(ds[var].dropna("state"))).all()
            else:
                assert (ds["y"].values == ds[var].values).all()


@pytest.mark.parametrize("central_lon", [240, [0, 360], [0, 180, 360]])
def test_config_rotate_chart(central_lon):
    base_ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 3))
    ah_ds = (
        ah.Dataset(base_ds, "lon", "lat", "air", preset="rotate")
        .config("projection", central_longitude=central_lon)
        .finalize()
    )
    ds = ah_ds[1, 1]
    central_lons = np.array(
        [proj.proj4_params["lon_0"] for proj in ds["projection"].values]
    )
    if isinstance(central_lon, int):
        max_lon = base_ds["lon"].max().values
        assert (central_lons >= central_lon).all() & (central_lons <= max_lon).all()
    elif len(central_lons) == len(central_lon):
        assert central_lons == central_lon
    else:
        assert (central_lons >= central_lon[0]).all() & (
            central_lons <= central_lon[-1]
        ).all()


@pytest.mark.parametrize("preset", ["scan_x", "scan_y"])
def test_config_scan_chart(preset):  # TODO: improve test
    base_ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 3))
    ah_ds = ah.Dataset(base_ds, "lon", "lat", "air", preset=preset).finalize()
    assert f"grid_{preset}" in ah_ds[1, 1]


def test_config_legend_sortby():
    ah_obj = (ah_array1 * ah_array2).config("legend", sortby="y").finalize()
    ds = ah_obj[1, 1]
    assert (pd.unique(ds["label"].values.ravel()) == [2, 1]).all()
    assert ds.attrs["legend_kwds"]["show"]


@pytest.mark.parametrize("num_items", [1, 11])
def test_config_legend_show(num_items):
    ah_obj = ah.merge([ah_array1 for _ in range(num_items)]).finalize()
    ds = ah_obj[1, 1]
    assert not ds.attrs["legend_kwds"]["show"]


def test_config_grid_axes_bare():
    ah_obj = ah.Array([0, 1], [2, 3], style="bare").finalize()
    assert not ah_obj[1, 1].attrs["grid_kwds"]["b"]


@pytest.mark.parametrize("chart", ["barh", "bar", "line"])
def test_config_grid_axes(chart):
    ah_obj = ah.Array([0, 1], [2, 3], chart=chart).finalize()
    if chart == "barh":
        axis = "x"
    elif chart == "bar":
        axis = "y"
    else:
        axis = "both"
    ah_obj[1, 1].attrs["grid_kwds"]["axis"] == axis


def test_fill_null():
    ah_obj = (ah_array1 - ah_array2).finalize()
    ds = ah_obj[1, 1]
    for item in ds["item"]:
        ds_item = ds.sel(item=item)
        for var in ds_item:
            da_item = ds_item[var]
            try:
                da_nans = np.isnan(da_item.values)
            except TypeError:
                da_nans = pd.isnull(da_item.values)
                if not da_nans.all():
                    assert da_nans.sum() == 0


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("padding", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_xlim0s(direction, limit, padding):
    # TODO: test datetimes, strings
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = -2
    elif limit.startswith("follow"):
        expected = np.array([-1.0, 1, -2])
    elif limit.startswith("explore"):
        expected = np.array([-1.0, -1, -2])

    if padding is not None:
        expected -= padding
        limit = f"{limit}_{padding}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], xlim0s=limit, xmargins=0, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["xlim0"].values
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("padding", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_xlim1s(direction, limit, padding):
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = 1
    elif limit.startswith("follow"):
        expected = np.array([0.0, 1, 0])
    elif limit.startswith("explore"):
        expected = np.array([0.0, 1, 1])

    if padding is not None:
        expected += padding
        limit = f"{limit}_{padding}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], xlim1s=limit, xmargins=0, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["xlim1"].values
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("padding", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_ylim0s(direction, limit, padding):
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = -5
    elif limit.startswith("follow"):
        expected = np.array([3.0, 4, -5])
    elif limit.startswith("explore"):
        expected = np.array([3.0, 3, -5])

    if padding is not None:
        expected -= padding
        limit = f"{limit}_{padding}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], ylim0s=limit, ymargins=0, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["ylim0"].values
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("padding", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_ylim1s(direction, limit, padding):
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = 6
    elif limit.startswith("follow"):
        expected = np.array([5.0, 6, 5])
    elif limit.startswith("explore"):
        expected = np.array([5.0, 6, 6])

    if padding is not None:
        expected += padding
        limit = f"{limit}_{padding}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], ylim1s=limit, ymargins=0, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["ylim1"].values
    assert np.isclose(actual, expected).all()


def test_add_color_kwds_bar():
    x = ["a", "a", "b", "b", "b"]
    y = [4, 5, 3, 8, 10]
    ah_obj = ah.Array(x, y, chart="bar", preset="race").finalize()
    assert not ah_obj.attrs["legend_kwds"]["show"]


def test_add_color_kwds_cticks():
    cticks = [0, 5, 6, 7, 8, 9]
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], cs=[6, 7, 8], cticks=cticks).finalize()
    attrs = ah_obj.attrs
    assert attrs["cticks_kwds"]["ticks"] == cticks
    assert attrs["colorbar_kwds"]["show"]


def test_add_color_kwds_cticks_grid():
    cticks = [0, 1, 2]
    ah_obj = ah.Array2D(GRID_XS[0], GRID_YS[0], GRID_CS[0], cticks=cticks).finalize()
    attrs = ah_obj.attrs
    assert attrs["cticks_kwds"]["ticks"] == cticks


@pytest.mark.parametrize("num_colors", [3, 5])
def test_add_color_kwds_num_colors(num_colors):
    ah_obj = (
        ah.Array([0, 1, 2], [3, 4, 5], cs=[6, 7, 8])
        .config("cticks", num_colors=num_colors)
        .finalize()
    )
    attrs = ah_obj.attrs
    assert len(attrs["cticks_kwds"]["ticks"]) == num_colors + 1
    assert attrs["plot_kwds"]["vmin"] == 6
    assert attrs["plot_kwds"]["vmax"] == 8
    assert attrs["colorbar_kwds"]["show"]


def test_add_color_kwds_none():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5])
    attrs = ah_obj.attrs
    assert not attrs["colorbar_kwds"]


def test_precompute_base_ticks_numeric():
    ah_obj = ah.Array([0.01, 0.02, 1], [5, 6, 7]).finalize()
    attrs = ah_obj.attrs
    attrs["xticks_kwds"]["is_str"] = False
    attrs["base_kwds"]["xticks"] == 0.002
    attrs["xticks_kwds"]["is_datetime"] = False

    attrs["yticks_kwds"]["is_str"] = False
    attrs["base_kwds"]["yticks"] == 0.6
    attrs["yticks_kwds"]["is_datetime"] = False


def test_precompute_base_ticks_str():
    ah_obj = ah.Array(["0", "1", "2"], [5, 6, 7]).finalize()
    attrs = ah_obj.attrs
    attrs["xticks_kwds"]["is_str"] = True
    "xticks" not in attrs["base_kwds"]
    "is_datetime" not in attrs["base_kwds"]


def test_precompute_base_ticks_datetime():
    ah_obj = ah.Array(pd.date_range("2021-01-01", "2021-01-03"), [5, 6, 7]).finalize()
    attrs = ah_obj.attrs
    attrs["xticks_kwds"]["is_str"] = True
    attrs["base_kwds"]["xticks"] == np.datetime64("2021-01-01")
    attrs["xticks_kwds"]["is_datetime"] = True


def test_precompute_base_labels_scalar():
    ah_obj = ah.Array([1, 2, 3], [5, 6, 7], state_labels=[10, 10, 10]).finalize()
    attrs = ah_obj.attrs
    attrs["base_kwds"]["state_label"] = 1


def test_precompute_base_labels_numeric():
    ah_obj = ah.Array([1, 2, 3], [5, 6, 7], state_labels=[10, 11, 12]).finalize()
    attrs = ah_obj.attrs
    attrs["base_kwds"]["state"] == 1.0


def test_precompute_base_labels_datetime():
    ah_obj = ah.Array(
        [1, 2, 3],
        [5, 6, 7],
        state_labels=pd.date_range("2021-01-01", "2021-01-03"),
    ).finalize()
    attrs = ah_obj.attrs
    float(attrs["base_kwds"]["state"]) == 86400000000000


def test_precompute_base_size():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], s=[6, 7, 8]).finalize()
    attrs = ah_obj.attrs
    attrs["base_kwds"]["s"] = 7


def test_add_margins():
    ah_obj = ah.Array(
        [-1, 1],
        [2, 3],
        xlims="explore",
        ylims="fixed",
        xmargins=10,
        ymargins=10,
        frames=1,
    )
    assert ah_obj.attrs["margins_kwds"]["x"] == 10
    assert ah_obj.attrs["margins_kwds"]["y"] == 10
    ah_obj = ah_obj.finalize()
    ds = ah_obj[1, 1]
    np.testing.assert_almost_equal(ds["xlim0"], [-1, -21])
    np.testing.assert_almost_equal(ds["xlim1"], [-1, 21])
    np.testing.assert_almost_equal(ds["ylim0"], [-8, -8])
    np.testing.assert_almost_equal(ds["ylim1"], [13, 13])


def test_add_margins_datetime():
    ah_obj = ah.Array(
        pd.date_range("2017-02-01", "2017-02-02"),
        [2, 3],
        xlims="explore",
        xmargins=1,
        frames=1,
    )
    assert ah_obj.attrs["margins_kwds"]["x"] == 1
    ah_obj = ah_obj.finalize()
    ds = ah_obj[1, 1]
    assert (ds["xlim0"] == pd.to_datetime(["2017-02-01", "2017-01-31"])).all()
    assert (ds["xlim1"] == pd.to_datetime(["2017-02-01", "2017-02-03"])).all()


def test_add_margins_tuple():
    ah_obj = ah.Array(
        [-1, 1],
        [2, 3],
        xlims="explore",
        ylims="fixed",
        xmargins=(0, 10),
        ymargins=(10, 0),
        frames=1,
    )
    assert ah_obj.attrs["margins_kwds"]["x"] == (0, 10)
    assert ah_obj.attrs["margins_kwds"]["y"] == (10, 0)
    ah_obj = ah_obj.finalize()
    ds = ah_obj[1, 1]

    np.testing.assert_almost_equal(ds["xlim0"], [-1, -1])
    np.testing.assert_almost_equal(ds["xlim1"], [-1, 21])
    np.testing.assert_almost_equal(ds["ylim0"], [-8, -8])
    np.testing.assert_almost_equal(ds["ylim1"], [3, 3])


def test_add_durations_default():
    ah_obj = ah.Array([0, 1], [2, 3], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert ds["duration"].attrs["aggregate"] == "max"
    assert ds["duration"].attrs["transition_frames"] == 0.022222222222222223
    assert (ds["duration"].values == [0.5, 0, 1.05, 0]).all()


def test_add_durations_input():
    ah_obj = (
        ah.Array([0, 1], [2, 3], frames=3, durations=[0, 1])
        .config("durations", final_frame=2, transition_frames=2, aggregate="min")
        .finalize()
    )
    ds = ah_obj[1, 1]
    assert ds["duration"].attrs["aggregate"] == "min"
    assert ds["duration"].attrs["transition_frames"] == 2
    assert (ds["duration"].values == [0, 0, 3, 0]).all()


def test_interp_dataset():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert (ds["x"] == [0, 0.5, 1.0, 1.0, 1.5, 2.0, 2]).all()
    assert (ds["y"] == [3, 3.5, 4, 4, 4.5, 5, 5]).all()


def test_interp_dataset_ref():
    ah_obj = ah.Reference([0, 1, 2], [3, 4, 5], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert (ds["ref_x0"] == [0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0]).all()
    assert (ds["ref_x1"] == [3, 3.5, 4, 4, 4.5, 5, 5]).all()


def test_interp_dataset_grid():
    c0 = [[0, 1], [2, 3]]
    c1 = [[2, 3], [4, 5]]
    ah_obj = ah.Array2D([0, 1], [2, 3], [c0, c1], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert (ds["grid_c"].isel(state=0) == c0).all()
    assert (ds["grid_c"].isel(state=-1) == c1).all()


@pytest.mark.parametrize(
    "crs", [None, True, "PlateCarree", "platecarree", ccrs.PlateCarree()]
)
@pytest.mark.parametrize(
    "projection", [None, True, "Robinson", "robinson", ccrs.Robinson()]
)
def test_add_geo_transform(crs, projection):
    ah_obj = (
        ah.Array([0, 1, 2], [3, 4, 5], crs=crs, projection=projection)
        .config("projection", central_longitude=180)
        .finalize()
    )
    if crs is None and projection is None:
        pytest.skip()

    ds = ah_obj.data[1, 1]
    attrs = ah_obj.attrs
    if isinstance(projection, (str, ccrs.Robinson)):
        projection = ccrs.Robinson
    else:
        projection = ccrs.PlateCarree

    assert attrs["projection_kwds"]["central_longitude"] == 180

    for key in ITEMS["transformables"]:
        assert isinstance(attrs[key]["transform"], ccrs.PlateCarree)

    assert isinstance(ds["projection"].item(), projection)

    ds = ah_obj[1, 1]
    assert "projection" in ds


@pytest.mark.parametrize("coastline", [True, cfeature.COASTLINE])
def test_add_geo_features(coastline):
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], coastline=coastline).finalize()
    attrs = ah_obj.attrs
    assert isinstance(
        attrs["coastline_kwds"]["coastline"], cfeature.NaturalEarthFeature
    )


@pytest.mark.parametrize("xlim0s", [0, "fixed", "follow", "explore"])
@pytest.mark.parametrize("tiles", [True, "osm", "OSM"])
@pytest.mark.parametrize("zoom", [None, 1])
def test_add_geo_tiles(xlim0s, tiles, zoom):
    ah_obj = (
        ah.Array([0, 1, 2], [3, 4, 5], xlim0s=xlim0s, tiles=tiles, zoom=zoom).finalize()
    ).finalize()
    ds = ah_obj.data[1, 1]
    zoom = np.repeat(zoom or 7, len(ds["state"]))
    np.testing.assert_almost_equal(ds["zoom"].values, zoom)


@pytest.mark.parametrize("animate", ["test", "head", "tail"])
@pytest.mark.parametrize("value", [None, 5])
def test_add_animate_kwds_str(animate, value):
    if value is None:
        value = 11
    else:
        animate = f"{animate}_{value}"

    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=animate).finalize()
    attrs = ah_obj.attrs
    num_states = len(ah_obj[1, 1]["state"])
    if animate.startswith("test"):
        states = np.linspace(1, num_states, value).astype(int)
    elif animate.startswith("head"):
        states = np.arange(1, value)
    elif animate.startswith("tail"):
        states = np.arange(-value, 0, 1)

    animate_kwds = attrs["animate_kwds"]
    assert (animate_kwds["states"] == states).all()
    assert animate_kwds["fps"] == 1
    assert animate_kwds["stitch"]
    assert not animate_kwds["static"]


def test_add_animate_kwds_slice():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=slice(1, 10)).finalize()
    attrs = ah_obj.attrs
    animate_kwds = attrs["animate_kwds"]
    states = np.arange(1, 10)
    assert (animate_kwds["states"] == states).all()
    assert animate_kwds["stitch"]
    assert not animate_kwds["static"]


@pytest.mark.parametrize("animate", [True, False])
def test_add_animate_kwds_bool(animate):
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=animate).finalize()
    attrs = ah_obj.attrs
    animate_kwds = attrs["animate_kwds"]
    assert animate_kwds["states"] is None
    assert animate_kwds["stitch"] == animate
    assert not animate_kwds["static"]


def test_add_animate_kwds_int():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=1).finalize()
    attrs = ah_obj.attrs
    animate_kwds = attrs["animate_kwds"]
    assert animate_kwds["states"] == 1
    assert animate_kwds["stitch"]
    assert animate_kwds["static"]


def test_array_invert():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5], "z": [5, 6, 7]}).melt("x")
    ah_df = ah.DataFrame(df, "x", "value", label="variable")
    ah_df_inv = ah_df.invert(group="A", state_labels=["1", "2"])

    ds = ah_df[1, 1]
    ds_inv = ah_df_inv[1, 1]
    assert len(ds_inv["item"]) == len(ds["state"])
    assert len(ds_inv["state"]) == len(ds["item"])
    for var in VARS["stateless"]:
        if var in ds_inv:
            assert ds_inv[var].dims == ("item",)
    assert (ds_inv["state_label"] == ["1", "2"]).all()
    assert (ds_inv["group"] == ["A", "A", "A"]).all()
    ah_df_inv.finalize()


@pytest.mark.parametrize("key", ["label", "xlabel", "ylabel", "clabel"])
@pytest.mark.parametrize("label", ["a", 1, 1.0])
def test_labels(key, label):
    label_kwd = {key: label}
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], **label_kwd).finalize()
    if key != "label":
        sub_key = "text" if key == "clabel" else key
        assert ah_obj[1, 1].attrs[f"{key}_kwds"][sub_key] == label
    else:
        assert np.unique(ah_obj[1, 1][key].values) == [label]


@pytest.mark.parametrize("x0s", [None, 0])
@pytest.mark.parametrize("y0s", [None, 0])
@pytest.mark.parametrize("inline_locs", [None, 0])
def test_reference_method_inline_labels(x0s, y0s, inline_locs):
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5])
    reference_kwds = dict(
        x0s=x0s, y0s=y0s, inline_locs=inline_locs, inline_labels="test"
    )
    if x0s is None and y0s is None:
        with pytest.raises(ValueError):
            ah_obj.reference(**reference_kwds)
    else:
        ah_obj = ah_obj.reference(**reference_kwds).finalize()
        ds = ah_obj[1, 1]

        if x0s is not None:
            assert (ds["ref_x0"] == x0s).all()

        if y0s is not None:
            assert (ds["ref_y0"] == y0s).all()

        if inline_locs is not None:
            assert (ds["ref_inline_loc"] == inline_locs).all()

        assert (ds["ref_inline_label"] == "test").all()


@pytest.mark.parametrize("crs", [True, False])
@pytest.mark.parametrize("tiles", [True, None])
def test_geo_default_coastline(crs, tiles):
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], crs=crs, tiles=tiles).finalize()
    ds = ah_obj[1, 1]
    if tiles:
        assert len(ds.attrs["coastline_kwds"]) == 0
    elif crs:
        assert len(ds.attrs["coastline_kwds"]) == 1


@pytest.mark.parametrize("how", ["even", "uneven"])
@pytest.mark.parametrize("chart", ["line", "bar", "barh", "scatter"])
def test_config_morph_chart(how, chart):
    x = [0, 1, 2, 3, 4]
    y1 = [4, 5, 6, 7, 8]
    y2 = [8, 4, 2, 3, 4]
    ah_obj = (
        ah.Array(x, y1, preset="morph", chart=chart, group="A")
        * ah.Array(x, y2, chart=chart, group="A")
        * ah.Array(x, y2, chart=chart, group="B")
    )
    if how == "even":
        ah_obj *= ah.Array(x, y1, chart=chart, group="B")
    ah_obj = ah_obj.finalize()
    ds = ah_obj[1, 1]
    assert len(ds["item"] == 2)
    assert len(ds["batch"] == 5)
    assert len(ds["state"] == 30)
    assert (ds["group"].values == ["B", "A"]).all()


@pytest.mark.parametrize("dtype", ["numeric", "datetime"])
@pytest.mark.parametrize("chart", ["line", "scatter"])
@pytest.mark.parametrize("first", [False, True])
def test_remark(dtype, chart, first):
    if dtype == "numeric":
        x = np.array([0.0, 1, 1, 3])
        xs_condition = [0, 1]
    elif dtype == "datetime":
        x = pd.to_datetime(
            ["2017-02-01", "2017-02-02", "2017-02-02", "2017-02-03"]
        ).values
        xs_condition = pd.to_datetime(["2017-02-01", "2017-02-02"])
    y = [4, 5, 6, 7]

    ah_obj = (
        ah.Array(x, y, chart=chart).remark("x", xs=xs_condition).remark("abcdef", ys=7)
    )
    ds = ah_obj[1, 1]
    remarks = ds["remark"].squeeze().values
    assert ds["remark"].ndim == 2
    if not first:
        assert (remarks[:3] == x[:3].astype(str)).all()
        assert remarks[-1] == "abcdef"
    else:
        assert (remarks[:2] == x[:2].astype(str)).all()
        assert remarks[-1] == "abcdef"


def test_remark_overlay():
    x = np.array([0, 1, 1, 3])
    y1 = [4, 5, 6, 7]
    y2 = [8, 9, 10, 11]

    ah_obj = ah.Array(x, y1) * ah.Array(x, y2)
    ah_obj = ah_obj.remark("abc", ys=[4, 11])
    ds = ah_obj[1, 1]
    remarks = ds["remark"]
    assert remarks.isel(item=0, state=0).item() == "abc"
    assert remarks.isel(item=1, state=-1).item() == "abc"


@pytest.mark.parametrize("first", [False, True])
def test_remark_cascade_first(first):
    xs = np.array([0, 1, 2])
    ys = np.array([3, 4, 4])

    arr = ah.Array(xs, ys)
    arr = arr - arr

    arr = arr.remark(ys=4, remarks="4!!", first=first)

    ds = arr._ds
    actual = ds["remark"]
    if first:
        expected = np.array([["", "4!!", "", "", "", ""], ["", "", "", "", "4!!", ""]])
    else:
        expected = np.array(
            [["", "4!!", "4!!", "4!!", "4!!", "4!!"], ["", "", "", "", "4!!", "4!!"]]
        )
    assert (actual == expected).all()


def test_stacked_fixed_limit():
    x = [0, 0]
    y1 = [1, 0]
    y2 = [2, 0]

    ah_obj = (
        ah.Array(x, y1, label="A", preset="stacked", chart="bar", revert="boomerang")
        * ah.Array(x, y2, label="B", preset="stacked", chart="bar", ylims="fixed")
    ).finalize()
    ds = ah_obj[1, 1]
    np.testing.assert_almost_equal(ds["ylim0"].values, 0)
    np.testing.assert_almost_equal(ds["ylim1"].values, 2)


def test_morph_stacked():
    ah_obj = (
        ah.Array([0, 1, 2], [5, 6, 7])
        * ah.Array([0, 1, 2], [5, 8, 9])
        * ah.Array(
            [0, 1, 2], [3, 4, 10], preset="morph_trail", chart="line", color="red"
        )
    ).finalize()
    ds = ah_obj[1, 1]
    assert "x_morph_trail" in ds.data_vars
    assert "y_morph_trail" in ds.data_vars
    assert (ds["group"] == "_morph_group").all()
    assert (ds["color"] == "red").all()


def test_pie_chart():
    ah_obj = (
        ah.Array([0, 1, 2], label="a", chart="pie", frames=1)
        * ah.Array([1, 0, 1], chart="pie")
    ).finalize()
    ds = ah_obj[1, 1]
    assert (ds["group"] == "_pie_group").all()
    assert (pd.unique(ds["labels"].values.ravel()) == ["a", ""]).all()
    assert ds["item"].values.tolist() == [1, 2]
    assert ds["state"].values.tolist() == [1, 2, 3]
    np.testing.assert_almost_equal(
        ds["y"].values.ravel(), np.array([0, 1, 0.6666666, 1, 0, 0.3333333])
    )  # normalize


def test_config_defaults_positional():
    ah.config_defaults("durations", final_frame=3, transition_frames=0.1)

    assert DEFAULTS["durations_kwds"]["final_frame"] == 3
    assert DEFAULTS["durations_kwds"]["transition_frames"] == 0.1


def test_config_defaults_batch():
    ah.config_defaults(
        **{
            "durations_kwds": dict(final_frame=5, transition_frames=0.5),
            "plot": {"scatter": dict(color="red", s=1000)},
        }
    )

    assert DEFAULTS["durations_kwds"]["final_frame"] == 5
    assert DEFAULTS["durations_kwds"]["transition_frames"] == 0.5
    assert DEFAULTS["plot_kwds"]["scatter"]["color"] == "red"
    assert DEFAULTS["plot_kwds"]["scatter"]["s"] == 1000
