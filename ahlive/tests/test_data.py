from collections.abc import ItemsView, KeysView, ValuesView

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ahlive as ah
from ahlive.configuration import CONFIGURABLES, ITEMS, OPTIONS
from ahlive.tests.test_configuration import (  # noqa: F401
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
    TYPES,
    XS,
    YS,
    ah_array1,
    ah_array2,
)
from ahlive.tests.test_util import assert_attrs, assert_types, assert_values
from ahlive.util import is_scalar


@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("x", XS)
@pytest.mark.parametrize("y", YS)
def test_ah_array(type_, x, y):
    x_iterable = type_(x) if isinstance(x, list) else x
    y_iterable = type_(y) if isinstance(y, list) else y
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
@pytest.mark.parametrize("join", ITEMS["join"])
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
        if join in ["overlay", "cascade"]:
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
@pytest.mark.parametrize("join", ITEMS["join"])
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
        var_dict = {
            "grid_x": sub_ds["x"],
            "grid_y": sub_ds["y"],
            "grid_c": sub_ds["c"],
            "grid_label": sub_ds["label"],
        }

        assert len(np.unique(sub_ds["label"])) == len(ds["grid_item"])
        if join != "cascade":
            assert_values(ds, var_dict)

        configurables = CONFIGURABLES.copy()
        assert_attrs(ds, configurables)

    ah_dataset.finalize()


@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("ref_x0", REF_X0S)
@pytest.mark.parametrize("ref_x1", REF_X1S)
@pytest.mark.parametrize("ref_y0", REF_Y0S)
@pytest.mark.parametrize("ref_y1", REF_Y1S)
def test_ah_reference(type_, ref_x0, ref_x1, ref_y0, ref_y1):
    refs = [ref_x0, ref_x1, ref_y0, ref_y1]
    iterables = [type_(ref) if isinstance(ref, list) else ref for ref in refs]
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


def test_keys(ah_array1):
    assert isinstance(ah_array1.keys(), KeysView)


def test_values(ah_array1):
    assert isinstance(ah_array1.values(), ValuesView)


def test_items(ah_array1):
    assert isinstance(ah_array1.items(), ItemsView)


def test_data(ah_array1):
    ah_data = ah_array1.data
    assert isinstance(ah_data, dict)
    assert isinstance(list(ah_data.keys())[0], tuple)
    assert isinstance(ah_data[1, 1], xr.Dataset)


def test_attrs(ah_array1):
    ah_attrs = ah_array1.attrs
    assert ah_array1[1, 1].attrs == ah_attrs
    assert isinstance(ah_attrs, dict)


@pytest.mark.parametrize("num_cols", [0, 1, 2, 3])
def test_cols(num_cols, ah_array1, ah_array2):
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


@pytest.mark.parametrize("preset", [None, "series", "race", "delta"])
def test_config_bar_chart(preset):
    x = ["a", "a", "b", "b", "b"]
    y = [4, 5, 3, 8, 10]
    df = pd.DataFrame({"x": x, "y": y})
    ah_df = ah.DataFrame(
        df, "x", "y", label="x", chart="barh", preset=preset, frames=1
    ).finalize()
    ds = ah_df[1, 1]
    print(ds)

    if preset == "race":
        actual = ds["x"].values.ravel()
        expected = [2, 1, 1, 1, 2, 2]
        assert (actual == expected).all()
    else:
        actual = ds["x"].values.ravel()
        expected = [1, 1, 1, 2, 2, 2]
        assert (actual == expected).all()

    actual = ds["y"].values.ravel()
    expected = [4, 5, np.nan, 3, 8, 10]
    np.testing.assert_equal(actual, expected)

    for var in ["tick_label", "bar_label"]:
        actual = ds[var].values.ravel()
        expected = ["a", "a", "a", "b", "b", "b"]
        assert (actual == expected).all()

    if preset is not None:
        assert ds.attrs["preset_kwds"]["preset"] == preset


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


def test_config_legend_sortby(ah_array1, ah_array2):
    ah_obj = (ah_array1 * ah_array2).config("legend", sortby="y").finalize()
    ds = ah_obj[1, 1]
    assert (ds["label"] == [2, 1]).all()
    assert ds.attrs["legend_kwds"]["show"]


@pytest.mark.parametrize("num_items", [1, 11])
def test_config_legend_show(num_items, ah_array1):
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


def test_fill_null(ah_array1, ah_array2):
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
@pytest.mark.parametrize("margin", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_xlim0s(direction, limit, margin):
    # TODO: test datetimes, strings
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = -2
    elif limit.startswith("follow"):
        expected = np.array([-1.0, 1, -2])
    elif limit.startswith("explore"):
        expected = np.array([-1.0, -1, -2])

    if margin is not None:
        expected -= np.nanmedian(np.abs(expected)) * margin
        limit = f"{limit}_{margin}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], xlim0s=limit, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["xlim0s"].values
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("margin", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_xlim1s(direction, limit, margin):
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = 1
    elif limit.startswith("follow"):
        expected = np.array([0.0, 1, 0])
    elif limit.startswith("explore"):
        expected = np.array([0.0, 1, 1])

    if margin is not None:
        expected += np.nanmedian(np.abs(expected)) * margin
        limit = f"{limit}_{margin}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], xlim1s=limit, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["xlim1s"].values
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("margin", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_ylim0s(direction, limit, margin):
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = -5
    elif limit.startswith("follow"):
        expected = np.array([3.0, 4, -5])
    elif limit.startswith("explore"):
        expected = np.array([3.0, 3, -5])

    if margin is not None:
        expected -= np.nanmedian(np.abs(expected)) * margin
        limit = f"{limit}_{margin}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], ylim0s=limit, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["ylim0s"].values
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("limit", OPTIONS["limit"])
@pytest.mark.parametrize("margin", [None, -0.1, 0, 0.1])
def test_add_xy01_limits_ylim1s(direction, limit, margin):
    if limit.startswith("zero"):
        expected = 0
    elif limit.startswith("fixed"):
        expected = 6
    elif limit.startswith("follow"):
        expected = np.array([5.0, 6, 5])
    elif limit.startswith("explore"):
        expected = np.array([5.0, 6, 6])

    if margin is not None:
        expected += np.nanmedian(np.abs(expected)) * margin
        limit = f"{limit}_{margin}"

    ah_array1 = ah.Array([0, 1, 0], [3, 4, 5], ylim1s=limit, frames=1)
    ah_array2 = ah.Array([-1, 1, -2], [5, 6, -5])

    ah_objs = [ah_array1, ah_array2]
    if direction == "backward":
        ah_objs = ah_objs[::-1]
    ah_obj = ah.merge(ah_objs).finalize()
    ds = ah_obj[1, 1]

    actual = ds["ylim1s"].values
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
    assert "norm" in attrs
    assert attrs["colorbar_kwds"]["show"]


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


def test_compress_var(ah_array1, ah_array2):
    ah_obj = (ah_array2 * ah_array1).finalize()
    ds = ah_obj[1, 1]
    assert is_scalar(ds["xlim0s"])


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


def test_add_margins_xlims():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], xlim0s=0, xlim1s=2, xmargins=1).finalize()
    ds = ah_obj[1, 1]
    assert ds["xlim0s"] == -2
    assert ds["xlim1s"] == 4


def test_add_margins_ylims():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], ylim0s=3, ylim1s=5, ymargins=1).finalize()
    ds = ah_obj[1, 1]
    assert ds["ylim0s"] == -2
    assert ds["ylim1s"] == 10


def test_add_durations_default():
    ah_obj = ah.Array([0, 1], [2, 3], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert ds["duration"].attrs["aggregate"] == "max"
    assert ds["duration"].attrs["transition_frames"] == 1 / 60
    assert (ds["duration"].values == [0.5, 0, 1.05]).all()


def test_add_durations_input():
    ah_obj = (
        ah.Array([0, 1], [2, 3], frames=3, durations=[0, 1])
        .config("durations", final_frame=2, transition_frames=2, aggregate="min")
        .finalize()
    )
    ds = ah_obj[1, 1]
    print(ds)
    assert ds["duration"].attrs["aggregate"] == "min"
    assert ds["duration"].attrs["transition_frames"] == 2
    assert (ds["duration"].values == [0, 0, 3]).all()


def test_interp_dataset():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert (ds["x"] == [0, 0.5, 1.0, 1.0, 1.5, 2.0]).all()
    assert (ds["y"] == [3, 3.5, 4, 4, 4.5, 5]).all()


def test_interp_dataset_ref():
    ah_obj = ah.Reference([0, 1, 2], [3, 4, 5], frames=3).finalize()
    ds = ah_obj[1, 1]
    assert (ds["ref_x0"] == [0, 0.5, 1.0, 1.0, 1.5, 2.0]).all()
    assert (ds["ref_x1"] == [3, 3.5, 4, 4, 4.5, 5]).all()


def test_interp_dataset_grid():
    c0 = [[0, 1], [2, 3]]
    c1 = [[2, 3], [4, 5]]
    ah_obj = ah.Array2D([0, 1], [2, 3], [c0, c1], frames=3).finalize()
    ds = ah_obj[1, 1]
    print(ds.isel(state=0))
    assert (ds["grid_c"].isel(state=0) == c0).all()
    assert (ds["grid_c"].isel(state=-1) == c1).all()


def test_add_geo_transform():
    ah_obj = (
        ah.Array([0, 1, 2], [3, 4, 5], crs="PlateCarree", projection="Robinson")
        .config("projection", central_longitude=180)
        .finalize()
    )
    attrs = ah_obj.attrs
    assert attrs["crs_kwds"]["crs"] == "PlateCarree"
    assert attrs["projection_kwds"]["projection"] == "Robinson"
    assert attrs["projection_kwds"]["central_longitude"] == 180

    for key in ITEMS["transformables"]:
        assert "transform" in attrs[key]

    ds = ah_obj[1, 1]
    assert "projection" in ds


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
    assert animate_kwds["num_states"] == num_states


def test_add_animate_kwds_slice():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=slice(1, 10)).finalize()
    attrs = ah_obj.attrs
    num_states = len(ah_obj[1, 1]["state"])
    animate_kwds = attrs["animate_kwds"]
    states = np.arange(1, 10)
    assert (animate_kwds["states"] == states).all()
    assert animate_kwds["stitch"]
    assert not animate_kwds["static"]
    assert animate_kwds["num_states"] == num_states


@pytest.mark.parametrize("animate", [True, False])
def test_add_animate_kwds_bool(animate):
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=animate).finalize()
    attrs = ah_obj.attrs
    num_states = len(ah_obj[1, 1]["state"])
    animate_kwds = attrs["animate_kwds"]
    assert animate_kwds["states"] is None
    assert animate_kwds["stitch"] == animate
    assert not animate_kwds["static"]
    assert animate_kwds["num_states"] == num_states


def test_add_animate_kwds_int():
    ah_obj = ah.Array([0, 1, 2], [3, 4, 5], animate=1).finalize()
    attrs = ah_obj.attrs
    num_states = len(ah_obj[1, 1]["state"])
    animate_kwds = attrs["animate_kwds"]
    assert animate_kwds["states"] == 1
    assert animate_kwds["stitch"]
    assert animate_kwds["static"]
    assert animate_kwds["num_states"] == num_states
