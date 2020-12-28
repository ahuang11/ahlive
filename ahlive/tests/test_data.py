import pytest
import numpy as np
import pandas as pd
import xarray as xr

import ahlive as ah
from ahlive.tests.test_util import assert_types, assert_values, assert_attrs
from ahlive.tests.test_configuration import (
    TYPES, XS, YS, LABELS, GRID_XS, GRID_YS, GRID_CS, GRID_LABELS)


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

    ah_array.finalize()


@pytest.mark.parametrize("x", XS)
@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("join", ah.configuration.ITEMS['join'])
def test_ah_dataframe(x, y, label, join):
    df = pd.DataFrame({
        'x': np.array(x).squeeze(), 'y': np.array(y).squeeze(), 'label': label
    })
    ah_df = ah.DataFrame(
        df, 'x', 'y', s='y', label='label', join=join, frames=2)
    assert_types(ah_df)

    for ds in ah_df.data.values():
        sub_df = df.loc[df['label'].isin(ds['label'].values.ravel())]
        var_dict = {
            "x": sub_df['x'].values,
            "y": sub_df['y'].values,
            "s": sub_df['y'].values,
            "label": sub_df['label'].values,
        }

        num_labels = len(np.unique(df['label']))
        if join in ['overlay', 'cascade']:
            assert num_labels == len(ds['item'])
        else:
            assert 1 == len(ds['item'])

        if num_labels > 1 and join == 'cascade':
            for i in range(num_labels):
                assert_values(
                    ds.isel(item=i),
                    {key: val[i] for key, val in var_dict.items()
                })
        else:
            assert_values(ds, var_dict)

        configurables = ah.CONFIGURABLES.copy()
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

        assert 1 == len(ds['grid_item'])
        assert_values(ds, var_dict)

        configurables = ah.CONFIGURABLES.copy()
        assert_attrs(ds, configurables)

    ah_array2d.finalize()


@pytest.mark.parametrize("grid_x", GRID_XS)
@pytest.mark.parametrize("grid_y", GRID_YS)
@pytest.mark.parametrize("grid_c", GRID_CS)
@pytest.mark.parametrize("grid_label", GRID_LABELS)
@pytest.mark.parametrize("join", ah.configuration.ITEMS['join'])
def test_ah_array2d(grid_x, grid_y, grid_c, grid_label, join):
    base_ds = xr.Dataset()
    base_ds['c'] = xr.DataArray(
        grid_c, dims=('label', 'y', 'x'),
        coords={'y': grid_y, 'x': grid_x, 'label': grid_label},
    )
    ah_dataset = ah.Dataset(
        base_ds, 'x', 'y', 'c', label='label', join=join)
    assert_types(ah_dataset)

    for ds in ah_dataset.data.values():
        sub_ds = base_ds.where(
            base_ds['label'] == np.unique(ds['grid_label']), drop=True)
        var_dict = {
            "grid_x": sub_ds['x'],
            "grid_y": sub_ds['y'],
            "grid_c": sub_ds['c'],
            "grid_label": sub_ds['label'],
        }

        assert len(np.unique(sub_ds['label'])) == len(ds['grid_item'])
        if join != 'cascade':
            assert_values(ds, var_dict)

        configurables = ah.CONFIGURABLES.copy()
        assert_attrs(ds, configurables)

    ah_dataset.finalize()
