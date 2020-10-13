import warnings
from collections.abc import Iterable

import dask
import param
import numpy as np
import xarray as xr

from . import config, easing, animation, util


OPTIONS = {
    'limit': ['fixed', 'follow']
}


class Ahlive(easing.Easing, animation.Animation):

    xlim0s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    xlim1s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    ylim0s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    ylim1s = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    delays = param.ClassSelector(
        default=None, class_=(Iterable, int, float))
    delays_kwds = param.Dict(
        default=None)
    show_out = param.Boolean(default=True)

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.ds = xr.Dataset({'item': [], 'state': []})
        self.num_states = len(self.ds['state'])
        self.num_items = len(self.ds['item'])
        self.final_ds = None
        self_attrs = set(dir(self))
        for key in list(kwds.keys()):
            key_and_s = key + 's'
            key_strip = key.rstrip('s')
            expected_key = None
            if key_and_s in self_attrs and key_and_s != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_and_s}!')
                expected_key = key_and_s
            elif key_strip in self_attrs and key_strip != key:
                warnings.warn(
                    f'Unexpected {key}; setting {key} '
                    f'as the expected {key_strip}!')
                expected_key = key_strip
            if expected_key:
                setattr(self, expected_key, kwds.pop(key))

    def _add_data(self, label=None, state_labels=None, **input_data_vars):
        if all(key in input_data_vars for key in ['x', 'y']):
            item_type = None
            self.num_states = len(input_data_vars['x'])
            saved_ds = self.ds
        else:
            if all(key in input_data_vars for key in ['x0', 'x1', 'y0', 'y1']):
                item_type = 'box'
                self.num_states = len(input_data_vars['x0'])
            elif all(key in input_data_vars for key in ['y0', 'x0', 'x1']):
                item_type = 'finite_hline'
                self.num_states = len(input_data_vars['y0'])
            elif all(key in input_data_vars for key in ['x0', 'y0', 'y1']):
                item_type = 'finite_vline'
                self.num_states = len(input_data_vars['x0'])
            elif 'y0' in input_data_vars:
                item_type = 'hline'
                self.num_states = len(input_data_vars['y0'])
            elif 'x0' in input_data_vars:
                item_type = 'vline'
                self.num_states = len(input_data_vars['x0'])
            saved_ds = self.ref_ds

        item_num = len(saved_ds['item'])
        coords = {'item': [item_num], 'state': np.arange(self.num_states)}
        if item_type is not None:
            coords['item_type'] = ('item', [item_type])

        if label is not None:
            if not isinstance(label, str):
                if len(label) > 0:
                    label = label[0]
            coords['label'] = ('item', [label])
            if item_num > 0 and 'label' not in saved_ds:
                saved_ds.coords['label'] = ('item', [label])
        elif 'label' in saved_ds:
            coords['label'] = saved_ds['label'].values[0]

        if state_labels is not None:
            coords['state_label'] = ('state', state_labels)

        data_vars = {
            key: val for key, val in input_data_vars.items()
            if val is not None}
        for var in list(data_vars.keys()):
            val = np.array(data_vars.pop(var))
            dims = ('item', 'state')
            if util.is_scalar(val):
                val = [val] * self.num_states
            val = np.reshape(val, (1, -1))
            data_vars[var] = dims, val
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        if len(saved_ds) == 0:
            saved_ds = ds
        else:
            saved_ds = xr.concat([saved_ds, ds], 'item')
        self.num_items = len(saved_ds['item'])
        return saved_ds

    def add_arrays(self, xs, ys, label=None, state_labels=None,
                   inline_labels=None, **input_data_vars):
        self.ds = self._add_data(
            x=xs, y=ys, label=label, state_labels=state_labels,
            inline_label=inline_labels, **input_data_vars)

    def add_annotations(self, condition, labels=None, delays=None):
        if 'annotation' not in self.ds:
            self.ds['annotation'] = (
                ('item', 'state'),
                np.full((len(self.ds['item']), self.num_states), '')
            )
            self.ds['delay'] = (
                'state', np.repeat(0, len(self.ds['state'])))

        if labels.startswith('$'):
            labels = self.ds[labels.lstrip('$')]
        self.ds['annotation'] = xr.where(
            condition, labels, self.ds['annotation']
        ).transpose('item', 'state')
        if delays is not None:
            self.ds['delay'] = xr.where(
                condition, delays, self.ds['delay'])

    def add_references(self, x0s=None, x1s=None, y0s=None, y1s=None,
                       label=None, state_labels=None, inline_labels=None,
                       **input_data_vars):
        if all(var is None for var in [x0s, x1s, y0s, y1s]):
            raise ValueError('Must provide either x0s, x1s, y0s, y1s!')
        elif x0s is None and x1s is not None:
            x0s, x1s = x1s, x0s
        elif y0s is None and y1s is not None:
            y0s, y1s = y1s, y0s

        x0s = x0s or np.nan
        x1s = x1s or np.nan
        y0s = y0s or np.nan
        y1s = y1s or np.nan

        self.ds_ref = self._add_data(
            x0=np.atleast_1d(x0s), x1=np.atleast_1d(x1s),
            y0=np.atleast_1d(y0s), y1=np.atleast_1d(y1s),
            label=label, state_labels=state_labels, inline_label=inline_labels,
            **input_data_vars
        )

    def add_dataframe(self, df, xs, ys, labels=None, state_labels=None,
                      inline_labels=None, **input_data_vars):
        if 'label' in input_data_vars:
            labels = input_data_vars.pop('label')

        if labels is None:
            df_iter = zip([None], [df])
        else:
            df_iter = df.groupby(labels)

        for label, df_item in df_iter:
            input_kwds = {
                key: df_item[val] if val in df_item.columns else val
                for key, val in input_data_vars.items()}
            self.add_arrays(
                df_item[xs], df_item[ys], label=label,
                state_labels=df_item.get(state_labels, None),
                inline_labels=df_item.get(inline_labels, None),
                **input_kwds
            )

        if self.chart == 'barh':
            xs, ys = ys, xs  # matplotlib draws x on y axis for barh
        if self.xlabel is None:
            self.xlabel = xs
        if self.ylabel is None:
            self.ylabel = ys
        if labels == inline_labels and self.legend is None:
            self.legend = False

    def _add_xy01_limits(self, ds):
        limits = {
            'xlim0': self.xlim0s,
            'xlim1': self.xlim1s,
            'ylim0': self.ylim0s,
            'ylim1': self.ylim1s
        }

        for key, limit in limits.items():
            axis = key[0]
            left = int(key[-1]) == 0

            axis_limit_key = f'{axis}lim'
            if self.axes_kwds is not None:
                in_axes_kwds = axis_limit_key in self.axes_kwds
            else:
                in_axes_kwds = False
            unset_limit = limit is None and not in_axes_kwds
            is_scatter = self.chart == 'scatter'
            is_line_y = self.chart == 'line' and axis == 'y'
            is_bar_y = self.chart.startswith('bar') and axis == 'y'
            if unset_limit and any([is_scatter, is_line_y, is_bar_y]):
                limit = 'fixed'
            elif isinstance(limit, str) and limit not in OPTIONS['limit']:
                raise ValueError(
                    f"Got {limit} for {key}; must be either "
                    f"from {OPTIONS['limit']} or numeric values!"
                )

            input_ = limit
            if isinstance(limit, str):
                stat = 'min' if left else 'max'
                dims = 'item' if limit == 'follow' else None
                limit = getattr(ds[axis], stat)(dims)

            if limit is not None:
                if self.chart == 'barh':
                    axis = 'x' if axis == 'y' else 'y'
                    key = axis + key[1:]
                if util.is_scalar(limit) == 1:
                    limit = [limit] * self.num_states
                ds[key] = ('state', limit)
        return ds

    def _add_delays(self, ds):
        if self.delays is None:
            delays = 0.5 if len(ds['state']) < 10 else 1 / 60
        else:
            delays = self.delays

        if isinstance(delays, (int, float)):
            transition_frames = delays
            delays = np.repeat(delays, len(ds['state']))
        else:
            transition_frames = (
                config.defaults['delays_kwds']['transition_frames'])

        delays_kwds = config._load(
            'delays_kwds', self.delays_kwds,
            transition_frames=transition_frames)
        delays[delays == 0] = delays_kwds['transition_frames']
        delays[-1] += delays_kwds['final_frame']
        if 'delay' in ds:
            ds['delay'] = ds['delay'] + delays
        else:
            ds['delay'] = ('state', delays)
        return ds, delays_kwds

    def finalize_settings(self):
        ds = self.ds.copy()

        if self.chart is None:
            self.chart = 'scatter' if len(ds['state']) <= 5 else 'line'
        elif self.chart.startswith('bar'):
            ds = xr.concat([
                ds_group.unstack().assign(**{
                    'item': [item],
                    'state': np.arange(len(ds_group['state']))})
                for item, (group, ds_group) in enumerate(ds.groupby('x'))
            ], 'item')
            ds['tick_label'] = ds['x']
            ds['x'] = ds['y'].rank('item')

        ds = self._add_xy01_limits(ds)
        ds, delays_kwds = self._add_delays(ds)

        if 'label' not in ds:
            ds['label'] = ('item', np.repeat('', len(ds['item'])))
        if self.loop is None:
            self.loop = 0 if self.chart == 'line' else 'boomerang'

        # sort legend
        if self.legend_kwds is not None:
            legend_sortby = self.legend_kwds.pop('sortby', None)
        else:
            legend_sortby = 'y'
        if legend_sortby and 'label' in ds:
            items = ds.mean('state').sortby(
                legend_sortby, ascending=False
            )['item']
            ds = ds.sel(**{'item': items})
            ds['item'] = np.arange(len(ds['item']))

        # initialize
        self.xmin = ds['x'].min()
        self.ymin = ds['y'].min()
        try:
            self.state_step = np.diff(ds.get(
                'state_label', np.array([0, 1])
            )).min()
        except TypeError:
            self.state_step = ds.get(
                'state_label', np.array([0, 1])
            ).min()
        try:
            self.inline_step = np.diff(ds.get(
                'inline_label', np.array([0, 1])
            )).min()
        except TypeError:
            self.inline_step = ds.get(
                'inline_label', np.array([0, 1])
            ).min()

        self.x_is_datetime = np.issubdtype(
            ds['x'].values.dtype, np.datetime64)
        self.y_is_datetime = np.issubdtype(
            ds['y'].values.dtype, np.datetime64)

        if 'c' in ds:
            self.vmin = ds['c'].min()
            self.vmax = ds['c'].max()

        if self.trail_chart and not self.chart.startswith('bar'):
            ds['x_trail'] = ds['x'].copy()
            ds['y_trail'] = ds['y'].copy()

        ds = ds.reset_coords().apply(self.interpolate)
        ds['delay'] = ds['delay'].fillna(delays_kwds['transition_frames'])

        self.final_ds = ds

    def save(self):
        if self.final_ds is None:
            self.finalize_settings()
        self.animate()
        if self.show_out:
            try:
                from IPython.display import Image
                with open(self.out_fp, 'rb') as f:
                    display(Image(data=f.read(), format='png'))
            except (ImportError):
                pass
        return self.out_fp
