from io import BytesIO

import param
import numpy as np
import xarray as xr
import dask.delayed
import dask.diagnostics
from matplotlib import pyplot as plt


class Animation(param.Parameterized):

    ds = param.ObjectSelector(default=None, allow_None=True)
    plot = param.ObjectSelector(
        default='scatter', objects=['scatter', 'line'])
    num_workers = param.Integer(
        default=4, bounds=(1, None))
    out_fp = param.Path(
        default='untitled.gif')

    def __init__(self, **kwds):
        super().__init__(**kwds)

    @dask.delayed()
    def _draw(self, ds, state):
        state_sel = slice(None, state) if self.plot == 'plot' else state
        ds_state = ds.isel(**{'state': state_sel}).reset_coords()
        fig_kwds = ds_state.attrs

        plot_kwds = {
            var: ds_state[var].values
            for var in ds_state.data_vars
            if var != 'state'
        }

        has_colors = 'c' in plot_kwds
        if has_colors:
            plot_kwds['cmap'] = 'RdBu_r'
            plot_kwds['vmin'] = float(self.ds['c'].min())
            plot_kwds['vmax'] = float(self.ds['c'].max())

        x = plot_kwds.pop('x')
        y = plot_kwds.pop('y')

        x0_limit = plot_kwds.pop('x0_limit').item()
        x1_limit = plot_kwds.pop('x1_limit').item()
        y0_limit = plot_kwds.pop('y0_limit').item()
        y1_limit = plot_kwds.pop('y1_limit').item()

        axes_kwds = {
            'xlim': (x0_limit, x1_limit),
            'ylim': (y0_limit, y1_limit)
        }
        item_label = plot_kwds.pop('item_label')
        state_label = plot_kwds.pop('state_label')
        if isinstance(state_label.item(), float):
            state_label = f'{state_label:.0f}'

        fig = plt.figure(**fig_kwds)
        ax = plt.axes(**axes_kwds)
        plot = getattr(ax, self.plot)(x, y, **plot_kwds)
        if has_colors: plt.colorbar(plot)

        for i, l in enumerate(item_label):
            ax.annotate(
                l, xy=(x[i], y[i]), xycoords='data', xytext=(1.5, 1.5),
                ha='left', va='bottom', textcoords='offset points')

        ax.text(0.05, 0.95, state_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
