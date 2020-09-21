from io import BytesIO

import param
import imageio
import numpy as np
import xarray as xr
import dask.delayed
import dask.diagnostics
from pygifsicle import optimize
from matplotlib import pyplot as plt


class Animation(param.Parameterized):

    ds = param.ObjectSelector(default=None, allow_None=True)
    plot = param.ObjectSelector(
        default='scatter', objects=['scatter', 'line'])
    num_workers = param.Integer(default=8, bounds=(1, None))
    out_fp = param.Path(default='untitled.gif')

    def __init__(self, **kwds):
        super().__init__(**kwds)

    @staticmethod
    @dask.delayed()
    def _draw(ds_state, plot, vmin=None, vmax=None):
        fig_kwds = ds_state.attrs

        plot_kwds = {
            var: ds_state[var].values
            for var in ds_state.data_vars
            if var != 'state'
        }

        has_colors = 'c' in plot_kwds
        if has_colors:
            plot_kwds['cmap'] = 'RdBu_r'
            plot_kwds['vmin'] = vmin.values
            plot_kwds['vmax'] = vmax.values

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

        inline_labels = plot_kwds.pop('inline_label', np.array([])).round(2)
        state_label = plot_kwds.pop('state_label', np.array(''))
        if isinstance(state_label.item(), float):
            state_label = f'{state_label:.0f}'

        fig = plt.figure(**fig_kwds)
        ax = plt.axes(**axes_kwds)

        for i, inline_label in enumerate(inline_labels):
            sub_kwds = {
                key: val[i:i + 1]
                if len(np.atleast_1d(val)) > 1 else val
                for key, val in plot_kwds.items()}
            label = sub_kwds.pop('label')[0]
            image = getattr(ax, plot)(
                x[i:i + 1], y[i:i + 1],
                label=label, **sub_kwds)
            ax.annotate(
                inline_label, xy=(x[i], y[i]),
                xycoords='data', xytext=(1.5, 1.5),
                ha='left', va='bottom', textcoords='offset points')
        if has_colors: plt.colorbar(image)

        ax.text(0.05, 0.95, state_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def save(self, ds):
        if 'duration' in ds:
            duration = ds['duration'].values.tolist()
            ds = ds.drop('duration')

        draw_kwds = dict(plot=self.plot)
        if 'c' in ds:
            draw_kwds['vmin'] = ds['c'].min()
            draw_kwds['vmax'] = ds['c'].max()

        num_states = len(ds['state'])
        if num_states > self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = num_states

        with dask.diagnostics.ProgressBar(minimum=3):
            buf_list = dask.compute([
                self._draw(
                    ds.isel(**{'state': state}).reset_coords(), **draw_kwds
                ) for state in ds['state'].values
            ], scheduler='processes', num_workers=num_workers)[0]

        if isinstance(self.loop, bool):
            loop = int(not self.loop)
        elif isinstance(self.loop, str):
            loop = 0
        else:
            loop = self.loop

        writer_kwds = dict(
            format='gif', mode='I', loop=loop,
            duration=duration, subrectangles=True)

        with imageio.get_writer(
            self.out_fp, **writer_kwds
        ) as writer:
            for buf in buf_list:
                image = imageio.imread(buf)
                writer.append_data(image)
        optimize(self.out_fp)
        gif = imageio.get_reader(self.out_fp)
        return gif
