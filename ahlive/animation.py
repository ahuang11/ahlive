from io import BytesIO

import param
import imageio
import numpy as np
import xarray as xr
import dask.delayed
import dask.diagnostics
from pygifsicle import optimize
from matplotlib import pyplot as plt


STATE_VARS = [
    'duration', 'label', 'x0_limit', 'x1_limit', 'y0_limit', 'y1_limit']

SIZES = {
    'small': 15,
    'medium': 18,
    'large': 20,
    'super': 28
}


class Animation(param.Parameterized):

    ds = param.ObjectSelector(default=xr.Dataset(), allow_None=True)
    plot = param.ObjectSelector(
        default='scatter', objects=['scatter', 'line'])
    num_workers = param.Integer(default=8, bounds=(1, None))
    out_fp = param.Path(default='untitled.gif')

    minimalist = param.Boolean(default=True)
    figsize = param.Tuple(default=(10, 8))
    title = param.String(default=None, allow_None=True)
    xlabel = param.String(default=None, allow_None=True)
    ylabel = param.String(default=None, allow_None=True)

    def __init__(self, **kwds):
        super().__init__(**kwds)

        plt.rc('font', size=SIZES['small'])
        plt.rc('axes', labelsize=SIZES['medium'])
        plt.rc('xtick', labelsize=SIZES['small'])
        plt.rc('ytick', labelsize=SIZES['small'])
        plt.rc('legend', fontsize=SIZES['small'])
        plt.rc('figure', titlesize=SIZES['large'])

    @dask.delayed()
    def _draw(self, ds_state):
        fig_kwds = ds_state.attrs
        fig_kwds['figsize'] = self.figsize

        axes_kwds = {
            'xlim': (
                ds_state['x0_limit'].item(),
                ds_state['x1_limit'].item()
            ),
            'ylim': (
                ds_state['y0_limit'].item(),
                ds_state['y1_limit'].item()
            ),
            'title': self.title,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
        }

        if self.minimalist:
            axes_kwds['xlabel'] = f'Higher {self.xlabel} ➜'
            axes_kwds['ylabel'] = f'Higher {self.ylabel} ➜'
            axes_kwds['xticks'] = [
                round(float(ds_state['x'].min()), 1),
                round(float(ds_state['x'].max()), 1),
            ]
            axes_kwds['yticks'] = [
                round(float(ds_state['y'].min()), 1),
                round(float(ds_state['y'].max()), 1),
            ]

        fig = plt.figure(**fig_kwds)
        ax = plt.axes(**axes_kwds)

        base_kwds = {}
        has_colors = 'c' in ds_state
        if has_colors:
            base_kwds['cmap'] = 'RdBu_r'
            base_kwds['vmin'] = self.vmin.values
            base_kwds['vmax'] = self.vmax.values

        for label, ds_overlay in ds_state.groupby('label'):
            plot_kwds = base_kwds.copy()
            plot_kwds.update({
                var: ds_overlay[var].values
                for var in ds_overlay if var not in STATE_VARS
            })

            if 'alpha' in plot_kwds:
                plot_kwds['alpha'] = plot_kwds['alpha'][0]

            x = plot_kwds.pop('x')
            y = plot_kwds.pop('y')

            if 'state_label' in plot_kwds:
                state_label = plot_kwds.pop('state_label')
                if isinstance(state_label.flat[0], float):
                    state_label = f'{state_label:.0f}'
                ax.text(0.9, 0.9, state_label, transform=ax.transAxes,
                        fontsize=SIZES['super'], alpha=0.9)

            if 'inline_label' in plot_kwds:
                inline_labels = plot_kwds.pop('inline_label')
                if isinstance(inline_labels.flat[0], float):
                    inline_labels = np.array(inline_labels).round(2)
                for i, inline_label in enumerate(inline_labels):
                    ax.annotate(
                        inline_label, xy=(x[i], y[i]),
                        xycoords='data', xytext=(1.5, 1.5),
                        ha='left', va='bottom', textcoords='offset points')

            image = getattr(ax, self.plot)(x, y, label=label, **plot_kwds)

        if has_colors:
            plt.colorbar(image)

        if ax.get_legend_handles_labels()[1]:
            ax.legend()

        plt.box(False)
        ax.tick_params(
            axis='both', which='both', length=0, color='gray')
        ax.grid()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def save(self, ds):
        if 'label' not in ds:
            ds['label'] = ('item', np.repeat('', len(ds['item'])))

        if 'duration' in ds:
            duration = ds['duration'].values.tolist()

        if 'c' in ds:
            self.vmin = ds['c'].min()
            self.vmax = ds['c'].max()

        num_states = len(ds['state'])
        if num_states > self.num_workers:
            num_workers = self.num_workers
        else:
            num_workers = num_states

        with dask.diagnostics.ProgressBar(minimum=3):
            buf_list = dask.compute([
                self._draw(
                    ds.isel(**{'state': state}).reset_coords(),
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
