import xarray as xr


sizes = {
    'xx-small': 10,
    'x-small': 16,
    'small': 22,
    'medium': 24,
    'large': 30,
    'x-large': 36,
    'xx-large': 60,
    'xxx-large': 84
}

defaults = {}

defaults['delays'] = {
    'aggregate': 'max',
    'transition_frames': 1 / 60,
    'final_frame': 1,
}

defaults['figure'] = {
    'figsize': (18, 9)
}

defaults['axes'] = {
    'frame_on': False
}

defaults['label'] = {
    'fontsize': sizes['medium'],
    'replacements': {'_': ' '},
    'casing': 'title',
    'format': 'auto'
}

defaults['chart'] = {}
defaults['chart']['bar'] = {
    'bar_label': True,
    'capsize': 6
}
defaults['chart']['scatter'] = {
    'expire': 1000,
    'stride': 1
}
defaults['chart']['barh'] = defaults['chart']['bar'].copy()

defaults['ref_plot'] = {}
defaults['ref_plot']['axvline'] = {
    'color': 'darkgray',
    'linestyle': '--'
}
defaults['ref_plot']['axhline'] = {
    'color': 'darkgray',
    'linestyle': '--'
}
defaults['ref_plot']['axvspan'] = {
    'color': 'darkgray',
    'alpha': 0.3
}
defaults['ref_plot']['axhspan'] = {
    'color': 'darkgray',
    'alpha': 0.3
}

defaults['ref_inline'] = defaults['label'].copy()
defaults['ref_inline'].update({
    'textcoords': 'offset points',
    'color': 'darkgray',
    'casing': False
})

defaults['remark_inline'] = defaults['label'].copy()
defaults['remark_inline'].update({
    'fontsize': sizes['small'],
    'textcoords': 'offset points',
    'xytext': (0, 1.5),
    'ha': 'left',
    'va': 'top',
    'casing': False
})

defaults['xlabel'] = defaults['label'].copy()
defaults['xlabel'].update({'fontsize': sizes['large']})

defaults['ylabel'] = defaults['label'].copy()
defaults['ylabel'].update({'fontsize': sizes['large']})

defaults['clabel'] = defaults['label'].copy()
defaults['clabel'].update({'fontsize': sizes['large']})

defaults['title'] = defaults['label'].copy()
defaults['title'].update({
    'fontsize': sizes['large'], 'loc': 'left', 'casing': None})

defaults['suptitle'] = defaults['label'].copy()
defaults['suptitle'].update({
    'fontsize': sizes['large'], 'casing': None})

defaults['state'] = defaults['label'].copy()
defaults['state'].update({
    'alpha': 0.5,
    'xy': (0.975, 0.025),
    'ha': 'right',
    'va': 'bottom',
    'xycoords': 'axes fraction',
    'fontsize': sizes['xxx-large'],
})

defaults['inline'] = defaults['label'].copy()
defaults['inline'].update({
    'textcoords': 'offset points',
    'casing': False
})

defaults['legend'] = defaults['label'].copy()
defaults['legend'].update({
    'show': True,
    'framealpha': 0,
    'loc': 'upper left',
})

defaults['colorbar'] = {
    'orientation': 'vertical',
}

defaults['ticks'] = defaults['label'].copy()
defaults['ticks'].pop('fontsize')
defaults['ticks'].update({
    'length': 0,
    'which': 'both',
    'labelsize': sizes['small'],
})

defaults['xticks'] = defaults['ticks'].copy()
defaults['xticks'].update({'axis': 'x'})

defaults['yticks'] = defaults['ticks'].copy()
defaults['yticks'].update({'axis': 'y'})

defaults['cticks'] = defaults['ticks'].copy()
defaults['cticks'].update({'num_ticks': 5})

defaults['land'] = {
    'facecolor': 'whitesmoke'
}

defaults['caption'] = {
    'x': .085,
    'y': .005,
    'ha': 'left',
    'va': 'bottom',
    'fontsize': sizes['small']
}

defaults['watermark'] = {
    'x': .995,
    'y': .005,
    'alpha': 0.28,
    'ha': 'right',
    'va': 'bottom',
    'fontsize': sizes['x-small']
}

defaults['frame'] = {
    'format': 'jpg',
    'backend': 'agg',
    'transparent': False
}

defaults['animate'] = {
    'format': 'gif',
    'mode': 'I',
    'subrectangles': True
}


def scale_sizes(scale, keys=None):
    if keys is None:
        keys = sizes.keys()

    for key in keys:
        sizes[key] = sizes[key] * scale


def load_defaults(default_key, input_kwds=None, **other_kwds):
    updated_kwds = defaults.get(default_key, {}).copy()
    if default_key in ['chart', 'ref_plot']:
        updated_kwds = updated_kwds.get(
            other_kwds.pop('base_chart', None), updated_kwds
        ).copy()
    if isinstance(input_kwds, xr.Dataset):
        input_kwds = input_kwds.attrs[default_key]
    updated_kwds.update(
        {key: val for key, val in other_kwds.items()
        if val is not None
    })
    if input_kwds is not None:
        updated_kwds.update(input_kwds)
    updated_kwds.pop('base_chart', None)
    return updated_kwds


def update_defaults(default_key, **kwds):
    defaults[default_key].update(**kwds)
