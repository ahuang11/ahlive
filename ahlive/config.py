import xarray as xr


sizes = {
    'xx-small': 12,
    'x-small': 17,
    'small': 22,
    'medium': 26,
    'large': 30,
    'x-large': 36,
    'xx-large': 60,
    'xxx-large': 84
}

defaults = {}

defaults['durations'] = {
    'aggregate': 'max',
    'transition_frames': 1 / 60,
    'final_frame': 1,
}

defaults['spacing'] = {
    'left': 0.05,
    'right': 0.975,
    'bottom': 0.1,
    'top': 0.9,
    'wspace': 0.2,
    'hspace': 0.2
}

defaults['label'] = {
    'fontsize': sizes['medium'],
    'replacements': {'_': ' '},
    'format': 'auto'
}

defaults['chart'] = {}
defaults['chart']['bar'] = {
    'bar_label': True,
    'capsize': 6
}
defaults['chart']['scatter'] = {
    'expire': 100,
    'stride': 1
}
defaults['chart']['barh'] = defaults['chart']['bar'].copy()

defaults['ref_plot'] = {}
defaults['ref_plot']['rectangle'] = {
    'facecolor': 'darkgray',
    'alpha': 0.45
}
defaults['ref_plot']['scatter'] = {
    'color': 'black',
}
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
    'alpha': 0.45
}
defaults['ref_plot']['axhspan'] = {
    'color': 'darkgray',
    'alpha': 0.45
}

defaults['ref_inline'] = defaults['label'].copy()
defaults['ref_inline'].update({
    'textcoords': 'offset points',
    'color': 'darkgray',
})

defaults['grid_inline'] = defaults['label'].copy()
defaults['grid_inline'].update({
    'textcoords': 'offset points',
    'color': 'darkgray',
})

defaults['remark_inline'] = defaults['label'].copy()
defaults['remark_inline'].update({
    'fontsize': sizes['small'],
    'textcoords': 'offset points',
    'xytext': (0, 1.5),
    'ha': 'left',
    'va': 'top',
})

defaults['xlabel'] = defaults['label'].copy()
defaults['xlabel'].update({'fontsize': sizes['large']})

defaults['ylabel'] = defaults['label'].copy()
defaults['ylabel'].update({'fontsize': sizes['large']})

defaults['clabel'] = defaults['label'].copy()
defaults['clabel'].update({'fontsize': sizes['large']})

defaults['title'] = defaults['label'].copy()
defaults['title'].update({
    'fontsize': sizes['large'], 'loc': 'left'})

defaults['subtitle'] = defaults['label'].copy()
defaults['subtitle'].update({
    'fontsize': sizes['medium'], 'loc': 'right'})

defaults['note'] = {
    'x': .01,
    'y': .05,
    'ha': 'left',
    'va': 'top',
    'fontsize': sizes['x-small']
}

defaults['caption'] = {
    'x': 0,
    'y': -0.28,
    'alpha': 0.7,
    'ha': 'left',
    'va': 'bottom',
    'fontsize': sizes['x-small']
}

defaults['suptitle'] = defaults['label'].copy()
defaults['suptitle'].update({'fontsize': sizes['large']})

defaults['state'] = defaults['label'].copy()
defaults['state'].update({
    'alpha': 0.5,
    'xy': (0.988, 0.01),
    'ha': 'right',
    'va': 'bottom',
    'xycoords': 'axes fraction',
    'fontsize': sizes['xxx-large'],
})

defaults['inline'] = defaults['label'].copy()
defaults['inline'].update({
    'textcoords': 'offset points',
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

defaults['watermark'] = {
    'x': .995,
    'y': .005,
    'alpha': 0.28,
    'ha': 'right',
    'va': 'bottom',
    'fontsize': sizes['xx-small'],
    's': 'animated using ahlive'
}

defaults['frame'] = {
    'format': 'jpg',
    'backend': 'agg',
    'transparent': False
}

defaults['compute'] = {
    'num_workers': 4,
    'scheduler': 'processes'
}

defaults['animate'] = {
    'format': 'gif',
    'mode': 'I',
}


def scale_sizes(scale, keys=None):
    if keys is None:
        keys = sizes.keys()

    for key in keys:
        sizes[key] = sizes[key] * scale


def load_defaults(default_key, input_kwds=None, **other_kwds):
    # get default values
    updated_kwds = defaults.get(default_key, {}).copy()

    # unnest dictionary if need
    if default_key in ['chart', 'ref_plot', 'grid_plot']:
        updated_kwds = updated_kwds.get(
            other_kwds.pop('base_chart', None), {}
        ).copy()
    if isinstance(input_kwds, xr.Dataset):
        input_kwds = input_kwds.attrs[default_key]

    # update with programmatically generated values
    updated_kwds.update(
        {key: val for key, val in other_kwds.items()
        if val is not None})

    # update with user input values
    if input_kwds is not None:
        updated_kwds.update({
            key: val for key, val in input_kwds.items()
            if val is not None})
    updated_kwds.pop('base_chart', None)
    return updated_kwds


def update_defaults(default_key, **kwds):
    defaults[default_key].update(**kwds)
