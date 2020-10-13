sizes = {
    'xx-small': 8,
    'x-small': 12,
    'small': 16,
    'medium': 20,
    'large': 24,
    'x-large': 36,
    'xx-large': 56,
    'xxx-large': 100
}

defaults = {}

defaults['delays_kwds'] = {
    'transition_frames': 1 / 60,
    'final_frame': 1
}

defaults['fig_kwds'] = {
    'figsize': (16, 10)
}

defaults['axes_kwds'] = {
    'frame_on': False
}

defaults['chart_kwds'] = {
    's': 80
}

defaults['trail_kwds'] = {
    'color': 'gray',
    'alpha': 0.5,
    'expire': 12,
    'stride': 2,
}

defaults['label_kwds'] = {
    'fontsize': sizes['medium'],
    'replacements': {'_': ' '},
    'casing': 'title',
    'format': 'auto'
}
defaults['annotation_kwds'] = defaults['label_kwds'].copy()
defaults['annotation_kwds'].update({
    'textcoords': 'offset points',
    'xytext': (0, 1.5),
    'ha': 'left',
    'va': 'top'
})

defaults['xlabel_kwds'] = defaults['label_kwds'].copy()
defaults['xlabel_kwds'].update({'fontsize': sizes['large']})

defaults['ylabel_kwds'] = defaults['label_kwds'].copy()
defaults['ylabel_kwds'].update({'fontsize': sizes['large']})

defaults['title_kwds'] = defaults['label_kwds'].copy()
defaults['title_kwds'].update({'fontsize': sizes['large']})

defaults['state_kwds'] = defaults['label_kwds'].copy()
defaults['state_kwds'].update({
    'alpha': 0.5,
    'xy': (0.975, 0.025),
    'ha': 'right',
    'va': 'bottom',
    'xycoords': 'axes fraction',
    'fontsize': sizes['xxx-large'],
})

defaults['inline_kwds'] = defaults['label_kwds'].copy()
defaults['inline_kwds'].update({'textcoords': 'offset points'})

defaults['legend_kwds'] = defaults['label_kwds'].copy()
defaults['legend_kwds'].update({
    'show': True,
    'framealpha': 0,
    'loc': 'upper left',
    'bbox_to_anchor': (0.025, 0.95),
})

defaults['colorbar_kwds'] = defaults['label_kwds'].copy()
defaults['colorbar_kwds'].update({})

defaults['tick_kwds'] = defaults['label_kwds'].copy()
defaults['tick_kwds'].pop('fontsize')
defaults['tick_kwds'] = {
    'length': 0,
    'which': 'both',
    'color': 'gray',
    'labelsize': sizes['small'],
}

defaults['xtick_kwds'] = defaults['tick_kwds'].copy()
defaults['xtick_kwds'].update({'axis': 'x'})

defaults['ytick_kwds'] = defaults['tick_kwds'].copy()
defaults['ytick_kwds'].update({'axis': 'y'})

defaults['watermark_kwds'] = {
    'alpha': 0.28,
    'xy': (0.995, -0.1),
    'ha': 'right',
    'va': 'bottom',
    'xycoords': 'axes fraction',
    'fontsize': sizes['small']
}

defaults['frame_kwds'] = {
    'format': 'png'
}

defaults['animate_kwds'] = {
    'format': 'gif',
    'mode': 'I',
    'subrectangles': True
}


def _load(default_key, input_kwds, **other_kwds):
    updated_kwds = defaults.get(default_key, {}).copy()
    updated_kwds.update(
        {key: val for key, val in other_kwds.items()
        if val is not None
    })
    if input_kwds is not None:
        updated_kwds.update(input_kwds)
    return updated_kwds


def scale_sizes(scale, keys=None):
    if keys is None:
        keys = sizes.keys()

    for key in keys:
        sizes[key] = sizes[key] * scale


def update_defaults(default_key, **kwds):
    defaults[default_key].update(**kwds)
