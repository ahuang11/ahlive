sizes = {
    'xx-small': 10,
    'x-small': 16,
    'small': 20,
    'medium': 24,
    'large': 30,
    'x-large': 36,
    'xx-large': 60,
    'xxx-large': 84
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

defaults['plot_kwds'] = {
    's': 80,
}

defaults['label_kwds'] = {
    'fontsize': sizes['medium'],
    'replacements': {'_': ' '},
    'casing': 'title',
    'format': 'auto'
}

defaults['chart_kwds'] = {}
defaults['chart_kwds']['bar'] = {
    'kind': 'race',
    'capsize': 5
}
defaults['chart_kwds']['barh'] = defaults['chart_kwds']['bar'].copy()

defaults['trail_kwds'] = {
    'color': 'gray',
    'alpha': 0.5,
    'expire': 12,
    'stride': 2,
}

defaults['annotation_kwds'] = defaults['label_kwds'].copy()
defaults['annotation_kwds'].update({
    'fontsize': sizes['small'],
    'textcoords': 'offset points',
    'xytext': (0, 1.5),
    'ha': 'left',
    'va': 'top'
})

defaults['xlabel_kwds'] = defaults['label_kwds'].copy()
defaults['xlabel_kwds'].update({'fontsize': sizes['large']})

defaults['ylabel_kwds'] = defaults['label_kwds'].copy()
defaults['ylabel_kwds'].update({'fontsize': sizes['large']})

defaults['clabel_kwds'] = defaults['label_kwds'].copy()
defaults['clabel_kwds'].update({'fontsize': sizes['large']})

defaults['title_kwds'] = defaults['label_kwds'].copy()
defaults['title_kwds'].update({'fontsize': sizes['large'], 'loc': 'left'})

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

defaults['colorbar_kwds'] = {
    'orientation': 'vertical'
}

defaults['tick_kwds'] = defaults['label_kwds'].copy()
defaults['tick_kwds'].pop('fontsize')
defaults['tick_kwds'].update({
    'length': 0,
    'which': 'both',
    'color': 'gray',
    'labelsize': sizes['small'],
})

defaults['xtick_kwds'] = defaults['tick_kwds'].copy()
defaults['xtick_kwds'].update({'axis': 'x'})

defaults['ytick_kwds'] = defaults['tick_kwds'].copy()
defaults['ytick_kwds'].update({'axis': 'y'})

defaults['ctick_kwds'] = defaults['tick_kwds'].copy()

defaults['watermark_kwds'] = {
    'alpha': 0.28,
    'ha': 'right',
    'va': 'bottom',
    'fontsize': sizes['xx-small']
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
    if default_key == 'chart_kwds':
        updated_kwds = updated_kwds.get(other_kwds.pop('chart'), updated_kwds)
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
