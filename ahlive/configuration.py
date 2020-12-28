from copy import deepcopy
from itertools import chain

import param
import xarray as xr

NULL_VALS = [(), {}, [], None, ""]

CONFIGURABLES = {
    "canvas": ["figure", "suptitle", "watermark", "spacing"],
    "subplot": ["axes", "plot", "preset", "legend", "grid", "xticks", "yticks", "margins"],
    "label": [
        "state",
        "inline",
        "xlabel",
        "ylabel",
        "title",
        "subtitle",
        "caption",
        "note",
        "preset_inline",
    ],
    "meta": [
        "output",
        "frame",
        "interpolate",
        "animate",
        "compute",
        "durations",
    ],
    "geo": [
        "crs",
        "projection",
        "borders",
        "coastline",
        "land",
        "lakes",
        "ocean",
        "rivers",
        "states",
    ],
    "color": ["colorbar", "clabel", "cticks"],
    "remark": ["remark_plot", "remark_inline"],
    "grid": ["grid_plot", "grid_inline"],
    "ref": ["ref_plot", "ref_inline"],
}

CHARTS = {
    "basic": ["scatter", "line", "barh", "bar"],
    "grid": ["pcolormesh", "pcolorfast", "contourf", "contour"],
    "ref": ["rectangle", "axvspan", "axhspan", "axvline", "axhline", "scatter"],
}
CHARTS["all"] = CHARTS["basic"] + CHARTS["grid"] + CHARTS["ref"]

PRESETS = {
    "scatter": ["trail"],
    **{chart: ["race", "delta", "series"] for chart in ["bar", "barh"]},
    **{chart: ["rotate", "scan_x", "scan_y"] for chart in CHARTS["grid"]},
}

DIMS = {
    "basic": (
        "item",
        "state",
    ),
    "grid": ("grid_item", "state", "grid_y", "grid_x"),
    "ref": ("ref_item", "state"),
    "item": ("grid_item", "item", "ref_item"),
}

VARS = {
    "ref": ["ref_x0", "ref_x1", "ref_y0", "ref_y1"],
    "stateless": [
        "chart",
        "group",
        "interp",
        "ease",
        "ref_label",
        "ref_chart",
        "grid_label",
        "grid_chart",
    ],
}

KWDS = {
    "canvas": [
        "figure_kwds",
        "suptitle_kwds",
        "watermark_kwds",
        "compute_kwds",
        "animate_kwds",
        "durations_kwds",
        "frame_kwds",
        "margins_kwds",
        "spacing_kwds",
        "output_kwds",
    ],  # figure-wide
    "geo": [
        "borders_kwds",
        "coastline_kwds",
        "land_kwds",
        "lakes_kwds",
        "ocean_kwds",
        "rivers_kwds",
        "states_kwds",
    ],  # geographic features
    "transform": [
        "plot_kwds",
        "inline_kwds",
        "ref_plot_kwds",
        "ref_inline_kwds",
        "grid_plot_kwds",
        "grid_inline_kwds",
        "preset_kwds",
        "preset_inline_kwds",
        "grid_kwds",
        "margins_kwds",
    ],  # transform
}

ITEMS = {
    "axes": ["x", "y", "c", "grid_c"],
    "limit": [
        "xlim0s",
        "xlim1s",
        "ylim0s",
        "ylim1s",
        "xlims",
        "ylims",
    ],
    "label": ["xlabel", "ylabel", "title", "subtitle"],
    "base": [
        "inline",
        "state",
        "delta",
        "ref_inline",
        "grid_inline",
        "grid_scan_x_0_inline",
        "grid_scan_y_0_inline",
    ],  # base magnitude
    "interpolate": ["interp", "ease"],
    "datasets": [
        "annual_co2",
        "tc_tracks",
        "covid19_us_cases",
        "covid19_global_cases",
        "gapminder_life_expectancy",
        "gapminder_income",
        "gapminder_population",
        "gapminder_country"
    ],
    "join": ["overlay", "layout", "cascade"],
}

OPTIONS = {
    "fmt": ["gif", "mp4", "jpg", "png"],
    "style": ["graph", "minimal", "bare"],
    "legend": [
        'upper left',
        'upper right',
        'lower left',
        'lower right',
        'right',
        'center left',
        'center right',
        'lower center',
        'upper center',
        'center',
        True,
        False
    ],
    "grid": [
        'x',
        'y',
        'both',
        True,
        False
    ],
    "limit": ["zero", "fixed", "follow", "explore"],
    "scheduler": ["processes", "single-threaded"],
}

SIZES = {
    "xx-small": 9,
    "x-small": 11,
    "small": 13,
    "medium": 16,
    "large": 20,
    "x-large": 28,
    "xx-large": 36,
    "xxx-large": 48,
}


DEFAULTS = {}

DEFAULTS["durations_kwds"] = {
    "aggregate": "max",
    "transition_frames": 1 / 60,
    "final_frame": 0.54,
}

DEFAULTS["label_kwds"] = {
    "fontsize": SIZES["medium"],
    "replacements": {"_": " "},
}

DEFAULTS["preset_kwds"] = {}
DEFAULTS["preset_kwds"]["trail_kwds"] = {
    "chart": "scatter",
    "expire": 100,
    "stride": 1,
}
DEFAULTS["preset_kwds"]["race_kwds"] = {"bar_label": True, 'limit': 8}
DEFAULTS["preset_kwds"]["delta_kwds"] = {"bar_label": True, "capsize": 6}
DEFAULTS["preset_kwds"]["scan_kwds"] = {"color": "black", "stride": 1}

DEFAULTS["ref_plot_kwds"] = {}
DEFAULTS["ref_plot_kwds"]["rectangle"] = {
    "facecolor": "darkgray",
    "alpha": 0.45,
}
DEFAULTS["ref_plot_kwds"]["scatter"] = {
    "color": "black",
}
DEFAULTS["ref_plot_kwds"]["axvline"] = {"color": "darkgray", "linestyle": "--"}
DEFAULTS["ref_plot_kwds"]["axhline"] = {"color": "darkgray", "linestyle": "--"}
DEFAULTS["ref_plot_kwds"]["axvspan"] = {"color": "darkgray", "alpha": 0.45}
DEFAULTS["ref_plot_kwds"]["axhspan"] = {"color": "darkgray", "alpha": 0.45}

DEFAULTS["inline_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["inline_kwds"].update(
    {"color": "darkgray", "textcoords": "offset pixels"}
)
DEFAULTS["ref_inline_kwds"] = DEFAULTS["inline_kwds"].copy()
DEFAULTS["grid_inline_kwds"] = DEFAULTS["inline_kwds"].copy()
DEFAULTS["preset_inline_kwds"] = DEFAULTS["inline_kwds"].copy()

DEFAULTS["remark_inline_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["remark_inline_kwds"].update(
    {
        "fontsize": SIZES["small"],
        "textcoords": "offset points",
        "xytext": (1, -1),
        "ha": "left",
        "va": "top",
    }
)

DEFAULTS["xlabel_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["xlabel_kwds"].update({"fontsize": SIZES["medium"]})

DEFAULTS["ylabel_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["ylabel_kwds"].update({"fontsize": SIZES["medium"]})

DEFAULTS["clabel_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["clabel_kwds"].update({"fontsize": SIZES["medium"]})

DEFAULTS["title_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["title_kwds"].update({"fontsize": SIZES["large"], "loc": "left"})

DEFAULTS["subtitle_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["subtitle_kwds"].update({"fontsize": SIZES["small"], "loc": "right"})

DEFAULTS["note_kwds"] = {
    "x": 0.05,
    "y": 0.05,
    "ha": "left",
    "va": "top",
    "fontsize": SIZES["xx-small"],
}

DEFAULTS["caption_kwds"] = {
    "x": 0,
    "y": -0.28,
    "alpha": 0.7,
    "ha": "left",
    "va": "bottom",
    "fontsize": SIZES["x-small"],
}

DEFAULTS["suptitle_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["suptitle_kwds"].update(
    {
        "fontsize": SIZES["large"],
    }
)

DEFAULTS["state_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["state_kwds"].update(
    {
        "alpha": 0.5,
        "xy": (0.988, 0.01),
        "ha": "right",
        "va": "bottom",
        "xycoords": "axes fraction",
        "fontsize": SIZES["xxx-large"],
    }
)

DEFAULTS["inline_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["inline_kwds"].update({"textcoords": "offset points"})

DEFAULTS["legend_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["legend_kwds"].update(
    {"show": True, "framealpha": 0, "loc": "upper left"}
)

DEFAULTS["colorbar_kwds"] = {"orientation": "vertical", "extend": "both"}

DEFAULTS["ticks_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["ticks_kwds"].pop("fontsize")
DEFAULTS["ticks_kwds"].update(
    {"length": 0, "which": "both", "labelsize": SIZES["small"]}
)

DEFAULTS["xticks_kwds"] = DEFAULTS["ticks_kwds"].copy()
DEFAULTS["xticks_kwds"].update({"axis": "x"})

DEFAULTS["yticks_kwds"] = DEFAULTS["ticks_kwds"].copy()
DEFAULTS["yticks_kwds"].update({"axis": "y"})

DEFAULTS["cticks_kwds"] = DEFAULTS["ticks_kwds"].copy()
DEFAULTS["cticks_kwds"].update({"num_colors": 11, "num_ticks": 12})

DEFAULTS["coastline_kwds"] = {"coastline": True}  # TODO: change to show

DEFAULTS["land_kwds"] = {"facecolor": "whitesmoke"}

DEFAULTS["watermark_kwds"] = {
    "x": 0.995,
    "y": 0.005,
    "alpha": 0.28,
    "ha": "right",
    "va": "bottom",
    "fontsize": SIZES["xx-small"],
    "s": "animated using ahlive",
}

DEFAULTS["frame_kwds"] = {
    "format": "jpg",
    "backend": "agg",
    "transparent": False,
}

DEFAULTS["compute_kwds"] = {"num_workers": 4, "scheduler": "processes"}

DEFAULTS["animate_kwds"] = {"mode": "I", "loop": 0}

sizes = SIZES.copy()
defaults = DEFAULTS.copy()


class Configuration(param.Parameterized):

    attrs = None

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _set_config(self, attrs, obj_label, val_key=None, mpl_key=None):
        if val_key is None:
            val_key = obj_label
        value = getattr(self, val_key)
        if value in NULL_VALS:
            return

        if mpl_key is None:
            mpl_key = obj_label
        obj_key = f"{obj_label}_kwds"
        if mpl_key not in attrs[obj_key]:
            attrs[obj_key][mpl_key] = value
        return attrs

    def _initial_config(self, attrs, obj_label):
        if obj_label == "figure":
            self._set_config(
                attrs, obj_label, val_key="figsize", mpl_key="figsize"
            )
        elif obj_label == "axes":
            self._set_config(attrs, obj_label, val_key="style", mpl_key="style")
        elif obj_label == "title":
            self._set_config(attrs, obj_label, mpl_key="label")
        elif obj_label == "subtitle":
            self._set_config(attrs, obj_label, mpl_key="label")
        elif obj_label == "suptitle":
            self._set_config(attrs, obj_label, mpl_key="t")
        elif obj_label == "note":
            self._set_config(attrs, obj_label, mpl_key="s")
        elif obj_label == "caption":
            self._set_config(attrs, obj_label, mpl_key="s")
        elif obj_label == "watermark":
            self._set_config(attrs, obj_label, mpl_key="s")
        elif obj_label == "legend":
            self._set_config(attrs, obj_label, mpl_key="show")
        elif obj_label == "grid":
            self._set_config(attrs, obj_label, mpl_key="show")
        elif obj_label == "xticks":
            self._set_config(attrs, obj_label, mpl_key="ticks")
        elif obj_label == "yticks":
            self._set_config(attrs, obj_label, mpl_key="ticks")
        elif obj_label == "projection":
            self._set_config(attrs, obj_label)
            self._set_config(
                attrs,
                obj_label,
                mpl_key="central_longitude",
                val_key="central_lon",
            )
        elif obj_label == "clabel":
            self._set_config(attrs, obj_label, mpl_key="text")
        elif obj_label == "colorbar":
            self._set_config(attrs, obj_label, mpl_key="show")
        elif obj_label == "cticks":
            self._set_config(attrs, obj_label, mpl_key="ticks")
            self._set_config(
                attrs, obj_label, val_key="ctick_labels", mpl_key="tick_labels"
            )
        elif obj_label == "compute":
            self._set_config(
                attrs, obj_label, val_key="workers", mpl_key="num_workers"
            )
            self._set_config(
                attrs, obj_label, val_key="scheduler", mpl_key="scheduler"
            )
        elif obj_label == "interpolate":
            self._set_config(
                attrs, obj_label, val_key="revert", mpl_key="revert"
            )
            self._set_config(
                attrs, obj_label, val_key="frames", mpl_key="frames"
            )
        elif obj_label == "animate":
            self._set_config(attrs, obj_label, val_key="fps", mpl_key="fps")
            self._set_config(attrs, obj_label, val_key="fmt", mpl_key="format")
            self._set_config(attrs, obj_label, val_key="loop", mpl_key="loop")
        elif obj_label == "output":
            self._set_config(attrs, obj_label, val_key="save", mpl_key="save")
            self._set_config(attrs, obj_label, val_key="show", mpl_key="show")
        elif obj_label == "margins":
            self._set_config(attrs, obj_label, val_key="xmargins", mpl_key="x")
            self._set_config(attrs, obj_label, val_key="ymargins", mpl_key="y")
        elif hasattr(self, obj_label):
            self._set_config(attrs, obj_label)
        return attrs

    def _config_data(
        self, input_data, *obj_labels, rowcols=None, reset=False, **config_kwds
    ):
        if rowcols is None:
            rowcols = input_data.keys()

        all_configurables = list(chain(*self.configurables.values()))
        if obj_labels:
            if len(obj_labels) > 1:
                raise ValueError(
                    "Cannot support multiple positional args! "
                    "Use one of the following structures:\n\n"
                    "For single object configurations:\n"
                    '\tconfig("obj", key=value)\n'
                    '\tconfig("obj", **{"key": "value"})\n\n'
                    "For multiple objects configurations:\n"
                    '\tconfig(obj1={"key": "value"}, obj2={"key": "value"})\n'
                    "\tconfig(**{"
                    '"obj1": {"key": "value"}, '
                    '"obj2": {"key": "value"}})\n'
                )
            configurables = obj_labels
        elif config_kwds.keys():
            configurables = list(config_kwds.keys())
        else:
            configurables = all_configurables

        data = {}
        for rowcol, ds in input_data.items():
            if rowcol not in rowcols:
                continue

            attrs = ds.attrs or {}
            for obj_label in configurables:
                if obj_label not in all_configurables:
                    if obj_label == "chart":
                        continue
                    raise KeyError(
                        f"{obj_label} is invalid; select from the following: "
                        f'{", ".join(all_configurables)}'
                    )

                obj_key = f"{obj_label}_kwds"
                if "configured" not in attrs:
                    attrs["configured"] = {}

                if obj_key not in attrs or reset:
                    attrs["configured"][obj_label] = False
                    attrs[obj_key] = {}

                if obj_labels:  # e.g. config('inline', format='%.0f')
                    obj_vals = config_kwds
                else:  # e.g. config(inline={'format': '%.0f'})
                    obj_vals = config_kwds.get(obj_label, attrs[obj_key])

                if not isinstance(obj_vals, dict):
                    raise ValueError(f"{obj_vals} must be a dict!")

                if len(obj_vals) > 0:
                    attrs[obj_key].update(**obj_vals)

                is_configured = attrs["configured"].get(obj_label, False)
                if is_configured or obj_label in obj_labels:
                    attrs = self._initial_config(attrs, obj_label)
                attrs["configured"][obj_label] = True

            ds.attrs.update(**attrs)
            data[rowcol] = ds

        if data:
            return data
        else:
            return input_data

    def config(self, *obj_labels, rowcols=None, reset=False, **config_kwds):
        self_copy = deepcopy(self)
        data = self._config_data(
            self_copy.data,
            rowcols=rowcols,
            reset=reset,
            *obj_labels,
            **config_kwds,
        )

        self_copy.data = data
        return self_copy


def scale_sizes(scale, keys=None):
    if keys is None:
        keys = sizes.keys()

    for key in keys:
        sizes[key] = sizes[key] * scale


def load_defaults(default_key, input_kwds=None, **other_kwds):
    # get default values
    updated_kwds = DEFAULTS.get(default_key, {}).copy()

    # unnest dictionary if need
    if default_key in [
        "preset_kwds",
        "plot_kwds",
        "ref_plot_kwds",
        "grid_plot_kwds",
    ]:
        updated_kwds = updated_kwds.get(
            f'{other_kwds.pop("base_chart", "")}_kwds', {}
        ).copy()
    if isinstance(input_kwds, xr.Dataset):
        input_kwds = input_kwds.attrs.get(default_key, {})

    # update with programmatically generated values
    updated_kwds.update(
        {key: val for key, val in other_kwds.items() if val is not None}
    )

    # update with user input values
    if input_kwds is not None:
        updated_kwds.update(
            {key: val for key, val in input_kwds.items() if val is not None}
        )
    updated_kwds.pop("preset", None)
    updated_kwds.pop("base_chart", None)
    return updated_kwds


def update_defaults(default_key, **kwds):
    defaults[default_key].update(**kwds)
