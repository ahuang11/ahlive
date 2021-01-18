from copy import deepcopy
from itertools import chain

import param
import xarray as xr

NULL_VALS = [(), {}, [], None, ""]

CONFIGURABLES = {  # used for like .config('figure', **kwds)
    "canvas": [
        "figure",
        "suptitle",
        "compute",
        "animate",
        "durations",
        "frame",
        "watermark",
        "spacing",
        "fontscale",
        "interpolate",
        "output",
    ],
    "subplot": [
        "axes",
        "plot",
        "preset",
        "legend",
        "grid",
        "xticks",
        "yticks",
        "margins",
    ],
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

GROUPS = {  # for each config group, maps param alias to function key
    "figure": [{"param": "figsize", "fn_key": "figsize"}],
    "axes": [{"param": "style", "fn_key": "style"}],
    "title": [{"param": "title", "fn_key": "label"}],
    "subtitle": [{"param": "subtitle", "fn_key": "label"}],
    "suptitle": [{"param": "suptitle", "fn_key": "t"}],
    "note": [{"param": "note", "fn_key": "s"}],
    "caption": [{"param": "caption", "fn_key": "s"}],
    "watermark": [{"param": "watermark", "fn_key": "s"}],
    "legend": [{"param": "legend", "fn_key": "show"}],
    "grid": [{"param": "grid", "fn_key": "show"}],
    "xticks": [{"param": "xticks", "fn_key": "ticks"}],
    "yticks": [{"param": "yticks", "fn_key": "ticks"}],
    "projection": [
        {"param": "projection", "fn_key": "projection"},
        {"param": "central_lon", "fn_key": "central_longitude"},
    ],
    "clabel": [
        {"param": "clabel", "fn_key": "text"},
    ],
    "colorbar": [{"param": "colorbar", "fn_key": "show"}],
    "cticks": [
        {"param": "cticks", "fn_key": "ticks"},
        {"param": "ctick_labels", "fn_key": "tick_labels"},
    ],
    "compute": [
        {"param": "workers", "fn_key": "num_workers"},
        {"param": "scheduler", "fn_key": "scheduler"},
    ],
    "interpolate": [
        {"param": "revert", "fn_key": "revert"},
        {"param": "frames", "fn_key": "frames"},
    ],
    "animate": [
        {"param": "fps", "fn_key": "fps"},
        {"param": "fmt", "fn_key": "format"},
        {"param": "loop", "fn_key": "loop"},
    ],
    "output": [
        {"param": "save", "fn_key": "save"},
        {"param": "show", "fn_key": "show"},
    ],
    "margins": [
        {"param": "xmargins", "fn_key": "x"},
        {"param": "ymargins", "fn_key": "y"},
    ],
}

POINTERS = {
    param["param"]: configurable
    for configurable in GROUPS
    for param in GROUPS[configurable]
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
        "label",
        "group",
        "interp",
        "ease",
        "ref_label",
        "ref_chart",
        "grid_label",
        "grid_chart",
    ],
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
        "grid_scan_x_diff_inline",
        "grid_scan_y_diff_inline",
    ],  # base magnitude
    "interpolate": ["interp", "ease"],
    "datasets": [
        "annual_co2",
        "tc_tracks",
        "covid19_us_cases",
        "covid19_global_cases",
        "covid19_population",
        "gapminder_life_expectancy",
        "gapminder_income",
        "gapminder_population",
        "gapminder_country",
    ],
    "join": ["overlay", "layout", "cascade"],
    "transformables": [
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
    ],
}

OPTIONS = {
    "fmt": ["gif", "mp4", "jpg", "png"],
    "style": ["graph", "minimal", "bare"],
    "legend": [
        "upper left",
        "upper right",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
        True,
        False,
    ],
    "grid": ["x", "y", "both", True, False],
    "limit": ["zero", "fixed", "follow", "explore"],
    "scheduler": ["processes", "single-threaded"],
}

SIZES = {
    "xx-small": 9,
    "x-small": 11,
    "small": 13,
    "medium": 16,
    "large": 19,
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
DEFAULTS["preset_kwds"]["trail"] = {
    "chart": "scatter",
    "expire": 100,
    "stride": 1,
}
DEFAULTS["preset_kwds"]["race"] = {"bar_label": True, "limit": 5}
DEFAULTS["preset_kwds"]["delta"] = {"bar_label": True, "capsize": 6}
DEFAULTS["preset_kwds"]["scan"] = {"color": "black", "stride": 1}

DEFAULTS["ref_plot_kwds"] = {}
DEFAULTS["ref_plot_kwds"]["rectangle"] = {
    "facecolor": "darkgray",
    "alpha": 0.45,
}
DEFAULTS["ref_plot_kwds"]["scatter"] = {"color": "darkgray", "marker": "x"}
DEFAULTS["ref_plot_kwds"]["axvline"] = {"color": "darkgray", "linestyle": "--"}
DEFAULTS["ref_plot_kwds"]["axhline"] = {"color": "darkgray", "linestyle": "--"}
DEFAULTS["ref_plot_kwds"]["axvspan"] = {"color": "darkgray", "alpha": 0.45}
DEFAULTS["ref_plot_kwds"]["axhspan"] = {"color": "darkgray", "alpha": 0.45}

DEFAULTS["inline_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["inline_kwds"].update(
    {
        "color": "darkgray",
        "textcoords": "offset points",
        "fontsize": SIZES["small"],
    }
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
DEFAULTS["legend_kwds"].update({"framealpha": 0, "loc": "upper left"})

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
DEFAULTS["cticks_kwds"].update({"num_colors": 11})

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

defaults = DEFAULTS.copy()


class Configuration(param.Parameterized):

    attrs = None

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _set_config(self, attrs, obj_label, param=None, fn_key=None):
        if param is None:
            param = obj_label
        value = getattr(self, param)
        if value in NULL_VALS:
            return

        if fn_key is None:
            fn_key = obj_label
        obj_key = f"{obj_label}_kwds"
        if fn_key not in attrs[obj_key]:
            attrs[obj_key][fn_key] = value
        return attrs

    def _initial_config(self, attrs, obj_label):
        groups_list = GROUPS.get(obj_label)
        if groups_list is not None:
            for config_kwds in groups_list:
                self._set_config(attrs, obj_label, **config_kwds)
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
                if not is_configured or obj_label in obj_labels:
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
            f'{other_kwds.pop("base_chart", "")}', {}
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
