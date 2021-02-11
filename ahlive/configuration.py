from copy import deepcopy
from itertools import chain

import param
import xarray as xr

TEMP_FILE = "TEMP_AHLIVE_PYGIFSICLE_OUTPUT.gif"
NULL_VALS = [(), {}, [], None, ""]

# a kind of grouping by intuition; doesn't really help code though
CONFIGURABLES = {  # used for like .config('figure', **kwds)
    "canvas": [
        "figure",
        "suptitle",
        "compute",
        "animate",
        "durations",
        "savefig",
        "watermark",
        "spacing",
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
        "limits",
        "margins",
        "hooks",
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

# a kind of grouping by how the code is structured
# each group's key is the key ahlive uses in its method call
# for example, figure is suffixed with _kwds to become figure_kwds
# and is used in `plt.figure(**figure_kwds)`
CONFIGURABLES_KWDS = {
    configurable: {configurable: configurable}
    for configurable in list(chain(*CONFIGURABLES.values()))
}
# outer key is configurable
# inner key is param
# inner value is method_key
CONFIGURABLES_KWDS.update(
    {
        "figure": {"figsize": "figsize"},
        "axes": {"style": "style"},
        "title": {"title": "label"},
        "subtitle": {"subtitle": "label"},
        "suptitle": {"suptitle": "t"},
        "note": {"note": "s"},
        "caption": {"caption": "s"},
        "watermark": {"watermark": "s"},
        "legend": {"legend": "show"},
        "grid": {"grid": "show"},
        "xticks": {"xticks": "ticks"},
        "yticks": {"yticks": "ticks"},
        "limits": {
            "worldwide": "worldwide",
            "xlims": "xlims",
            "ylims": "ylims",
            "xlim0s": "xlim0s",
            "ylim0s": "ylim0s",
            "xlim1s": "xlim1s",
            "ylim1s": "ylim1s",
        },
        "projection": {"projection": "projection", "central_lon": "central_longitude"},
        "clabel": {"clabel": "text"},
        "colorbar": {"colorbar": "show"},
        "cticks": {"cticks": "ticks", "ctick_labels": "tick_labels"},
        "compute": {"workers": "num_workers", "scheduler": "scheduler"},
        "interpolate": {"revert": "revert", "frames": "frames"},
        "animate": {
            "fps": "fps",
            "fmt": "format",
            "loop": "loop",
            "pygifsicle": "pygifsicle",
        },
        "output": {"save": "save", "show": "show"},
        "margins": {"xmargins": "x", "ymargins": "y"},
    }
)

PARAMS = {
    param_: configurable
    for configurable in CONFIGURABLES_KWDS
    for param_ in CONFIGURABLES_KWDS[configurable]
}

CHARTS = {
    "basic": ["scatter", "line", "barh", "bar"],
    "grid": ["pcolormesh", "pcolorfast", "contourf", "contour"],
    "ref": ["rectangle", "axvspan", "axhspan", "axvline", "axhline", "scatter"],
}
CHARTS["all"] = CHARTS["basic"] + CHARTS["grid"] + CHARTS["ref"]

PRESETS = {
    "none": [None],
    "scatter": ["trail"],
    **{chart: ["race", "delta", "series"] for chart in ["bar", "barh"]},
    **{chart: ["rotate", "scan_x", "scan_y"] for chart in CHARTS["grid"]},
}
PRESETS["all"] = PRESETS["scatter"] + PRESETS["bar"] + PRESETS["pcolormesh"]

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
    "limit": ["xlim0s", "xlim1s", "ylim0s", "ylim1s", "xlims", "ylims"],
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
    "spacing": ["left", "right", "bottom", "top", "wspace", "hspace"],
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
    "scheduler": ["single-threaded", "processes"],
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
    "final_frame": 0.55,
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
    {"color": "darkgray", "textcoords": "offset points", "fontsize": SIZES["small"]}
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
DEFAULTS["suptitle_kwds"].update({"fontsize": SIZES["large"]})

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

DEFAULTS["savefig_kwds"] = {
    "format": "png",
    "backend": "agg",
    "facecolor": "white",
    "transparent": False,
}

DEFAULTS["compute_kwds"] = {"num_workers": 1, "scheduler": "single-threaded"}

DEFAULTS["animate_kwds"] = {"mode": "I", "loop": 0, "pygifsicle": True}

defaults = DEFAULTS.copy()


class Configuration(param.Parameterized):

    attrs = None

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _set_config(self, attrs, configurable, param_=None, method_key=None):
        if param_ is None:
            param_ = configurable
        elif not hasattr(self, param_):
            return

        value = getattr(self, param_)
        if value in NULL_VALS:
            return

        if method_key is None:
            method_key = configurable
        configurable_key = f"{configurable}_kwds"
        if method_key not in attrs[configurable_key]:
            attrs[configurable_key][method_key] = value
        return attrs

    def _initial_config(self, attrs, configurable):
        for param_, method_key in CONFIGURABLES_KWDS[configurable].items():
            self._set_config(attrs, configurable, param_=param_, method_key=method_key)
        return attrs

    def _config_data(
        self,
        input_data,
        *configurables,
        rowcols=None,
        reset=False,
        **configurable_kwds,
    ):
        if rowcols is None:
            rowcols = input_data.keys()

        all_configurables = list(chain(*self.configurables.values()))
        if configurables:
            if len(configurables) > 1:
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
            select_configurables = configurables
        elif configurable_kwds.keys():
            select_configurables = list(configurable_kwds.keys())
        else:
            select_configurables = all_configurables

        data = {}
        for rowcol, ds in input_data.items():
            if rowcol not in rowcols:
                continue

            attrs = ds.attrs or {}
            for configurable in select_configurables:
                if configurable not in all_configurables:
                    if configurable == "chart":
                        continue
                    raise KeyError(
                        f"{configurable} is invalid; select from the "
                        f'following: {", ".join(all_configurables)}'
                    )

                configurable_key = f"{configurable}_kwds"
                if "configured" not in attrs:
                    attrs["configured"] = {}

                if configurable_key not in attrs or reset:
                    attrs["configured"][configurable] = False
                    attrs[configurable_key] = {}

                if configurables:  # e.g. config('inline', format='%.0f')
                    obj_vals = configurable_kwds
                else:  # e.g. config(inline={'format': '%.0f'})
                    obj_vals = configurable_kwds.get(
                        configurable, attrs[configurable_key]
                    )

                if not isinstance(obj_vals, dict):
                    raise ValueError(f"{obj_vals} must be a dict!")

                if len(obj_vals) > 0:
                    attrs[configurable_key].update(**obj_vals)

                is_configured = attrs["configured"].get(configurable, False)
                if not is_configured or configurable in configurables:
                    attrs = self._initial_config(attrs, configurable)
                attrs["configured"][configurable] = True

            ds.attrs.update(**attrs)
            data[rowcol] = ds

        if data:
            return data
        else:
            return input_data

    def config(self, *configurables, rowcols=None, reset=False, **configurable_kwds):
        self_copy = deepcopy(self)
        data = self._config_data(
            self_copy.data,
            rowcols=rowcols,
            reset=reset,
            *configurables,
            **configurable_kwds,
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
