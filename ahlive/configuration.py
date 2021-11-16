import os
from copy import deepcopy
from itertools import chain

import matplotlib.pyplot as plt
import param
import xarray as xr
from cycler import cycler

# https://github.com/matplotlib/matplotlib/issues/9460#issuecomment-875185352
colors = [
    "#bd1f01",
    "#3f90da",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#ffa90e",
    "#b9ac70",
    "#92dadd",
    "#717581",
    "#94a4a2",
]
plt.rc("axes", prop_cycle=cycler("color", colors))

TEMP_FILE = "TEMP_AHLIVE_PYGIFSICLE_OUTPUT.gif"
NULL_VALS = [(), {}, [], None, "", "nan"]  # np.nan is not picked up here

PRECEDENCES = [
    "data",
    "xyc",
    "common",
    "export",
    "label",
    "limit",
    "style",
    "geo",
    "animate",
    "sub_label",
    "ticks",
    "interp",
    "pool",
    "misc",
    "attr",
]
PRECEDENCES = {label: precedence for precedence, label in enumerate(PRECEDENCES)}

INTERPS = [
    "fill",
    "linear",
    "cubic",
    "exponential",
    "quadratic",
    "quartic",
    "quintic",
    "circular",
    "sine",
    "bounce",
    "elastic",
    "back",
]
EASES = ["in_out", "in", "out"]

REVERTS = ["boomerang", "traceback", "rollback"]

# a kind of grouping by intuition; doesn't really help code though
CONFIGURABLES = {  # used for like .config('figure', **kwds)
    "canvas": [
        "figure",
        "suptitle",
        "pool",
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
        "adjust_text",
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
        "tiles",
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
# this maps ahlive's configurable's params to matplotlib keys
# key may match value if ahlive's param name matches matplotlib
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
        "projection": {
            "projection": "projection",
            "central_lon": "central_longitude",
        },
        "tiles": {"tiles": "tiles", "zoom": "zoom"},
        "clabel": {
            "clabel": "text",
        },
        "colorbar": {"colorbar": "show"},
        "cticks": {"cticks": "ticks", "ctick_labels": "tick_labels"},
        "pool": {
            "workers": "max_workers",
            "scheduler": "scheduler",
            "progress": "progress",
        },
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

CANVAS = {
    param
    for param, configurable in PARAMS.items()
    if configurable in CONFIGURABLES["canvas"]
}

CHARTS = {
    "basic": [
        "scatter",
        "line",
        "barh",
        "bar",
        "pie",
        "errorbar",
        "area",
        "annotation",
    ],
    "grid": [
        "pcolormesh",
        "pcolorfast",
        "contourf",
        "contour",
        "hexbin",
        "quiver",
        "streamplot",
        "windbarb",
    ],
    "ref": ["rectangle", "axvspan", "axhspan", "axvline", "axhline", "scatter"],
}
CHARTS["mpl"] = {
    "line": "plot",
    "area": "fill_between",
    "annotation": "annotate",
    "windbarb": "barbs",
}
CHARTS["all"] = CHARTS["basic"] + CHARTS["grid"] + CHARTS["ref"]

DIMS = {
    "basic": (
        "item",
        "state",
    ),
    "grid": ("grid_item", "state", "grid_y", "grid_x"),
    "ref": ("ref_item", "state"),
}

VARS = {
    "ref": ["ref_x0", "ref_x1", "ref_y0", "ref_y1"],
    "item": ("grid_item", "item", "ref_item"),
    "stateless": [
        "chart",
        "group",
        "interp",
        "ease",
        "ref_chart",
        "grid_chart",
    ],
    "itemless": ["state_label", "ref_last_item"],
}

# internal item mappings
ITEMS = {
    "axes": ["x", "y", "c", "grid_c"],
    "limit": ["xlim0s", "xlim1s", "ylim0s", "ylim1s", "xlims", "ylims"],
    "label": ["xlabel", "ylabel", "title", "subtitle"],
    "base": [
        "bar",
        "tick",
        "inline",
        "state",
        "delta",
        "ref_inline",
        "grid_inline",
        "grid_scan_x_diff_inline",
        "grid_scan_y_diff_inline",
    ],  # base magnitude
    "interpolate": ["interp", "ease"],
    "transformables": [
        "plot_kwds",
        "inline_kwds",
        "text_inline_kwds",
        "ref_plot_kwds",
        "ref_inline_kwds",
        "remark_inline_kwds",
        "remark_plot_kwds",
        "grid_plot_kwds",
        "grid_inline_kwds",
        "preset_kwds",
        "preset_inline_kwds",
        "grid_kwds",
        "margins_kwds",
    ],
    "continual_charts": ["line", "errorbar", "area"],  # need more than one data point
    "uv_charts": ["quiver", "streamplot", "windbarb"],
    "bar_charts": ["bar", "barh"],  # need more than one data point
    "not_scalar": ["labels", "xerr", "yerr", "y2", "u", "v"],
}

PRESETS = {
    "trail": ["scatter", "annotation"],
    "morph": ["scatter"] + ITEMS["continual_charts"] + ITEMS["bar_charts"],
    "morph_trail": ["scatter"] + ITEMS["continual_charts"],
    "morph_stacked": ITEMS["bar_charts"],
    "rotate": ["scatter"] + ITEMS["continual_charts"] + CHARTS["grid"],
    "scan_x": ["scatter"] + ITEMS["continual_charts"] + CHARTS["grid"],
    "scan_y": ["scatter"] + ITEMS["continual_charts"] + CHARTS["grid"],
    "race": ITEMS["bar_charts"],
    "delta": ITEMS["bar_charts"],
    "stacked": ITEMS["bar_charts"],
}

# frontend facing options
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
    "join": ["overlay", "layout", "cascade", "stagger", "slide"],
    "limit": ["zero", "fixed", "follow", "explore"],
    "scheduler": ["threads", "processes"],
    "state_xy": [
        "title",
        "subtitle",
        "suptitle",
        "title_start",
        "subtitle_start",
        "suptitle_start",
    ],
    "iem_tz": {
        "utc": "Etc/UTC",
        "akst": "America/Anchorage",
        "wst": "America/Los_Angeles",
        "mst": "America/Denver",
        "cst": "America/Chicago",
        "est": "America/New_York",
    },
    "iem_data": [
        "tmpf",
        "tmpc",
        "dwpf",
        "dwpc",
        "relh",
        "feel",
        "drct",
        "sknt",
        "sped",
        "alti",
        "mslp",
        "p01m",
        "p01i",
        "vsby",
        "gust",
        "gust_mph",
        "skyc1",
        "skyc2",
        "skyc3",
        "skyl1",
        "skyl2",
        "skyl3",
        "wxcodes",
        "ice_accretion_1hr",
        "ice_accretion_3hr",
        "ice_accretion_6hr",
        "peak_wind_gust",
        "peak_wind_gust_mph",
        "peak_wind_drct",
        "peak_wind_time",
        "snowdepth",
        "metar",
    ],
}

SIZES = {
    "xx-small": 9,
    "x-small": 11,
    "small": 14.8,
    "medium": 16,
    "large": 19,
    "x-large": 28,
    "xx-large": 36,
    "xxx-large": 48,
}

DEFAULTS = {}

DEFAULTS["durations_kwds"] = {
    "aggregate": "max",
    "transition_frames": 1 / 45,
    "final_frame": 0.55,
}

DEFAULTS["label_kwds"] = {
    "fontsize": SIZES["medium"],
    "replacements": {"_": " "},
    "color": "#262626",
    "comma": False,
}

DEFAULTS["preset_kwds"] = {}
DEFAULTS["preset_kwds"]["trail"] = {
    "chart": "scatter",
    "expire": 100,
    "stride": 1,
    "zorder": 0,
}
DEFAULTS["preset_kwds"]["morph_trail"] = {
    "chart": "line",
    "expire": 1,
    "stride": 1,
    "zorder": 0,
}
DEFAULTS["preset_kwds"]["race"] = {
    "bar_label": True,
    "limit": 5,
    "ascending": False,
    "annotation_clip": True,
}
DEFAULTS["preset_kwds"]["delta"] = {"bar_label": True, "capsize": 6}
DEFAULTS["preset_kwds"]["scan"] = {"color": "black", "stride": 1}

DEFAULTS["plot_kwds"] = {}
DEFAULTS["plot_kwds"]["pie"] = {"normalize": False}
DEFAULTS["plot_kwds"]["scatter"] = {"alpha": 0.9}
DEFAULTS["plot_kwds"]["bar"] = {"alpha": 0.9}
DEFAULTS["plot_kwds"]["barh"] = {"alpha": 0.9}

DEFAULTS["grid_plot_kwds"] = {}
DEFAULTS["grid_plot_kwds"]["streamline"] = {"zorder": 2}
DEFAULTS["grid_plot_kwds"]["quiver"] = {"zorder": 2}
DEFAULTS["grid_plot_kwds"]["barbs"] = {"zorder": 2}

DEFAULTS["ref_plot_kwds"] = {}
DEFAULTS["ref_plot_kwds"]["rectangle"] = {
    "facecolor": "#696969",
    "alpha": 0.45,
}
DEFAULTS["ref_plot_kwds"]["scatter"] = {"color": "#696969", "marker": "x"}
DEFAULTS["ref_plot_kwds"]["axvline"] = {"color": "#696969", "linestyle": "--"}
DEFAULTS["ref_plot_kwds"]["axhline"] = {"color": "#696969", "linestyle": "--"}
DEFAULTS["ref_plot_kwds"]["axvspan"] = {"color": "#696969", "alpha": 0.45}
DEFAULTS["ref_plot_kwds"]["axhspan"] = {"color": "#696969", "alpha": 0.45}

DEFAULTS["remark_plot_kwds"] = {"marker": "x", "persist": True}

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
    "y": 0.08,
    "ha": "left",
    "va": "top",
    "width": 75,
    "fontsize": SIZES["xx-small"],
}

DEFAULTS["caption_kwds"] = {
    "x": 0,
    "y": -0.28,
    "alpha": 0.7,
    "ha": "left",
    "va": "bottom",
    "width": 90,
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

DEFAULTS["text_inline_kwds"] = DEFAULTS["inline_kwds"].copy()

DEFAULTS["ref_inline_kwds"] = DEFAULTS["inline_kwds"].copy()
DEFAULTS["grid_inline_kwds"] = DEFAULTS["inline_kwds"].copy()
DEFAULTS["preset_inline_kwds"] = DEFAULTS["inline_kwds"].copy()
DEFAULTS["preset_inline_kwds"]["clip_on"] = True

DEFAULTS["remark_inline_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["remark_inline_kwds"].update(
    {
        "textcoords": "offset points",
        "xytext": (-8, -5),
        "ha": "right",
        "va": "top",
        "width": 38,
        "persist": False,
    }
)

DEFAULTS["margins_kwds"] = {"x": 0.03, "y": 0.03}

DEFAULTS["adjust_text_kwds"] = {"precision": 1}

DEFAULTS["legend_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["legend_kwds"]["labelcolor"] = DEFAULTS["legend_kwds"].pop("color")
DEFAULTS["legend_kwds"].update(
    {"framealpha": 0.28, "facecolor": "whitesmoke", "loc": "upper left"}
)

DEFAULTS["colorbar_kwds"] = {"orientation": "vertical"}

DEFAULTS["ticks_kwds"] = DEFAULTS["label_kwds"].copy()
DEFAULTS["ticks_kwds"]["labelcolor"] = DEFAULTS["ticks_kwds"].pop("color")
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

DEFAULTS["grid_kwds"] = {"show": True, "alpha": 0.28, "color": "lightgray"}

DEFAULTS["tiles_kwds"] = {"style": "toner"}  # TODO: change to show

DEFAULTS["land_kwds"] = {"facecolor": "whitesmoke"}

DEFAULTS["num_kwds"] = {
    "default": 1,
    "bounds": (1, None),
    "constant": True,
    "precedence": PRECEDENCES["attr"],
}

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

DEFAULTS["pool_kwds"] = {
    "max_workers": 2,
    "scheduler": "processes",
    "progress": True,
}

DEFAULTS["animate_kwds"] = {"mode": "I", "loop": 0, "pygifsicle": True}

DEFAULTS["cache_kwds"] = {"directory": os.path.expandvars("$HOME/.ahlive/")}

ORIGINAL_DEFAULTS = DEFAULTS.copy()


class CartopyCRS(param.ClassSelector):

    __slots__ = ["crs_dict"]

    def __init__(self, default=None, **params):
        try:
            import cartopy.crs as ccrs

            self.crs_dict = {
                name.lower(): obj
                for name, obj in vars(ccrs).items()
                if isinstance(obj, type)
                and issubclass(obj, ccrs.Projection)
                and not name.startswith("_")
                and name not in ["Projection"]
                or name == "GOOGLE_MERCATOR"
            }
            objects = tuple(list(self.crs_dict.values()) + [str, bool])
            super(CartopyCRS, self).__init__(objects, **params)
            self._validate(self.default)
        except ImportError:
            self.crs_dict = {}
            super(CartopyCRS, self).__init__(None, **params)

    def _validate(self, val):
        if val is None:
            return

        elif len(self.crs_dict) == 0:
            raise ImportError(
                "cartopy is not installed for geographic support. "
                "To install: `conda install -c conda-forge cartopy`"
            )

        elif isinstance(val, str):
            crs_names = self.crs_dict.keys()
            if val.lower() not in crs_names:
                raise ValueError(f"Expected one of these {crs_names}; got {val}")


class CartopyFeature(param.ClassSelector):
    def __init__(self, default=None, **params):
        try:
            import cartopy.feature as cfeature

            objects = (cfeature.NaturalEarthFeature, bool)
            super(CartopyFeature, self).__init__(objects, **params)
            self._validate(self.default)
        except ImportError:
            super(CartopyFeature, self).__init__(None, **params)


class CartopyTiles(param.ClassSelector):

    __slots__ = ["tiles_dict"]

    def __init__(self, default=None, **params):
        try:
            import cartopy.io.img_tiles as ctiles

            self.tiles_dict = {
                name.lower(): obj
                for name, obj in vars(ctiles).items()
                if isinstance(obj, type)
                and issubclass(obj, ctiles.GoogleWTS)
                and not name.startswith("_")
            }
            objects = tuple(list(self.tiles_dict.values()) + [str, bool])
            super(CartopyTiles, self).__init__(objects, **params)
            self._validate(self.default)
        except ImportError:
            self.tiles_dict = {}
            super(CartopyTiles, self).__init__(None, **params)


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

    @staticmethod
    def _parse_configurables(configurables, configurable_kwds):
        select_configurables = None
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
        return select_configurables

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
        select_configurables = self._parse_configurables(
            configurables, configurable_kwds
        )
        if select_configurables is None:
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

                if not configurable.endswith("_kwds"):
                    configurable_key = f"{configurable}_kwds"
                else:
                    configurable_key = configurable
                    configurable = configurable.replace("_kwds", "")

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


def config_defaults(*default_key, **default_kwds):
    """
    See ah.DEFAULTS.items() for available settings.
    """
    select_defaults = Configuration._parse_configurables(default_key, default_kwds)

    for default_key in select_defaults:
        valid_keys = DEFAULTS.keys()
        if not default_key.endswith("_kwds"):
            original_default_key = default_key
            default_key = f"{default_key}_kwds"
            if original_default_key in default_kwds:
                default_kwds[default_key] = default_kwds.pop(original_default_key)

        if default_key is not None and default_key not in valid_keys:
            raise KeyError(f"{default_key} must be one of: {valid_keys}")

        if default_key in default_kwds.keys():
            DEFAULTS[default_key].update(**default_kwds[default_key])
        else:
            DEFAULTS[default_key].update(**default_kwds)
