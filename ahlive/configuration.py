from copy import deepcopy
from itertools import chain

import param
import xarray as xr

NULL_VALS = [(), {}, [], None, ""]

CONFIGURABLES = {
    "canvas": ["figure", "suptitle", "watermark", "spacing", "margins"],
    "subplot": ["axes", "plot", "preset", "legend", "grid", "xticks", "yticks"],
    "label": [
        "state",
        "inline",
        "xlabel",
        "ylabel",
        "title",
        "subtitle",
        "note",
        "preset_inline",
    ],
    "meta": ["frame", "animation", "compute", "durations"],
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
    "type": ["race", "delta", "trail", "rotate", "scan", "scan_x", "scan_y"],
}

DIMS = {
    "basic": ("item", "state",),
    "grid": ("grid_item", "state", "grid_y", "grid_x"),
    "ref": ("ref_item", "state"),
    "item": ("grid_item", "item", "ref_item"),
}

VARS = {
    "ref": ["ref_x0", "ref_x1", "ref_y0", "ref_y1"],
    "stateless": [
        "chart",
        "ref_label",
        "ref_chart",
        "grid_label",
        "grid_chart",
    ],
}
ITEMS = {
    "fmt": ["gif", "mp4"],
    "limit": [
        "xlims0",
        "xlims1",
        "ylims0",
        "ylims1",
        "xlim0",
        "xlim1",
        "ylim0",
        "ylim1",
    ],
    "figurewide": [
        "figure",
        "suptitle",
        "watermark",
        "compute",
        "animation",
        "durations",
        "frame",
        "spacing",
    ],  # figure-wide
    "base": [
        "inline",
        "state",
        "preset",
        "ref_inline",
        "grid_inline",
    ],  # base magnitude
    "geo": [
        "borders",
        "coastline",
        "land",
        "lakes",
        "ocean",
        "rivers",
        "states",
    ],  # geographic features
    "transform": [
        "plot",
        "inline",
        "ref_plot",
        "ref_inline",
        "grid_plot",
        "grid_inline",
        "preset",
        "preset_inline",
        "grid",
        "axes",
    ],  # transform
}

OPTIONS = {
    "style": ["graph", "minimal", "bare"],
    "limit": ["fixed", "follow"],
    "scheduler": ["processes", "single-threaded"],
}


SIZES = {
    "xx-small": 12,
    "x-small": 17,
    "small": 22,
    "medium": 26,
    "large": 30,
    "x-large": 36,
    "xx-large": 60,
    "xxx-large": 84,
}

DEFAULTS = {}

DEFAULTS["durations"] = {
    "aggregate": "max",
    "transition_frames": 1 / 60,
    "final_frame": 1,
}

DEFAULTS["spacing"] = {
    "left": 0.05,
    "right": 0.925,
    "bottom": 0.1,
    "top": 0.9,
    "wspace": 0.2,
    "hspace": 0.2,
}

DEFAULTS["label"] = {
    "fontsize": SIZES["medium"],
    "replacements": {"_": " "},
    "format": "auto",
}

DEFAULTS["plot"] = {}
DEFAULTS["plot"]["bar"] = {"bar_label": True}
DEFAULTS["plot"]["barh"] = DEFAULTS["plot"]["bar"].copy()

DEFAULTS["preset"] = {}
DEFAULTS["preset"]["trail"] = {"expire": 100, "stride": 1}
DEFAULTS["preset"]["delta"] = {"capsize": 6}
DEFAULTS["preset"]["scan"] = {"color": "black", "stride": 1}

DEFAULTS["ref_plot"] = {}
DEFAULTS["ref_plot"]["rectangle"] = {"facecolor": "darkgray", "alpha": 0.45}
DEFAULTS["ref_plot"]["scatter"] = {
    "color": "black",
}
DEFAULTS["ref_plot"]["axvline"] = {"color": "darkgray", "linestyle": "--"}
DEFAULTS["ref_plot"]["axhline"] = {"color": "darkgray", "linestyle": "--"}
DEFAULTS["ref_plot"]["axvspan"] = {"color": "darkgray", "alpha": 0.45}
DEFAULTS["ref_plot"]["axhspan"] = {"color": "darkgray", "alpha": 0.45}

DEFAULTS["inline"] = DEFAULTS["label"].copy()
DEFAULTS["inline"].update({"color": "darkgray", "textcoords": "offset points"})
DEFAULTS["ref_inline"] = DEFAULTS["inline"].copy()
DEFAULTS["grid_inline"] = DEFAULTS["inline"].copy()
DEFAULTS["preset_inline"] = DEFAULTS["inline"].copy()

DEFAULTS["remark_inline"] = DEFAULTS["label"].copy()
DEFAULTS["remark_inline"].update(
    {
        "fontsize": SIZES["small"],
        "textcoords": "offset points",
        "xytext": (0, 1.5),
        "ha": "left",
        "va": "top",
    }
)

DEFAULTS["xlabel"] = DEFAULTS["label"].copy()
DEFAULTS["xlabel"].update({"fontsize": SIZES["large"], "casing": "title"})

DEFAULTS["ylabel"] = DEFAULTS["label"].copy()
DEFAULTS["ylabel"].update({"fontsize": SIZES["large"], "casing": "title"})

DEFAULTS["clabel"] = DEFAULTS["label"].copy()
DEFAULTS["clabel"].update({"fontsize": SIZES["large"], "casing": "title"})

DEFAULTS["title"] = DEFAULTS["label"].copy()
DEFAULTS["title"].update({"fontsize": SIZES["large"], "loc": "left"})

DEFAULTS["subtitle"] = DEFAULTS["label"].copy()
DEFAULTS["subtitle"].update({"fontsize": SIZES["medium"], "loc": "right"})

DEFAULTS["note"] = {
    "x": 0.01,
    "y": 0.05,
    "ha": "left",
    "va": "top",
    "fontsize": SIZES["x-small"],
}

DEFAULTS["caption"] = {
    "x": 0,
    "y": -0.28,
    "alpha": 0.7,
    "ha": "left",
    "va": "bottom",
    "fontsize": SIZES["x-small"],
}

DEFAULTS["suptitle"] = DEFAULTS["label"].copy()
DEFAULTS["suptitle"].update({"fontsize": SIZES["large"]})

DEFAULTS["state"] = DEFAULTS["label"].copy()
DEFAULTS["state"].update(
    {
        "alpha": 0.5,
        "xy": (0.988, 0.01),
        "ha": "right",
        "va": "bottom",
        "xycoords": "axes fraction",
        "fontsize": SIZES["xxx-large"],
    }
)

DEFAULTS["inline"] = DEFAULTS["label"].copy()
DEFAULTS["inline"].update({"textcoords": "offset points"})

DEFAULTS["legend"] = DEFAULTS["label"].copy()
DEFAULTS["legend"].update({"show": True, "framealpha": 0, "loc": "upper left"})

DEFAULTS["colorbar"] = {"orientation": "vertical", "extend": "both"}

DEFAULTS["ticks"] = DEFAULTS["label"].copy()
DEFAULTS["ticks"].pop("fontsize")
DEFAULTS["ticks"].update(
    {"length": 0, "which": "both", "labelsize": SIZES["small"]}
)

DEFAULTS["xticks"] = DEFAULTS["ticks"].copy()
DEFAULTS["xticks"].update({"axis": "x"})

DEFAULTS["yticks"] = DEFAULTS["ticks"].copy()
DEFAULTS["yticks"].update({"axis": "y"})

DEFAULTS["cticks"] = DEFAULTS["ticks"].copy()
DEFAULTS["cticks"].update({"num_colors": 11, "num_ticks": 12})

DEFAULTS["coastline"] = {"coastline": True}  # TODO: change to show

DEFAULTS["land"] = {"facecolor": "whitesmoke"}

DEFAULTS["watermark"] = {
    "x": 0.995,
    "y": 0.005,
    "alpha": 0.28,
    "ha": "right",
    "va": "bottom",
    "fontsize": SIZES["xx-small"],
    "s": "animated using ahlive",
}

DEFAULTS["frame"] = {"format": "jpg", "backend": "agg", "transparent": False}

DEFAULTS["compute"] = {"num_workers": 4, "scheduler": "processes"}

DEFAULTS["animation"] = {
    "format": "gif",
    "mode": "I",
}

sizes = SIZES.copy()
defaults = DEFAULTS.copy()


class Configuration(param.Parameterized):

    attrs = None

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _set_config(self, attrs, obj_key, val_key=None, mpl_key=None):
        if val_key is None:
            val_key = obj_key
        value = getattr(self, val_key)
        if value in NULL_VALS:
            return

        if mpl_key is None:
            mpl_key = obj_key
        attrs[obj_key][mpl_key] = value
        return attrs

    def _final_config(self, attrs, obj_key):
        if obj_key == "figure":
            self._set_config(
                attrs, obj_key, val_key="figsize", mpl_key="figsize"
            )
        elif obj_key == "axes":
            self._set_config(attrs, obj_key, val_key="style", mpl_key="style")
        elif obj_key == "title":
            self._set_config(attrs, obj_key, mpl_key="label")
        elif obj_key == "subtitle":
            self._set_config(attrs, obj_key, mpl_key="label")
        elif obj_key == "note":
            self._set_config(attrs, obj_key, mpl_key="s")
        elif obj_key == "caption":
            self._set_config(attrs, obj_key, mpl_key="s")
        elif obj_key == "legend":
            self._set_config(attrs, obj_key, mpl_key="show")
        elif obj_key == "xticks":
            self._set_config(attrs, obj_key, mpl_key="ticks")
        elif obj_key == "yticks":
            self._set_config(attrs, obj_key, mpl_key="ticks")
        elif obj_key == "projection":
            self._set_config(attrs, obj_key)
            self._set_config(
                attrs,
                obj_key,
                mpl_key="central_longitude",
                val_key="central_lon",
            )
        elif obj_key == "clabel":
            self._set_config(attrs, obj_key, mpl_key="text")
        elif obj_key == "colorbar":
            self._set_config(attrs, obj_key, mpl_key="show")
        elif obj_key == "cticks":
            self._set_config(attrs, obj_key, mpl_key="ticks")
            self._set_config(
                attrs, obj_key, val_key="ctick_labels", mpl_key="tick_labels"
            )
        elif obj_key == "compute":
            self._set_config(
                attrs, obj_key, val_key="workers", mpl_key="num_workers"
            )
            self._set_config(
                attrs, obj_key, val_key="scheduler", mpl_key="scheduler"
            )
        elif obj_key == "animation":
            self._set_config(attrs, obj_key, val_key="fps", mpl_key="fps")
            self._set_config(attrs, obj_key, val_key="fmt", mpl_key="format")
        elif hasattr(self, obj_key):
            self._set_config(attrs, obj_key)
        attrs["configured"] = True
        return attrs

    def config(self, *obj_keys, rowcols=None, reset=False, **config_kwds):
        if rowcols is None:
            rowcols = self.data.keys()

        final_configure = False
        all_configurables = list(chain(*self.configurables.values()))
        if obj_keys:
            if len(obj_keys) > 1:
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
            configurables = obj_keys
        elif config_kwds.keys():
            configurables = list(config_kwds.keys())
        else:
            configurables = all_configurables
            final_configure = True

        data = {}
        self_copy = deepcopy(self)
        for rowcol, ds in self_copy.data.items():
            if rowcol not in rowcols:
                continue

            attrs = ds.attrs or {}
            for obj_key in configurables:
                if obj_key not in all_configurables:
                    raise KeyError(
                        f"{obj_key} is invalid; select from the following: "
                        f'{", ".join(all_configurables)}'
                    )

                if obj_key not in attrs or reset:
                    attrs[obj_key] = {}

                if obj_keys:
                    obj_vals = config_kwds
                else:
                    obj_vals = config_kwds.get(obj_key, attrs[obj_key])

                if not isinstance(obj_vals, dict):
                    raise ValueError(f"{obj_vals} must be a dict!")

                attrs[obj_key].update(obj_vals)
                if final_configure:
                    attrs = self._final_config(attrs, obj_key)

            ds.attrs.update(**attrs)
            data[rowcol] = ds

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
    if default_key in ["preset", "plot", "ref_plot", "grid_plot"]:
        updated_kwds = updated_kwds.get(
            other_kwds.pop("base_chart", None), {}
        ).copy()
    if isinstance(input_kwds, xr.Dataset):
        input_kwds = input_kwds.attrs[default_key]

    # update with programmatically generated values
    updated_kwds.update(
        {key: val for key, val in other_kwds.items() if val is not None}
    )

    # update with user input values
    if input_kwds is not None:
        updated_kwds.update(
            {key: val for key, val in input_kwds.items() if val is not None}
        )
    updated_kwds.pop("base_chart", None)
    return updated_kwds


def update_defaults(default_key, **kwds):
    defaults[default_key].update(**kwds)
