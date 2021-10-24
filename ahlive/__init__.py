from pkg_resources import DistributionNotFound, get_distribution

from .configuration import CONFIGURABLES, DEFAULTS, PRESETS, config_defaults  # noqa
from .data import Array, Array2D, DataFrame, Dataset, Reference  # noqa
from .easing import Easing  # noqa
from .join import cascade, layout, merge, overlay, slide, stagger  # noqa
from .tutorial import list_datasets, open_dataset  # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover
