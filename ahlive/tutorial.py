import os
import glob

import pandas as pd
import xarray as xr


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'data')
DATASETS = sorted(glob.glob(os.path.join(DATA_DIR, '*')))
ALIASES = {os.path.splitext(os.path.basename(fp))[0]: fp for fp in DATASETS}


def load_data(label=None):
    if label is None:
        raise ValueError(f'Select a dataset to open: {list(ALIASES)}')
    fp = ALIASES.get(label, label)
    if fp.endswith('.pkl'):
        return pd.read_pickle(fp)
    elif fp.endswith('.nc'):
        return xr.open_dataset(fp)
    else:
        raise NotImplementedError
