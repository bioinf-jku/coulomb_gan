'''
Various utility functions
Copyright © 2016 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import math

# Importing matplotlib might fail under special conditions
# e.g. when using ssh w/o X11 forwarding
try:
    import matplotlib.pyplot as plt
except ImportError:
    import warnings
    warnings.warn("matplotlib unavailable")


def startup_bookkeeping(logdir, curent_file):
    '''
    Performs common operations at start of each run.
    Writes PID and argv into files in the logdir and copies all the source
    files in the from current directory there as well.
    logdir is expected to be a pathlib.Path
    '''

    import shutil
    from pathlib import Path

    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    with open(logdir / "pid", 'w') as f:  # write PID so we can abort from outside
        f.write(str(os.getpid()))
        f.write("\n")

    # make copy of current source files
    dest = logdir / 'sources'
    dest.mkdir(exist_ok=True)
    localdir = Path(curent_file).parent
    pyfiles = localdir.glob("*.py")
    for fn in localdir.glob("*.py"):
        shutil.copy(fn, dest / fn)

    with open(logdir / "argv", 'w') as f:
        f.write(' '.join(sys.argv))
    f.write("\n")


def plot_tiles(data, nrows=8, ncols=8, ax=None, local_norm="maxabs", data_format='NWHC', **kwargs):
    ''' Plots several images stored in data.
    Data is assumed to be in NWHC format. If it is shaped wrongly, we will
    try to heuristically determine the correct thing.
    TODO: for now, only square images are supported
    '''
    assert data_format in ('NWHC', 'NCWH'), "Unknown data format"

    n = data.shape[0]
    if len(data.shape) == 2:  # flattened array
        a = np.sqrt(data.shape[-1])
        if a == int(a): # size is square, so probably just 1 channel
            w, h, c = int(a), int(a), 1
        else: # try assuming 3 channels
            a = np.sqrt(data.shape[-1]//3)
            if a == int(a):
                w, h, c = int(a), int(a), 3
            else:
                raise RuntimeError("Unable to determine shape of images")
    elif len(data.shape) == 3: # single color channel
        w, h, c = data.shape[1], data.shape[2], 1
    elif len(data.shape) == 4:
        n, w, h, c = data.shape
    else:
        raise RuntimeError("Unable to determine shape of images")

    if data_format == 'NWHC':
        data = data.reshape(n, w, h, c)
    else:
        data = data.reshape(n, c, w, h)
        data = data.transpose(0, 2, 3, 1)  # rest of code always assumes NWHC


    assert c == 1 or c == 3, "Data must be in NWHC format"
    assert w == h, "Only square images are supported"

    ppi = w + 2  # +2 for borders
    imgshape = (nrows*ppi, ncols*ppi, c)
    # make sure border is black
    img = {"maxabs": lambda s: (data.min() / np.abs(data).max()) * np.ones(imgshape, dtype=data.dtype),
           "minmax": lambda s: np.zeros(imgshape, dtype=data.dtype),
           "none":   lambda s: np.ones(imgshape, dtype=data.dtype)*data.min()
            }[local_norm.lower()](None)

    n = min(nrows*ncols, data.shape[0])
    normfunc = {"maxabs": lambda d: d / np.abs(d).max(),
                "minmax": lambda d: (d - d.min()) / d.ptp(),
                "none":   lambda d: d}[local_norm.lower()]
    idx = 0
    for ri in range(nrows):
        for ci in range(ncols):
            if idx >= n:
                break
            img[ri*ppi+1:(ri+1)*ppi-1, ci*ppi+1:(ci+1)*ppi-1] = normfunc(data[idx])
            idx += 1
    if ax is None:
        fig = plt.figure(facecolor="black", **kwargs)
        fig.subplots_adjust(hspace=0, top=1, bottom=0,
                            wspace=0, left=0, right=1)
        ax = fig.gca()
    else:
        fig = None
    if c > 1:
        ax.imshow(img, interpolation="none")
    else:
        ax.imshow(img.reshape(nrows*ppi, ncols*ppi),
                    interpolation="none", cmap="Greys_r")
    ax.axis("off")
    return fig


def get_timestamp(fmt='%y%m%d_%H%M'):
    '''Returns a string that contains the current date and time.
    Suggested formats:
        short_format=%y%m%d_%H%M  (default)
        long format=%Y%m%d_%H%M%S
    '''
    import datetime
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, fmt)
