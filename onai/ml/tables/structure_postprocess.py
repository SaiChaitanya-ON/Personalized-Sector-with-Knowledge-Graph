import logging

import numpy as np
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu

from .table import Table

logger = logging.getLogger(__name__)

PAD = 0.02


def _clean_response_1D(resp, pad=PAD):
    """
    Cleans 1D response based on thresholding function and padding ratio
    Padding ration controls the fraction of samples ignored for thresholding around edges.
    Example. pad = 2% means that 2% of samples on either end of response will be ignored
    for thresholding (e.g. building value historgrams).
    """
    pad = int(resp.shape[0] * pad)
    resp[:pad] = 0
    if pad != 0:
        resp[-pad:] = 0
    try:
        t = threshold_otsu(resp)
    except ValueError:
        logger.exception("Exception thresholding with Otsu; setting .0")
        t = 0.0
    resp = resp.clip(min=t)
    return resp


def _argpeak_with_edges(response, p=PAD):
    """
    Findes peaks in 1D response AND includes edges
    Pad controls the fraction of samples ignored around edges (edges are known 0 and response.shape[0]-1)
    """
    pad = max(int(response.shape[0] * p), 1)
    return np.array(
        [
            0,
            *(find_peaks(response[pad:-pad], distance=pad)[0] + pad),
            response.shape[0] - 1,
        ]
    )


def _boundaries(resp, pad=PAD):
    r = _clean_response_1D(resp, pad)
    r = _argpeak_with_edges(r, pad)
    return r


def table_grid(row_hm: np.ndarray, col_hm: np.ndarray, pad: float = PAD) -> Table:
    cols = _boundaries(col_hm, pad=pad)
    rows = _boundaries(row_hm, pad=pad)
    table = Table(rows, cols)
    return table
