#select new dimming pixel
# (c) K. Dissauer, University of Graz, 2025

import numpy as np

def new_dimming_pixel(dim_pixel, dim_pixel_arr):
    """
    Parameters
    ----------
    dim_pixel : array-like or None
        Flat pixel indices of the current dimming detection.
        If no dimming: dim_pixel = None.
    dim_pixel_arr : array-like or None
        Accumulated dimming pixels from previous timestamps.

    Returns
    -------
    ndp : np.ndarray
        Newly dimmed pixels (flat indices).
        Empty array if none.
    """

    #if dim pixel is None return empy 
    if dim_pixel is None:
        return np.array([], dtype=int)

    dim_pixel = np.asarray(dim_pixel, dtype=int)

    # if initial detection return dim pixel or empty array if dim pixel is None
    if dim_pixel_arr is None or len(dim_pixel_arr) == 0:
        if len(dim_pixel) > 0:
            print("initial dimming")
            return dim_pixel.copy()
        else:
            return np.array([], dtype=int)

    # existing dimming pixel array
    dim_pixel_arr = np.asarray(dim_pixel_arr, dtype=int)

    # select only dim_pixel that have not been part of dim_pixel_arr yet
    mask_new = ~np.isin(dim_pixel, dim_pixel_arr)

    # Extract new pixels
    new_dimming = dim_pixel[mask_new]

    return new_dimming