# newly detected flare pixel from one time step to the next
# (c) K. Dissauer, University of Graz, 2025

import numpy as np

def new_flare_pixel(pixel, pixel_arr):
    """

    Parameters
    ----------
    pixel : array-like (1D)
        Current flare pixel indices (flattened). May contain [-1] meaning none.
    pixel_arr : array-like (1D)
        Array of all previously detected pixel indices.

    Returns
    -------
    ndp : numpy.ndarray
        Pixels detected *newly* at this timestep.
            - If first detection: returns `pixel`
            - If none new: returns empty array
    """

    pixel = np.asarray(pixel, dtype=int)
    pixel_arr = np.asarray(pixel_arr, dtype=int)

    num = pixel_arr.size

    #if first detection returns pixel
    if num == 0 and pixel.size > 0 and pixel[0] != -1:
        print("initial detection")
        return pixel.copy()

    if num != 0:
        # select only pixels that were never detected before
        new_mask = ~np.isin(pixel, pixel_arr)

        if not np.any(new_mask):
            ndp = np.array([], dtype=int)
        else:
            ndp = pixel[new_mask]

        return ndp

    # fallback (should not happen)
    return np.array([], dtype=int)
