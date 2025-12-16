# select flare pixel
# (c) K. Dissauer, University of Graz, 2025


import numpy as np
from sunpy.map import Map
from typing import Optional, Tuple

from skimage.morphology import binary_closing, binary_opening

def select_flare_pixel(
    smap: Map,
    flare_thresh: float,
    angle_map: Optional[Map] = None,
    use_morphology: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Parameters
    ----------
    smap : sunpy.map.Map
        Intensity map (equivalent to IDL 'map').
    flare_thresh : float
        Threshold on intensity.
    angle_map : sunpy.map.Map, optional
        If provided (IDL 'angle' keyword), only pixels with angle <= 60 deg
        are allowed.
    use_morphology : bool, optional
        If True, apply morphological closing and opening as in the
        commented-out IDL code.

    Returns
    -------
    flare_pixel : np.ndarray
        1D array of flattened indices of selected flare pixels.
        If no pixels detected, returns array([-1]) to mimic IDL behavior.
    flare_count : int
        Number of detected pixels.
    """
    
    data = np.asarray(smap.data, dtype=float)


    if angle_map is not None:
        angle = np.asarray(angle_map.data, dtype=float)
        mask = (data >= flare_thresh) & (angle <= 60.0)
    else:
        mask = data >= flare_thresh

    flare = np.where(mask.ravel())[0]
    pre_count = flare.size

    if use_morphology and pre_count != 0:
        # Build binary mask of flare pixels
        flare_mask = np.zeros_like(data, dtype=bool)
        flare_mask.ravel()[flare] = True

        # 3x3 structuring elements equivalent
        flare_mask = binary_closing(flare_mask)
        flare_mask = binary_opening(flare_mask)

        flare = np.where(flare_mask.ravel())[0]
        pre_count = flare.size

    if pre_count != 0:
        #print("detection possible")
        flare_pixel = flare.astype(int)
        flare_count = int(pre_count)
    else:
        flare_pixel = np.array([-1], dtype=int)
        flare_count = 0

    return flare_pixel, flare_count
