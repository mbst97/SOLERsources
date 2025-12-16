#select dimming pixel based on thresholding and location on solar disk
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
from scipy import ndimage

def select_dimming_pixel(ratio, angle, ratio_thresh, angle_thresh):
    """

    Parameters
    ----------
    ratio : sunpy.map.Map or np.ndarray
        Ratio map (e.g. base-ratio dimming map).
    angle : sunpy.map.Map or np.ndarray
        Angle map (in degrees).
    ratio_thresh : float
        Threshold on the ratio for dimming detection (ratio <= ratio_thresh).
    angle_thresh : float
        Threshold on the angle for dimming detection (angle <= angle_thresh).
    Returns
    -------
    dim_pixel : tuple of np.ndarray or None
        (y_indices, x_indices) of dimming pixels after region growing.
        Returns None if no dimming region is found.
    dim_count : int
        Number of dimming pixels.
    """

    if hasattr(ratio, "data"):
        ratio_data = ratio.data
    else:
        ratio_data = np.asarray(ratio)

    if hasattr(angle, "data"):
        angle_data = angle.data
    else:
        angle_data = np.asarray(angle)

    ratio_data = np.asarray(ratio_data, dtype=float)
    angle_data = np.asarray(angle_data, dtype=float)

    # select dimming pixel based on log. base ratio threshold and angle requirement
    dim_mask = (ratio_data <= ratio_thresh) & (angle_data <= angle_thresh)
    dim_indices = np.where(dim_mask)
    pre_count = dim_indices[0].size

    if pre_count == 0:
        dim_pixel = None
        dim_count = 0
        return dim_pixel, dim_count

    # build dimming mask and ROI mask
    dimming_mask = np.zeros_like(ratio_data, dtype=bool)
    dimming_mask[dim_indices] = True

    roi_mask = dimming_mask.copy()

    # apply morphological closing and opening to mask for region growing
    structure_close = np.ones((3, 3), dtype=bool)
    structure_open = np.ones((30, 30), dtype=bool)

    roi_mask = ndimage.binary_closing(roi_mask, structure=structure_close)
    roi_mask = ndimage.binary_opening(roi_mask, structure=structure_open)

    roi_pixels = np.where(roi_mask)
    roi_count = roi_pixels[0].size

    # if no dimming pixels or ROI pixels are detected return
    if pre_count == 0 or roi_count == 0:
        dim_pixel = None
        dim_count = 0
        return dim_pixel, dim_count

    # apply region growing through binary propagation and 8-connected neighborhood:
    #   - dimming_mask is 1 where possible dimming exists
    #   - roi_mask is the ROI after morph. ops (seed region)
    # seeds must lie within dimming_mask
    seed_mask = roi_mask & dimming_mask

    # 8-connected neighbors in 2D:
    neighbors = ndimage.generate_binary_structure(2, 2)

    grown = ndimage.binary_propagation(seed_mask, mask=dimming_mask, structure=neighbors)

    dim_pixel = np.where(grown)
    dim_count = dim_pixel[0].size

    if dim_count == 0:
        dim_pixel = None
        
    # #testing
    # #dim_pixel is (y_indices, x_indices)
    # y_idx, x_idx = dim_pixel

    # # shape from grown mask (or any same-shape map)
    # ny, nx = grown.shape

    # # create mask of zeros
    # dim_mask_binary = np.zeros((ny, nx), dtype=np.uint8)

    # # set dimming pixels to 1
    # dim_mask_binary[y_idx, x_idx] = 1

    return dim_pixel, dim_count