#calculate average base map
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
import sunpy.map as smap
#from sunpy.physics.differential_rotation import differential_rotate
from aiapy.calibrate import register, update_pointing
from aiapy.calibrate.util import get_pointing_table
from astropy import units as u
import copy

##########################################################################################
#support functions
def read_aia_map(filename):
    """
    Read an AIA FITS file and perform preprocessing equivalent to IDL aia_prep.
    """
    m = smap.Map(filename)

    # update pointing and register (resample to common plate scale & align centers)
    pointing_table = get_pointing_table("LMSAL", time_range=(m.date - 12 * u.h,m.date + 12 * u.h))
    m_updated_pointing = update_pointing(m, pointing_table=pointing_table)
    m = register(m_updated_pointing)

    # normalize by exposure time
    if hasattr(m, "exposure_time"):
        m = smap.Map((m.data / m.exposure_time.to_value("s")), m.meta)
    else:
        print("WARNING: No EXPOSURE_TIME keyword found — normalization skipped.")

    # convert to float as data was normalized by exposure time
    m = smap.Map(m.data.astype(np.float32), m.meta)

    return m


def rebin_map(m, nx, ny):
    """
    Resample a SunPy map to (ny, nx) pixels.
    """
    new_dimensions = [ny, nx] * u.pixel
    return m.resample(new_dimensions)


def drot_map(m, ref_map):
    """
    Differentially rotate map 'm' to the time of 'ref_map'.
    """
    #t_ref = ref_map.date
    #return differential_rotate(m, time=t_ref)
    out_wcs=ref_map.wcs
    mtemp=m.reproject_to(out_wcs)
    return mtemp

##########################################################################################
# main routine

def mean_base_map(files, dimension, n_base, ref=None):
    """
    Parameters
    ----------
    files : list of str
        List of AIA FITS filenames.
    dimension : int
        Output image size (e.g. 4096).
    n_base : int
        Number of frames used for base image.
    ref : sunpy.map.Map (optional)
        Reference map for differential rotation.

    Returns
    -------
    base : sunpy.map.Map
        Mean base map (rebinned + derotated)
    """

    # use first file as initial pre_base
    base_file = files[0]
    pre_base = read_aia_map(base_file)
    pre_base = rebin_map(pre_base, dimension, dimension)

    # differential rotation to supplied ref (if applicable)
    if ref is not None:
        pre_base = drot_map(pre_base, ref)

    # prepare base array
    base_arr = np.zeros((n_base, dimension, dimension), dtype=np.float32)
    base_arr[0, :, :] = pre_base.data

    # loop over remaining n_base−1 maps
    for m in range(1, n_base):
        fname = files[m]
        pre_map = read_aia_map(fname)
        pre_map = rebin_map(pre_map, dimension, dimension)

        # apply differential rotation, if ref is not supplied use pre_base
        if ref is not None:
            pre_map = drot_map(pre_map, ref)
        else:
            pre_map = drot_map(pre_map, pre_base)

        base_arr[m, :, :] = pre_map.data
        print("Processed frame", m)

    # compute average
    av = np.mean(base_arr, axis=0)

    # return final averaged base map
    base = smap.Map(av.astype(np.float32), copy.deepcopy(pre_base.meta))

    return base