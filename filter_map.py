#apply median filtering to SDO/AIA maps for plotting and further analysis depending on their wavelength
# (c) K. Dissauer, University of Graz, 2025

import sunpy.map as smap
from scipy.ndimage import median_filter
import copy

def filter_map(sratio_log, wave):
    """
    Parameters
    ----------
    sratio_log : sunpy.map.Map
        Input SunPy map to be filtered.
    wave : int
        AIA wavelength (e.g., 171, 193, 211, 94, 131).

    Returns
    -------
    map_show : sunpy.map.Map
        Filtered copy of sratio_log for plotting only.
    sratio_log_filtered : sunpy.map.Map
        Filtered version of sratio_log.
    """

    # wavelengths that require stronger smoothing
    strong_smooth = [94, 131, 304, 335]

    map_show = smap.Map(copy.deepcopy(sratio_log.data), copy.deepcopy(sratio_log.meta))

    #apply median filter for plotting only
    map_show_data = median_filter(map_show.data, size=3)
    map_show = smap.Map(map_show_data, copy.deepcopy(sratio_log.meta))

    #apply median filter for data as well, choose smoothing depending on wavelength
    if wave in strong_smooth:
        size = 10
    else:
        size = 3

    sratio_filtered_data = median_filter(sratio_log.data, size=size)

    # return as sunpy map
    sratio_log_filtered = smap.Map(sratio_filtered_data, copy.deepcopy(sratio_log.meta))

    return map_show, sratio_log_filtered