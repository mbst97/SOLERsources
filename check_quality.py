#check quality of SDO/AIA fits files
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
from astropy.io import fits

def check_quality(in_files):
    """
    Parameters
    ----------
    in_files : sequence of str or Path
        List of input AIA FITS files.

    Returns
    -------
    out_files : list
        Files with QUALITY == 0 (good files).
    cm_quality : str
        Comment string describing quality status.
    """

    # get quality keywords from fits file headers
    qualities = []
    for f in in_files:
        with fits.open(f, memmap=True) as hdul:
            # AIA often has data in extension 1, but header in 0 or 1
            hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
            qualities.append(hdr.get("QUALITY", 0))

    qualities = np.array(qualities, dtype=int)

    quality = qualities  # mirrors pre_index.quality

    crit = np.where(quality != 0)[0]
    crit_q = np.where(quality == 0)[0]

    if crit.size == 0:
        print("quality of all files is ok")
        cm_quality = "quality=0"
        out_files = list(in_files)
    else:
        print(f"{crit.size} files are damaged")
        cm_quality = f"{crit.size} files are damaged"
        out_files = [in_files[i] for i in crit_q]

    #print("out_files:", out_files)
    return out_files, cm_quality
