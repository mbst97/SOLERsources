#check exposure time of SDO/AIA files
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
from astropy.io import fits

def check_exptime(in_files):
    """
    Parameters
    ----------
    in_files : list of str or Path

    Returns
    -------
    out_files : list
        Files whose exposure time is in the allowed range (e.g., 1.8–3.0 s)
    times : list
        Corresponding DATE-OBS timestamps
    cm_exptime : str
        Comment string summarizing results
    """
    # read headers of .fits files and extract times and exposure times
    alltimes = []
    exptimes = []

    for f in in_files:
        with fits.open(f, memmap=True) as hdul:
            hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
            alltimes.append(hdr.get("DATE-OBS", ""))
            exptimes.append(hdr.get("EXPTIME", np.nan))

    alltimes = np.array(alltimes)
    exptimes = np.array(exptimes, dtype=float)

    # select files based on exposure time cirteria
    crit_exp = np.where((exptimes >= 1.8) & (exptimes <= 3.0))[0]

    if crit_exp.size == len(in_files):
        print("all files fulfill the exposure time criteria")
        cm_exptime = f"exptime={crit_exp.size}({len(in_files)})"
        out_files = list(in_files)
        times = alltimes
    else:
        print(f"{crit_exp.size} of {len(in_files)} fulfill the exposure time criteria")
        cm_exptime = f"exptime={crit_exp.size}({len(in_files)})"
        out_files = [in_files[i] for i in crit_exp]
        times = alltimes[crit_exp]

    return out_files, times, cm_exptime
