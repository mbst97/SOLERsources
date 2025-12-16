#load all flare fits files into one array for further time evolution analysis
# (c) K. Dissauer, University of Graz, 2025

from pathlib import Path
import numpy as np
from astropy.io import fits
from sunpy.map import Map


def flare_map_array(event, wave, data_path, outpath):

    str_wave = str(wave).strip()

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
   
    # load times array from info fits file
    info_file = data_path / f"{event}_info.fits"
    hdul = fits.open(info_file)

    # the table is in the first extension (HDU index 1)
    table = hdul[1].data

    times  = np.array(table['DATE_OBS'])

    hdul.close()

    #change in case not all files will be used
    st = 0
    en = len(times)
    times = times[st:en]
    num = len(times)

    # load each sunpy map from fits and store in python list
    map_list = []

    for i in range(num):
        i_str = f"{i + st:03d}"   # zero-padded 3 digits

        map_file = data_path / f"{event}_maps_{i_str}_{str_wave}.fits"

        # load fits file into a sunpy map
        fmap = Map(str(map_file))

        map_list.append(fmap)

        print(f"Loaded map {i}: {map_file}")


    # save output as .npz for easy reuse
    out_file = outpath / f"{event}_maps_arr.npz"
    np.savez(out_file, map_arr=map_list, times=times)

    print(f"\nSaved sunpy map array → {out_file}")

#################################################################################
# Example call:
# event='20240509_1653'
# wave=1600
# data_path = '/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/maps/'
# outpath = '/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/'

# flare_map_array(event, wave, data_path, outpath)
