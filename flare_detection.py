# detect flare ribbon pixels following Kazachenko et al. 2017 (https://iopscience.iop.org/article/10.3847/1538-4357/aa7ed6/pdf) 
# and create instantaneous and cumulative flare masks, flare time map and flare heat map
# (c) K. Dissauer, University of Graz, 2025

import os

import numpy as np
from astropy.io import fits

from astropy.time import Time
from sunpy.map import Map
from select_flare_pixel import select_flare_pixel
from new_flare_pixel import new_flare_pixel
from flare_plot_quicklook import flare_plot_quicklook
from flare_plot_heat_time_map import flare_plot_heat_time_map
import copy
import glob
from pathlib import Path

# support functions
def findel(value, array):

    array = np.asarray(array)
    return int(np.argmin(np.abs(array - value)))

def coords_to_idx(coords, width, height):

    x = coords[0]
    y = coords[1]
    # rows = y, cols = x, shape = (height, width)
    return np.ravel_multi_index((y, x), dims=(height, width), order='C')

def linear_indices_to_coords_unravel(indices, width, height):
    
    i = np.asarray(indices)
    y, x = np.unravel_index(i, (height, width))
    return np.vstack((x, y))

##################################################################################
# main program

def flare_detection(event, wave, fl_tm, en_tm, outpath, data_path, path_sat_maps):

    fl_tm=Time(fl_tm)
    en_tm=Time(en_tm)
    
    print(fl_tm)
    print(en_tm)
    
    str_wave = str(wave).strip()
    
    # intensity range for quicklook images
    drange = (10, 100)
    
    #define directory for quicklook output
    outimg = outpath / "quicklook" / str_wave
    os.makedirs(outimg, exist_ok=True)
    
    
    # load base vector magnetic field map for reference
    vmf_file = data_path / f"{event}_hmi_vector_magnetic_field.fits"
    br_map = Map(str(vmf_file))
    
    # load times
    info_file = data_path / f"{event}_info.fits"
    hdul = fits.open(info_file)
    table = hdul[1].data
    
    times  = np.array(table['DATE_OBS'])
    
    hdul.close()
    
    # determine indices of flare interval
    times_astropy = Time(times)
    
    fl_ind = findel(times_astropy.mjd, fl_tm.mjd)
    en_ind = findel(times_astropy.mjd, en_tm.mjd)
    
    fl_times = times_astropy[fl_ind:en_ind + 1]
    num = len(fl_times)
    
    # define masks
    mask_data= np.full_like(br_map.data, np.nan, dtype=float)
    time_map_data = np.full_like(br_map.data, np.nan, dtype=float)
    heat_map_data = np.full_like(br_map.data, 0.0, dtype=float)
    
    # create instantaneous masks
    masks_inst = [Map(copy.deepcopy(mask_data), copy.deepcopy(br_map.meta)) for _ in range(num)]
    for i, t in enumerate(fl_times):
        # store time information in the meta/header
        masks_inst[i].meta["time"] = t.isot
    
    # define constantly growing array with indices of already detected flaring pixels
    flare_pixel_arr = np.array([], dtype=int)
    
    # check if saturation corrected maps exist
    sat_files=sorted(path_sat_maps.glob(f"{event}_saturation_corrected_map*.fits"))
    
    if len(sat_files) != 0: 
        sat_corr=1
    else:
        sat_corr=0
        
    # flare pixel detection for each frame
    for i in range(num):
        frame_index = i + fl_ind

        ind_str = f"{frame_index:03d}"
    
        print(fl_times[i].isot)
    
        # if saturation corrected maps exist use them, otherwise use original maps
        if sat_corr ==1:
            file = path_sat_maps / f"{event}_saturation_corrected_map_{ind_str}_{str_wave}.fits"
        else:
            file = data_path / f"{event}_maps_{ind_str}_{str_wave}.fits"
        
        smap = Map(str(file))
    
        #  define threshold
        med = np.nanmedian(smap.data.astype(float))
        flare_thresh = med * 8.0  # (Kazachenko+2017, Qiu+2002)
     
        # select flare pixels
        flare_pixel, flare_count = select_flare_pixel(smap, flare_thresh)
    
        # define newly detected flare pixels
        if flare_count != 0:
            ndp = new_flare_pixel(flare_pixel, flare_pixel_arr)
            n_ndp = ndp.size
        else:
            ndp = np.array([], dtype=int)
            n_ndp = 0
    
        # create time map
        if n_ndp != 0:
            ndp_coords=linear_indices_to_coords_unravel(ndp, br_map.data.shape[1], br_map.data.shape[0])
            time_map_data[ndp_coords[1,:], ndp_coords[0,:]]=float(i)
    
        # create instantaneous masks and heat map
        if flare_count != 0:
            # set instantaneous mask
            mflat = masks_inst[i].data.ravel()
            mflat[flare_pixel] = 1.0
            masks_inst[i].data[:] = mflat.reshape(masks_inst[i].data.shape)
    
            # heat map: count number of detections per pixel
            idx = linear_indices_to_coords_unravel(flare_pixel, br_map.data.shape[1], br_map.data.shape[0])
     
            heat_map_data[idx[1,:], idx[0,:]] += 1.0
        
        # update flare_pixel_arr 
        if flare_count != 0 and n_ndp != 0:
            flare_pixel_arr = np.concatenate((flare_pixel_arr, ndp))
    
        # create quicklook plots
        flare_plot_quicklook(event, smap, flare_pixel_arr, ndp, wave, outimg, drange, i)
    
        #print(i)
    
    #final format of time and heat map
    time_map_data_new = np.full_like(time_map_data, np.nan, dtype=float)
    
    ttimes = np.asarray(fl_times)
    t = Time(ttimes)  # astropy times
    t_rel_s = (t - t[0]).to_value("s")
    t_min = t_rel_s / 60.0  # minutes since start time
    
    for i in range(num):
        pix_i = np.where(time_map_data == float(i))
        if pix_i[0].size > 0:
            time_map_data_new[pix_i] = t_min[i]
    
    time_map = Map(time_map_data_new, br_map.meta)
    heat_map = Map(heat_map_data, br_map.meta)
    
    nan_pix=np.asarray(np.where(heat_map.data == 0.0))
    heat_map.data[nan_pix[0,:], nan_pix[1,:]]=np.nan
    
    #plot heat and time map
    drange_hmi=(-500,500)
    flare_plot_heat_time_map(event, br_map, heat_map, time_map, outpath, drange_hmi)
    
    #store heat map, time map, instantaneous maps and cumulative map
    heat_file = outpath / f"{event}_heat_map_{str_wave}.fits"
    heat_map.save(heat_file, overwrite=True)
    
    time_file = outpath / f"{event}_time_map_{str_wave}.fits"
    time_map.save(time_file, overwrite=True)
    
    mask_cumulative_data = np.full_like(br_map.data, np.nan, dtype=float)
    cumul_pix=np.asarray(np.where(heat_map.data >= 1.0))
    mask_cumulative_data[cumul_pix[0,:], cumul_pix[1,:]]=1.0
    mask_cumulative= Map(mask_cumulative_data, br_map.meta)
    mask_cumulative.save(outpath / f"{event}_flare_cumulative_mask_{str_wave}.fits", overwrite=True)
    
    np.savez_compressed(
        outpath / f"{event}_masks_inst",
        data=[m.data for m in masks_inst],
        meta=[m.meta for m in masks_inst]
    )
    outfile_mask=outpath / f"{event}_flare_masks_inst.npz"
    np.savez(outfile_mask, map_arr=masks_inst, times=times)
    
    print(f"finished event {event}")

###############################################################################
# Example call

# fl_tm='2024-05-09T17:23:00.000'
# en_tm='2024-05-09T18:00:00.000'
# wave=1600
# event='20240509_1653'
# outpath = '/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/'
# data_path = outpath+'/maps/'
# path_sat_maps = outpath+'saturation_corrected_maps/'

# flare_detection(event, wave, fl_tm, en_tm, outpath, data_path, path_sat_maps)