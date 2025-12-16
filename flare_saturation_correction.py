#perform correction for saturated flare pixel following 
#Kazachenko et al. 2017 (https://iopscience.iop.org/article/10.3847/1538-4357/aa7ed6/pdf) 
# & Maybhate et al. 2008 (https://www.stsci.edu/instruments/wfpc2/Wfpc2_isr/wfpc2_isr0803.pdf)
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
from sunpy.map import Map
from skimage.morphology import dilation
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import warnings
import matplotlib.colors as colors
import os
from scipy.interpolate import interp1d
from astropy.time import Time
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

#define support functions
def convert_times_to_astropy(times):

    return Time(times)

def interp_to_equidistant(x, y, x_new):
    f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    return f(x_new)

def flare_saturation_correction(event, wave, data_path, outpath):
   
    str_wave = str(wave).strip()
    
    #define directory to store saturation corrected maps
    outpath_maps = outpath / "saturation_corrected_maps"
    os.makedirs(outpath_maps, exist_ok=True)
    
    # define constants
    sat_thresh = 4000
    mad_thresh = 30
    
    # structure element (3 x 30) for blooming correction
    structure = np.ones((3, 30), dtype=bool)
    
    # load base vector magnetic field map for reference
    vmf_file = data_path / f"{event}_hmi_vector_magnetic_field.fits"
    hmi_map = Map(str(vmf_file))
    hmi_data = hmi_map.data.copy()
    sz = hmi_data.shape
    
    # load SDO/AIA map array and times
    arr_file = outpath / f"{event}_maps_arr.npz"
    info = np.load(arr_file, allow_pickle=True)
    map_arr = info["map_arr"]
    times = info["times"]
    
    num = len(map_arr)
    
    # identify frames with saturation (≥250 pixels)
    sat_vec = np.zeros(num, dtype=int)
    
    for k in range(num):
        data = map_arr[k].data
        count = np.sum(data >= sat_thresh)
        if count > 250:
            sat_vec[k] = 1
    
    sat_indices = np.where(sat_vec == 1)[0]
    
    if sat_indices.size != 0:
        
        print('saturated pixels detected in time range...')
    
        st_ind = sat_indices[0]
        en_ind = sat_indices[-1]
        
        print("begin saturation:", times[st_ind])
        print("end saturation:", times[en_ind])
        
        # detect saturated pixels across full time series and create mask
        sat_pixel = []
        
        for i in range(num):
            data = np.asarray(map_arr[i].data, dtype=float)
        
            # find saturated or negative pixels
            y_idx, x_idx = np.where((data >= sat_thresh) | (data < 0))
        
            # store as individual pixel index pairs
            for y, x in zip(y_idx, x_idx):
                sat_pixel.append((y, x))
        
        # remove duplicates
        sat_pixel_unique = list(set(sat_pixel))
        
        mask = np.zeros(sz, dtype=bool)
        
        if len(sat_pixel_unique) > 0:
            sat_pixel_unique=np.asarray(sat_pixel_unique)
            ysat_idx, xsat_idx = np.transpose(sat_pixel_unique)
            mask[ysat_idx, xsat_idx] = True
        else:
            print("no saturated pixel found")

        # add neighbors using dilation (blooming correction)
        mask = dilation(mask, structure)

        # get time evolution of all selected saturated pixels
        sat_pixel = np.where(mask.ravel() == True)[0]
        sat_pixel_time = np.full((len(sat_pixel), num), np.nan, dtype=float)
        
        for p in range(num):
            dat = map_arr[p].data.ravel()
            sat_pixel_time[:, p] = dat[sat_pixel]
        
        # apply pixel-wise correction
        # ----------------------------------------------------------
        for j in range(len(sat_pixel)):
            
            dat = sat_pixel_time[j, :]
            #calculate mad and median over time evolution of each pixel
            mad = median_abs_deviation(dat, scale=1, nan_policy="omit")
            m = np.nanmedian(dat)
            #define saturated pixel either above threshold or above MAD
            bad_tmp = np.where((dat >= sat_thresh) | (dat >= m + mad_thresh * mad))[0]

            if len(bad_tmp) == 0:
                continue
        
            # define times when intensity needs to be corrected
            if len(bad_tmp) <= 10:
                # Expand around detected bad indices
                neib = 1
                first = bad_tmp[0] - neib
                last = bad_tmp[-1] + neib
                bad_idx = np.arange(first, last + 1)
                bad_idx = bad_idx[(bad_idx >= 0) & (bad_idx < num)]
            else:
                continue
        
            all_idx = np.arange(num)
        
            # remove negative pixel values too
            neg_idx = np.where(dat < 0)[0]
            bad_idx = np.unique(np.concatenate((bad_idx, neg_idx)))
        
            good = np.setdiff1d(all_idx, bad_idx)
        
            # interpolate between unsaturated times of each pixel
            if len(good) > 1:
                # Interpolation
                sat_pixel_time[j, :] = np.interp(all_idx, good, dat[good])
            else:
                # no correction possible
                pass

        # apply saturation correction to each image
        
        for i in range(num):
            smap = map_arr[i]
            

            data = np.asarray(smap.data.copy(), dtype=float)
            # get linear indices of saturated or negative pixels of this frame
            ybad_idx, xbad_idx = np.where((data >= sat_thresh) | (data < 0))
            
            # too few → no correction needed
            if len(xbad_idx) < 250:
                print(f"[{i}] no correction applied")
                corrected_map = Map(data, smap.meta)
            
            else:
                print(f"[{i}] correction applied")
            
                # build blooming mask for this frame
                mask = np.zeros(sz, dtype=bool)
                mask[ybad_idx, xbad_idx] = True
                mask = dilation(mask, structure)
                
                bad_idx=np.where(mask.ravel() == True)[0]
                
                # apply correction using precomputed time series
                flat = data.ravel()
                
                for pix in bad_idx:
                    loc = np.where(sat_pixel == pix)[0]
                    if loc.size !=0:
                        flat[pix] = sat_pixel_time[loc[0], i]
                
                data = flat.reshape(sz)
                corrected_map = Map(data, smap.meta)
                
                # plot original map (testing)
                cmap = plt.get_cmap("sdoaia{str_wave}")
                fig = plt.figure(figsize=(12, 8))
                ax0 = fig.add_subplot(1, 2, 1, projection=smap)
                norm = colors.Normalize(vmin=10, vmax=1000)
                smap.plot(axes=ax0, norm=norm, cmap=cmap)
                ax0.set_title(f"Original {smap.meta.get('date-obs','')} UT")
                
                # plot saturation corrected map (testing)
                ax1 = fig.add_subplot(1, 2, 2, projection=corrected_map)
                norm= colors.Normalize(vmin=10, vmax=1000)
                corrected_map.plot(axes=ax1, norm=norm, cmap=cmap)
                ax1.set_title(f"Saturation Corrected {corrected_map.meta.get('date-obs','')} UT")
                plt.tight_layout()
                plt.show()
            
            #store saturation corrected map in fits file
            i_str = f"{i:03d}"
            
            outname = outpath_maps / f"{event}_saturation_corrected_map_{i_str}_{str_wave}.fits"
            corrected_map.save(outname, overwrite=True)
            
            print("Saved:", outname)
    
        else:
            print('not enough saturated pixels for corrections found in image sequence... no correction applied...')


###############################################################################
# Example usage:
# flare_saturation_correction(
#     event="20240509_1653",
#     wave=1600,
#     data_path = "/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/maps/",
#     outpath="/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/"
# )
