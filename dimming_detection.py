# detect coronal dimming pixels following Dissauer et al. 2018a (https://iopscience.iop.org/article/10.3847/1538-4357/aaadb5/pdf) 
# and create instantaneous and cumulative dimming masks, and dimming time map
# (c) K. Dissauer, University of Graz, 2025

import os
import numpy as np
import sunpy.map as smap
from astropy.coordinates import SkyCoord
from astropy import units as u
import sunpy
from astropy.io import fits
from astropy.time import Time
import copy
from filter_map import filter_map
from select_dimming_pixel import select_dimming_pixel
from new_dimming_pixel import new_dimming_pixel
from dimming_plot_quicklook import dimming_plot_quicklook
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#support functions
def coords_to_idx(coords, width, height):

    x = coords[0]
    y = coords[1]
    # rows = y, cols = x, shape = (height, width)
    return np.ravel_multi_index((y, x), dims=(height, width), order='C')

def linear_indices_to_coords_unravel(indices, width, height):
    
    i = np.asarray(indices)
    y, x = np.unravel_index(i, (height, width))
    return np.vstack((x, y))

######################################################################################
# main routine

def dimming_detection(event, outdir, wave, dim_thresh, drange, x0, y0, len_x, len_y, angle_thresh):
    
    os.makedirs(outdir, exist_ok=True)
    str_wave = str(wave)
    
    print("dim_thresh:", dim_thresh)
    
    outdir_maps = outdir / "maps" / str_wave
    
    outimg = outdir / "quicklook" / str_wave
    os.makedirs(outimg, exist_ok=True)
    
    #read info .fits file
    info_file = outdir_maps / f"{event}_info.fits"
    hdul = fits.open(info_file)
    table = hdul[1].data
    files  = np.array(table['FILE'])
    times  = np.array(table['DATE_OBS'])
    hdul.close()
    
    num=len(files)
    
    # allocate arrays
    # cumulative dim pixels
    dim_pixel_arr = []
    n_pixels_arr = np.full((1, num), -1)
    
    # load base map (stored as .fits file)
    base_file=outdir_maps / f"{event}_base_{str_wave}.fits"
    base_map = sunpy.map.Map(base_file)
    
    # define region of interest   
    # create the coordinates (bottom-left + top_right) that define the subframe
    bottom_left = SkyCoord(x0 * u.arcsec - len_x * u.arcsec, y0 * u.arcsec - len_y * u.arcsec, frame=base_map.coordinate_frame)
    top_right   = SkyCoord(x0 * u.arcsec + len_x * u.arcsec, y0 * u.arcsec + len_y * u.arcsec, frame=base_map.coordinate_frame)
    
    #define subfield base map
    sbase_map = base_map.submap(bottom_left, top_right=top_right)
    
    mask =  smap.Map(copy.deepcopy(sbase_map.data), copy.deepcopy(sbase_map.meta))
    mask.data[:] = np.nan
    
    mask_inst_data= np.full_like(sbase_map.data, np.nan, dtype=float)
    ttimes = np.asarray(times)
    ttimes_astro = Time(ttimes)
    # create instantaneous masks
    masks_inst = [smap.Map(copy.deepcopy(mask_inst_data), copy.deepcopy(sbase_map.meta)) for _ in range(num)]
    for i, t in enumerate(files):
        # store time information in the meta/header
        masks_inst[i].meta["time"] = ttimes_astro[i].isot
    
    # load angle map (store as .fits file)
    fangle=outdir_maps / f"{event}_angle.fits"
    angle_map = sunpy.map.Map(fangle)
    sangle_map = angle_map.submap(bottom_left, top_right=top_right)
    del angle_map
    
    # load SDO/HMI map and create subfield map (stored as .fits file)
    fhmi=outdir_maps / f"{event}_hmi.fits"
    hmi_map = sunpy.map.Map(fhmi)
    shmi_map = hmi_map.submap(bottom_left, top_right=top_right)
    del hmi_map
    
    # run dimming detection for each image of the time series
    start = 0
    end   = num
    
    for i in range(start,end):
    
        i_str = f"{i:03d}" 
        map_file = outdir_maps / f"{event}_maps_{i_str}_{str_wave}.fits"
        aia_map = smap.Map(map_file)
        
        aia_map_temp =  smap.Map(copy.deepcopy(aia_map.data), copy.deepcopy(base_map.meta))
        saia_map = aia_map_temp.submap(bottom_left, top_right=top_right)
        saia_map.meta["date-obs"]=aia_map.meta["date-obs"]
        
        del aia_map, aia_map_temp
        
        # create base-difference and logarithmic base-ratio images
        diff_data = np.float32(saia_map.data - sbase_map.data)
        sdiff_map = smap.Map(diff_data, copy.deepcopy(sbase_map.meta))
        
        # on-disk mask in FOV only
        base_coords = all_coordinates_from_map(sbase_map)
        on_disk_mask = coordinate_is_on_solar_disk(base_coords)
        
        ratio_data = np.float32(saia_map.data / sbase_map.data)
        ratio_data = np.log10(ratio_data)
        
        sratio_map = smap.Map(ratio_data, copy.deepcopy(sbase_map.meta))
        sratio_map.data[~on_disk_mask] = np.log10(1.0)
        
        #apply median filter for further processing and plotting
        map_show, sratio_log_filtered = filter_map(sratio_map, wave)
        
        #select dimming pixel
        dim_pixel, dim_count = select_dimming_pixel(sratio_log_filtered, sangle_map, ratio_thresh=dim_thresh, angle_thresh=angle_thresh)
        
        ny, nx=saia_map.data.shape
          
        # select new dimming pixels between this time step and all previous ones
        if dim_count > 0:
            dim_pixel_flt=coords_to_idx(dim_pixel, ny, nx)
            ndp = new_dimming_pixel(dim_pixel_flt, dim_pixel_arr)
            n_ndp = len(ndp)
            
            # set instantaneous mask
            inst_pix=np.asarray(dim_pixel)
            masks_inst[i].data[inst_pix[0,:], inst_pix[1,:]] = 1.0
        
        else:
            ndp = []
            n_ndp = 0
            ndp_coords =[]
        
        # update cumulative dimming mask
        if n_ndp > 0:
            for p in ndp:
                p_coords=linear_indices_to_coords_unravel(p,ny,nx)
                mask.data[p_coords[0],p_coords[1]] = float(i)
            ndp_coords=linear_indices_to_coords_unravel(ndp,ny,nx)
        
        # quicklook plotting of dimming detection
        dimming_plot_quicklook(saia_map, sratio_log_filtered, ndp_coords, n_ndp, dim_pixel_arr, wave, outimg, drange, i, event)
        
        # grow dim_pixel_arr
        if dim_count and n_ndp:
            dim_pixel_arr.extend(ndp)
        
        # Store results
        n_pixels_arr[:, i] = n_ndp
        
        print(i)
    
    #create time map    
    # build time_map: for each pixel, store dimming time in hours
    times = np.asarray(times)
    t = Time(times)  # astropy times
    t_rel_s = (t - t[0]).to_value("s")
    t_min = t_rel_s / 60.0  # hours since first time

    # we only care about existing time indices
    num = len(times)  # or len(n_pixels_arr), conceptually

    time_map_data = np.full_like(mask.data, np.nan, dtype=float)

    for i in range(num):
        pix_i = np.where(mask.data == float(i))
        if pix_i[0].size > 0:
            time_map_data[pix_i] = t_min[i]

    time_map = smap.Map(time_map_data, mask.meta)

    #plotting
    fig = plt.figure(figsize=(12, 6))

    # panel 1: cumulative dimming mask
    ax1 = fig.add_subplot(1, 2, 1, projection=shmi_map)
    norm = colors.Normalize(vmin=-100, vmax=100)
    im1=shmi_map.plot(axes=ax1, norm=norm)
    shmi_map.draw_limb()
    ax1.set_title(f"Dimming region — {event}")
    cbar = plt.colorbar(im1, ax=ax1, orientation="vertical", pad=0.05)

    ax1.imshow(
        mask.data,
        origin="lower",
        cmap="autumn_r",
         alpha=0.4,
        )

    # panel 2: dimming time map
    ax2 = fig.add_subplot(1, 2, 2, projection=time_map)
    # choose a colormap similar to IDL table 33
    norm = colors.Normalize(vmin=0.0, vmax=t_min[-1])
    im2 = time_map.plot(
        axes=ax2,
        cmap="jet",
        norm=norm
    )
    time_map.draw_limb()
    ax2.set_title("Dimming time [min since ref frame]")

    cbar = plt.colorbar(im2, ax=ax2, orientation="vertical", pad=0.05)
    cbar.set_label("Dimming time [min] since ref. frame")

    plt.tight_layout()

    # save PNG
    png_file = outdir / f"{event}_brightness_secondary_{wave}.png"
    fig.savefig(png_file, dpi=150)
    print("Saved cumulative dimming mask/time map plot:", png_file)
    plt.show()
    # save map output
    
    #subfield base map
    sbase_map.save(outdir / f"{event}_sbase_map_{str_wave}.fits", overwrite=True)
    #subfield HMI map
    shmi_map.save(outdir / f"{event}_shmi_map_{str_wave}.fits", overwrite=True)
    #subfield cumulative dimming mask
    mask.save(outdir / f"{event}_dimming_cumulative_mask_{str_wave}.fits", overwrite=True)
    
    #instantaneous dimming masks
    np.savez_compressed(
        outdir / f"{event}_dimming_masks_inst",
        data=[m.data for m in masks_inst],
        meta=[m.meta for m in masks_inst]
    )
    outfile_mask=outdir / f"{event}_dimming_masks_inst.npz"
    np.savez(outfile_mask, map_arr=masks_inst, times=times)
    
    print("finished event", event)

#########################################################################################
#Example call
# event = '20240509_1653'
# wave = 211
# dim_thresh = -0.19
# drange = [10, 2000]
# x0=450
# y0=-280
# len_x=1000
# len_y=1000
# angle_thresh=60.
# outpath='/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/results/'
# outdir = '/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/'

# dimming_detection(event, outdir, outpath, wave, dim_thresh, drange, x0, y0, len_x, len_y, angle_thresh)