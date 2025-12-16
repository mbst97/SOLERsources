#create SDO/AIA EUV maps for coronal dimming detection
# (c) K. Dissauer, University of Graz, 2025

import os
import numpy as np
from astropy.io import fits
import sunpy.map as smap
from astropy import units as u
import glob
import copy
from check_quality import check_quality
from check_exptime import check_exptime
from mean_base_map import mean_base_map
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from aiapy.calibrate import register, update_pointing
from aiapy.calibrate.util import get_pointing_table
from pathlib import Path

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

def dimming_processing_aia(event, outdir, aia_dir, dimension, n_base, wave, cadence):

    str_wave = str(wave)
    
    # get AIA data from somewhere like VSO, JSOC or through sunpy and fido
    # load in the fits files (SDO/AIA and SDO/HMI (either use line-of-sight or vector magnetogram))
    pre_aia_files=sorted(aia_dir.glob(f"aia*{str_wave}*image_lev1*.fits"))
    
    hmi_file=sorted(aia_dir.glob("hmi.M_720s*.fits"))
    
    outdir_maps = outdir / "maps" / str_wave
    os.makedirs(outdir_maps, exist_ok=True)
    
    # check the qulaity and exposure time of the AIA files (select only long exposures for dimming)
    in_files, cm_quality = check_quality(pre_aia_files)
    files, times, cm_exptime = check_exptime(in_files)
    
    comment_cadence='cadence:'+str(cadence)
    files=files[0::cadence]
    times=times[0::cadence]
    
    comment=''
    comment=cm_quality+','+cm_exptime+','+comment_cadence
    
    num=len(files)	
    
    # create info file with final file list, time array and comment on quality, exposure time and cadence
    info_file = outdir_maps / f"{event}_info.fits"
    
    file_col   = fits.Column(name='FILE',    format='200A', array=np.array([str(f) for f in files], dtype='U200'))
    time_col   = fits.Column(name='DATE_OBS', format='50A', array=np.array(times, dtype='U50'))
    comment_col = fits.Column(name='COMMENT', format='200A', array=np.array([comment], dtype='U200'))
    
    hdu = fits.BinTableHDU.from_columns([file_col, time_col, comment_col])
    hdu.writeto(info_file, overwrite=True)
    
    # create base image (mean over n_base images of the time series)
    base = mean_base_map(files, dimension=dimension, n_base=n_base)
    base.save(outdir_maps / f"{event}_base_{str_wave}.fits", overwrite=True)
    
    ny, nx = base.data.shape
    
    # create angle mask (and if appicable also correction mask) as copy of base
    #corr_mask =  smap.Map(np.full((ny, nx), np.nan), copy.deepcopy(base.meta))
    angle_mask = smap.Map(np.full((ny, nx), np.nan), copy.deepcopy(base.meta))
    
    # on-disk mask
    base_coords= all_coordinates_from_map(base)
    on_disk_mask= coordinate_is_on_solar_disk(base_coords)
    
    # map coordinates
    # build pixel index grid (note the order: y, x)
    yy, xx = np.mgrid[0:ny, 0:nx] * u.pix
    
    # convert pixel coordinates to world coordinates
    coords = base.pixel_to_world(xx, yy)  # helioprojective coordinates
    
    # helioprojective X and Y in arcsec
    x = coords.Tx.to(u.arcsec).value
    y = coords.Ty.to(u.arcsec).value
    
    rsun = base.rsun_obs.to(u.arcsec).value
    
    # calculate radial distance
    r = np.sqrt(x**2 + y**2)
    alpha = np.arcsin(r / rsun)  # in radians
    
    # get angle in degrees and apply to mask
    angle_deg = np.rad2deg(alpha)
    angle_mask.data[on_disk_mask] = angle_deg[on_disk_mask]
    
    #if line-of-sight magnetogram is used correction for seeing needs to be applied
    #correction = 1.0 / np.cos(alpha)  # LOS correction
    #corr_mask.data[on_disk_mask] = correction[on_disk_mask]
    
    # write angle mask (and if applicable correction mask) to fits file
    angle_file = outdir_maps / f"{event}_angle.fits"
    #corr_file = outdir_maps+event+'_correction.fits'
    
    angle_mask.save(angle_file, overwrite=True)
    #corr_mask.save(corr_file, overwrite=True)
    
    #reprjoect SDO/HMI map to SDO/AIA base image
    hmi = smap.Map(hmi_file)
    hmi = hmi.reproject_to(base.wcs)
    hmi=rebin_map(hmi, dimension, dimension)
    # if line-of-sight magnetogram is used HMI data needs to be corrected
    #hmi.data[:] = hmi.data[:]*corr_mask.data[:]
    
    hmi.save(outdir_maps / f"{event}_hmi.fits", overwrite=True)
    
    # prepare output maps for all SDO/AIA files and save as .fits files:
    #   - AIA-prepped
    #   - rebinned to 'dimension'
    #   - differentially rotated to 'base'

    start = 0
    end   = len(files)
    
    for i in range(start,end):
        f = files[i]
        print(f"Processing file {i+1}/{num}: {f}")
        m = read_aia_map(f)
        m = rebin_map(m, dimension, dimension)
        t_out=m.meta["t_obs"]
        m = drot_map(m, ref_map=base)  # differential rotate to base
    
        #update meta info of map with correct observation time
        m.meta["date-obs"]=t_out
    
        i_str = f"{i:03d}"
        map_file = outdir_maps / f"{event}_maps_{i_str}_{str_wave}.fits"
    
        m.save(map_file, overwrite=True)
        
    print("Done.")

####################################################################
#Example call:
    
# event='20240509_1653'
# outdir='/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/'
# aia_dir='/Users/karindissauer/Desktop/SOLER_test/'+event+'/'

# dimension=2048
# n_base=10
# wave=211
# cadence=4

# dimming_processing_aia(event, outdir, aia_dir, dimension, n_base, wave, cadence)