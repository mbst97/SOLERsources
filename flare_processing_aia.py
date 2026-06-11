#create SDO/AIA UV maps for flare ribbon detection
# (c) K. Dissauer, University of Graz, 2025

import os
import numpy as np
from astropy.io import fits
import sunpy.map as smap
from astropy import units as u
import glob
from check_quality import check_quality
#from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
#from sunpy.physics.differential_rotation import differential_rotate
#from sunpy.coordinates import propagate_with_solar_surface
from aiapy.calibrate import register, update_pointing
from aiapy.calibrate.utils import get_pointing_table
from pathlib import Path

######################################################################################
# read an SDO/AIA fits file and perform preprocessing equivalent to IDL aia_prep
def read_aia_map(filename):

    m = smap.Map(filename)

    # aia_pre equivalent:
    # update pointing and register (resample to common plate scale & align centers)
    pointing_table = get_pointing_table("LMSAL", time_range=(m.date - 12 * u.h,m.date + 12 * u.h))
    m_updated_pointing = update_pointing(m, pointing_table=pointing_table)
    m = register(m_updated_pointing)

    # normalize by exposure time
    if hasattr(m, "exposure_time"):
        m = smap.Map((m.data / m.exposure_time.to_value("s")), m.meta)
    else:
        print("WARNING: No EXPOSURE_TIME keyword found — normalization skipped.")

    # Convert to float since normalization by exposure time requires that
    m = smap.Map(m.data.astype(np.float32), m.meta)

    return m

# define support functions
######################################################################################
# resample a SunPy map to (ny, nx) pixels.
def rebin_map(m, nx, ny):

    new_dimensions = [ny, nx] * u.pixel
    return m.resample(new_dimensions)

######################################################################################
# differentially rotate map 'm' to the time of 'ref_map'.
def drot_map(m, ref_map):

    #t_ref = ref_map.date
    #return differential_rotate(m, time=t_ref)
    out_wcs=ref_map.wcs
    mtemp=m.reproject_to(out_wcs)
    return mtemp

######################################################################################
# main routine
def flare_processing_aia(event, wave, cadence, aia_dir, outdir, dimension):

    str_wave=str(wave)
    os.makedirs(outdir, exist_ok=True)

    # get SDO/AIA data from somewhere like VSO, JSOC or through sunpy and fido
    pre_aia_files=sorted(aia_dir.glob(f"aia*{str_wave}*image_lev1*.fits"))
    
    hmi_file=outdir / f"{event}_hmi_vector_magnetic_field.fits"
    
    # check for data quality
    files, cm_quality = check_quality(pre_aia_files)
    
    files=files[0::cadence]
    times = []
    
    for f in files:
        with fits.open(f, memmap=True) as hdul:
             hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
             times.append(hdr.get("DATE-OBS", ""))
    
    times = np.array(times)
    
    comment_cadence='cadence: '+ str(cadence)
    comment=''
    comment=cm_quality+'_'+comment_cadence
    
    num=len(files)	
    
    #create info file of files that will be used
    info_file = outdir / f"{event}_info.fits"
    
    file_col   = fits.Column(name='FILE',    format='200A', array=np.array([str(f) for f in files], dtype='U200'))
    time_col   = fits.Column(name='DATE_OBS', format='50A', array=np.array(times, dtype='U50'))
    comment_col = fits.Column(name='COMMENT', format='200A', array=np.array([comment], dtype='U200'))
    
    # create fits binary table and write out
    hdu = fits.BinTableHDU.from_columns([file_col, time_col, comment_col])
    hdu.writeto(info_file, overwrite=True)
    
    #read-in SDO/HMI map to use as reference map
    hmi_file = outdir / f"{event}_hmi_vector_magnetic_field.fits"
    hmi_map = smap.Map(hmi_file)
    
    #use first map in SDO/AIA fits file series as base map
    base = read_aia_map(files[0])
    base = rebin_map(base, dimension, dimension)
    base = base.reproject_to(hmi_map.wcs)
    
    base.save(outdir / f"{event}_base_{str_wave}.fits", overwrite=True)
    
    # prepare output maps for all SDO/AIA files:
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
    
        m.meta["date-obs"]=t_out
    
        i_str = f"{i:03d}"
        map_file = outdir / f"{event}_maps_{i_str}_{str_wave}.fits"
    
        m.save(map_file, overwrite=True)
        
    print("Done.")

###############################################################################
#Example call:
# dimension=2048
# event='20240509_1653'
# wave=1600
# cadence=3
# outpath='/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/maps/'
# path='/Users/karindissauer/Desktop/SOLER_test/'+event+'/'

# flare_processing_aia(event=event, wave=wave, cadence=cadence, aia_dir=path, outpath=outpath, dimension=dimension)