#create subfield HMI vector field map for reference
# (c) K. Dissauer, University of Graz, 2025

import os
import sunpy.map as smap
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.colors as colors
from pathlib import Path
import glob

def flare_vector_field_map(event, hmi_dir, outdir, xcen, ycen, len_x, len_y):
    
    outdir_maps = outdir / "maps"
    os.makedirs(outdir_maps, exist_ok=True)
    
    #read-in SDO/HMI vector magnetogram and create map structure
    hmi_file=sorted(hmi_dir.glob("hmi.B_720s.*.Br.fits"))
    hmi_map = smap.Map(hmi_file)
    hmi_map = hmi_map.rotate(angle=hmi_map.meta["crota2"]*u.deg)
    
    #define subfield of view
    len_x = len_x * u.arcsec
    len_y = len_y * u.arcsec
    x0 = xcen * u.arcsec
    y0 = ycen * u.arcsec
    
    # create the coordinates (bottom-left + top_right) that define the subframe
    bottom_left = SkyCoord(x0 - len_x, y0 - len_y, frame=hmi_map.coordinate_frame)
    top_right   = SkyCoord(x0 + len_x, y0 + len_y, frame=hmi_map.coordinate_frame)
    
    shmi_map = hmi_map.submap(bottom_left, top_right=top_right)
    
    #plot HMI map for double-checking
    #fig=plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(projection=shmi_map)
    #norm= colors.Normalize(vmin=-500, vmax=500)
    #shmi_map.plot(axes=ax, norm=norm)
    
    #store HMI map in output directory
    shmi_map.save(outdir_maps / f"{event}_hmi_vector_magnetic_field.fits", overwrite=True)
    
####################################################################################    
# Example call of flare_vector_field_map:
    
# event='20240509_1653'
# outpath='/Users/karindissauer/Desktop/SOLER_test/out/'+event+'/ribbons/'
# path='/Users/karindissauer/Desktop/SOLER_test/'+event+'/'
# xcen=450
# ycen=-280
# len_x=300
# len_y=150

# flare_vector_field_map(event, path=path, outpath=outpath, xcen=xcen, ycen=ycen, len_x=len_x, len_y=len_y)