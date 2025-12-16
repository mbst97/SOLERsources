# combine flare ribbon, coronal dimmings and KSO filament masks
# (c) K. Dissauer, University of Graz, 2025

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sunpy.map as smap
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from pathlib import Path

def combine_flare_dimming_filament(event, dimming_wave, flare_wave, outpath, hmi_file, kso_file, fl_st_tm, x0, y0, length):

    # read-in dimming mask (e.g., instantaneous,  cumulative dimming mask, time map, etcs.)
    dimming_file=outpath / "dimming" / f"{event}_dimming_cumulative_mask_{str(dimming_wave)}.fits"
    dimming_map=smap.Map(dimming_file)
    
    # read-in flare ribbon mask (e.g., instantaneous, cumulative flare ribbon mask, time map, heat map)
    flare_file=outpath / "flare" / f"{event}_time_map_{str(flare_wave)}.fits"
    flare_map=smap.Map(flare_file)
    
    # read-in HMI data
    hmi_map = smap.Map(hmi_file)
    hmi_map = hmi_map.rotate(angle=hmi_map.meta["crota2"]*u.deg)
    
    # co-align ribbon and dimming masks with SDO/HMI map
    new_dimming_data, new_dimming_wcs = reproject_interp(dimming_map, hmi_map.wcs)
    dimming_map_reproj = smap.Map(new_dimming_data, hmi_map.meta)
    
    new_flare_data, new_wcs = reproject_interp(flare_map, hmi_map.wcs)
    flare_map_reproj = smap.Map(new_flare_data, hmi_map.meta)
    
    # read-in KSO filament mask
    kso_map = smap.Map(kso_file)
    kso_map.meta['CROTA2'] = kso_map.meta['CROTA1']
    
    # co-align filament mask with SDO/HMI map
    new_kso_data, new_kso_wcs = reproject_interp(kso_map, hmi_map.wcs)
    kso_map_reproj = smap.Map(new_kso_data, hmi_map.meta)
    
    # plot combined masks for full-disk map
    fig = plt.figure(figsize=(8, 8))
    
    ax= fig.add_subplot(projection=hmi_map)
    norm = colors.Normalize(vmin=-500, vmax=500)
    hmi_map.plot(axes=ax, norm=norm)
    # ax.set_title(fl_st_tm)
    
    # overplot dimming mask
    ax.imshow(
          dimming_map_reproj.data,
          origin="lower",
          cmap="autumn_r",
           alpha=0.6,
          )
    
    # overplot flare ribbon mask
    ax.imshow(
          flare_map_reproj.data,
          origin="lower",
          cmap="jet",
           alpha=0.8,
          )
    
    # overplot filament mask as contour
    contours = kso_map_reproj.find_contours(0.5 * u.G)
    for contour in contours:
        ax.plot_coord(contour, color='cyan', linewidth=1.2)
        
    # save plot as PNG image
    png_file = outpath / f"{event}_combined_ribbon_dimming_filament_fulldisk.png"
    fig.savefig(png_file, dpi=150)
    plt.show()
    print("Saved combined dimming ribbon filament plot:", png_file)
    
    #######################################################################################
    # plot subfield to make flare ribbon evolution better visible
    # define submap
    
    top_right = SkyCoord(x0 * u.arcsec + length * u.arcsec, y0 * u.arcsec + length * u.arcsec, frame=hmi_map.coordinate_frame)
    bottom_left = SkyCoord(x0 * u.arcsec - length * u.arcsec, y0 * u.arcsec - length * u.arcsec, frame=hmi_map.coordinate_frame)
    hmi_submap = hmi_map.submap(bottom_left, top_right=top_right)
    
    # co-align ribbon, dimming, filament mask with new subfield
    new_dimming_data, new_dimming_wcs = reproject_interp(dimming_map, hmi_submap.wcs)
    dimming_submap_reproj = smap.Map(new_dimming_data, hmi_submap.meta)
    
    new_flare_data, new_wcs = reproject_interp(flare_map, hmi_submap.wcs)
    flare_submap_reproj = smap.Map(new_flare_data, hmi_map.meta)
    
    new_kso_data, new_kso_wcs = reproject_interp(kso_map, hmi_submap.wcs)
    kso_submap_reproj = smap.Map(new_kso_data, hmi_submap.meta)
    
    # plot subfield
    fig = plt.figure(figsize=(8, 8))
    
    ax= fig.add_subplot(projection=hmi_submap)
    norm = colors.Normalize(vmin=-500, vmax=500)
    hmi_submap.plot(axes=ax, norm=norm)
    ax.set_title(fl_st_tm)
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    
    # overplot dimming mask
    ax.imshow(
          dimming_submap_reproj.data,
          origin="lower",
          cmap="autumn_r",
           alpha=0.6,
          )
    
    #overplot flare ribbon mask
    ax.imshow(
          flare_submap_reproj.data,
          origin="lower",
          cmap="jet",
           alpha=0.8,
          )
    
    # overplot filament mask as contours
    contours = kso_submap_reproj.find_contours(0.5 *u.G)
    for contour in contours:
        ax.plot_coord(contour, color='cyan', linewidth=1.2)
    
    # save subfield plot as PNG image
    png_file = outpath / f"{event}_combined_ribbon_dimming_filament_subfield.png"
    fig.savefig(png_file, dpi=150)
    plt.show()
    print("Saved combined dimming ribbon filament subfield plot:", png_file)
    return