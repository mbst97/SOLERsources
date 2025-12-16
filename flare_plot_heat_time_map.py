# plotting routine to visulaize flare heat and time map
# (c) K. Dissauer, University of Graz, 2025

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from reproject import reproject_interp
import sunpy.map as smap
from pathlib import Path

def flare_plot_heat_time_map(event, map_in, heat_map, time_map, outimg, drange):
    """
    Parameters
    ----------
    event : str
        event ID
    map_in : sunpy.map.Map
        The intensity map to display.
    flare_pixel : array-like
        Flat indices of flare pixels.
    ndp : array-like
        Flat indices of "newly detected" pixels.
    wave : int
        AIA wavelength (e.g., 1600, 1700).
    outimg : str
        Output directory / prefix for the image.
    drange : (float, float)
        Min, max display range for map_in.
    i : int
        Frame index (for filename).
    """
    
    # figure setup
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)

    # all two panels use the same map projection
    ax0 = fig.add_subplot(1, 2, 1, projection=map_in)
    ax1 = fig.add_subplot(1, 2, 2, projection=map_in)
    
    vmin, vmax = drange
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    #panel 1 - heat map on top of SDO/HMI magnetogram
    map_in.plot(axes=ax0, norm=norm, cmap='gray', origin='lower')
    ax0.set_title(event+' Flare Ribbon Heat Map ')
    ax0.set_xlabel("X (arcsec)")
    ax0.set_ylabel("Y (arcsec)")
    ax0.coords.grid(False)
    
    new_heat_data, new_wcs = reproject_interp(heat_map, map_in.wcs)
    heat_map_reproj = smap.Map(new_heat_data, map_in.meta)
    
    heat_im = ax0.imshow(
          heat_map_reproj.data,
          origin="lower",
          cmap="plasma_r",
           alpha=0.9,
          )
    
    fig.colorbar(heat_im, ax=ax0, orientation="horizontal", location="top",
    pad=0.05, label="")
    
    #panel 2 - time map on top of SDO/HMI magnetogram
    map_in.plot(axes=ax1, norm=norm, cmap='gray', origin='lower')
    ax1.set_title('Flare Ribbon Time Map ')
    ax1.set_xlabel("X (arcsec)")
    ax1.set_ylabel("Y (arcsec)")
    ax1.coords.grid(False)
    
    new_time_data, new_wcs = reproject_interp(time_map, map_in.wcs)
    time_map_reproj = smap.Map(new_time_data, map_in.meta)
    
    time_im = ax1.imshow(
          time_map_reproj.data,
          origin="lower",
          cmap="turbo",
           alpha=0.9,
          )
    
    fig.colorbar(time_im, ax=ax1, orientation="horizontal", location="top",
    pad=0.05, label="Time since flare start [min]")

    #save plot as PNG image
    outfile = outimg / f"{event}_heat_time_map.png"
    fig.savefig(outfile, format="png", dpi=150)
    plt.show()

    print("Saved:", outfile)