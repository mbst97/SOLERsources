#plotting routine for quicklook images of dimming detection
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.colors as colors
import matplotlib.cm as cm
from astropy.coordinates import SkyCoord
from pathlib import Path

###############################################################################
#support functions
def linear_indices_to_coords_unravel(indices, width, height):
    i = np.asarray(indices)
    y, x = np.unravel_index(i, (height, width))
    return np.vstack((x, y))
###############################################################################

def dimming_plot_quicklook(map_in, ratio_map, ndp, n_ndp, dim_pixel_arr,
                          wave, outimg, drange, i, event):
    """
    Parameters
    ----------
    map_in : sunpy.map.Map
        The intensity map.
    ratio_map : sunpy.map.Map
        The log base-ratio map.
    ndp : array-like
        Newly dimmed pixels (flat indices).
    n_ndp : int
        Number of newly dimmed pixels.
    dim_pixel_arr : array-like
        All accumulated dimming pixels (flat indices).
    wave : int
        AIA wavelength for coloring (e.g., 211).
    outimg : str
        Output directory.
    drange : (float, float)
        Min/max for map display.
    i : int
        Frame index.
    event : str
        Event ID.
    """
    
    str_wave=str(wave)

    # figure setup
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)

    ax0 = fig.add_subplot(1, 3, 1, projection=map_in)
    ax1 = fig.add_subplot(1, 3, 2, projection=ratio_map)
    ax2 = fig.add_subplot(1, 3, 3, projection=ratio_map)


    # plot 1 — plot original SDO/AIA image
    norm = colors.LogNorm(vmin=drange[0], vmax=drange[1])
    map_in.plot(axes=ax0, norm=norm)
    ax0.set_title(f"SDO/AIA {wave} Å {map_in.meta.get('date-obs','')} UT")
    ax0.set_xlabel("X (arcsec)")
    ax0.set_ylabel("Y (arcsec)")
    ax0.coords.grid(False)

    # plot 2 — plot logarithmic base ratio map + cumulative dimming pixels
    
    norm = colors.Normalize(vmin=-0.7, vmax=0)
    ratio_map.plot(axes=ax1, norm=norm, origin='lower',
                 cmap='gray')
    ax1.set_title('Cumulative Dimming Pixel Mask')
    ax1.set_xlabel("X (arcsec)")
    ax1.set_ylabel("Y (arcsec)")
    ax1.coords.grid(False)

    # overplot cumulative dimming pixels
    if dim_pixel_arr is not None and len(dim_pixel_arr) > 0:
        
        ny, nx =ratio_map.data.shape
        dim_pixel_arr_coords=linear_indices_to_coords_unravel(dim_pixel_arr, ny, nx)
        x_pix_arr = dim_pixel_arr_coords[1, :] * u.pixel
        y_pix_arr = dim_pixel_arr_coords[0, :] * u.pixel

        # convert pixel → world coordinates
        hp_arr = map_in.pixel_to_world(x_pix_arr, y_pix_arr)

        # extract world coordinates
        xnew_arr = hp_arr.Tx.to(u.arcsec)
        ynew_arr = hp_arr.Ty.to(u.arcsec)
        ax1.plot_coord(SkyCoord(xnew_arr, ynew_arr, frame=ratio_map.coordinate_frame), 
                   "o", c='purple', markersize=1, alpha=0.6)

    # plot 3 — plot logarithmic base ratio map + new dimming pixels
    norm = colors.Normalize(vmin=-0.7, vmax=0)
    ratio_map.plot(axes=ax2, norm=norm, origin='lower',
                 cmap='gray')
    ax2.set_title('Newly Dimmed Pixels')
    ax2.set_xlabel("X (arcsec)")
    ax2.set_ylabel("Y (arcsec)")
    ax2.coords.grid(False)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='gray'), ax=ax2, label="log ratio")

    #overplot new dimming pixels
    if (n_ndp is not None) and (n_ndp > 0):
        x_pix = ndp[1, :] * u.pixel
        y_pix = ndp[0, :] * u.pixel

        # convert pixel → world coordinates
        hp = map_in.pixel_to_world(x_pix, y_pix)

        # extract world coordinates
        xnew = hp.Tx.to(u.arcsec)
        ynew = hp.Ty.to(u.arcsec)

        ax2.plot_coord(SkyCoord(xnew, ynew, frame=ratio_map.coordinate_frame), 
                   "o", c='red', markersize=1, alpha=0.6)

    # save plot as PNG image
    i_str = f"{i:03d}"
    outfile = outimg / f"{event}_dimming_quicklook_{i_str}_{str_wave}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print("Saved:", outfile)