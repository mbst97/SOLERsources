# plotting routine for quicklook images to illustrate flare pixel detection
# (c) K. Dissauer, University of Graz, 2025

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from pathlib import Path

##############################################################################
# support functions

def linear_indices_to_coords_unravel(indices, width, height):
    i = np.asarray(indices)
    y, x = np.unravel_index(i, (height, width))
    return np.vstack((x, y))
##############################################################################

def flare_plot_quicklook(event, map_in, flare_pixel, ndp, wave, outimg, drange, i):
    """
    Parameters
    ----------
    map_in : sunpy.map.Map
        The intensity map to display.
    flare_pixel : array-like
        Flat indices of flare pixels.
    ndp : array-like
        Flat indices of "newly detected" pixels.
    wave : int
        AIA ultraviolet wavelength (e.g., 1600, 1700).
    outimg : str
        Output directory / prefix for the image.
    drange : (float, float)
        Min, max display range for map_in.
    i : int
        Frame index (for filename).
    """

    n_flare = 0 if flare_pixel is None else len(flare_pixel)
    n_ndp = 0 if ndp is None else len(ndp)

    #str_wave = str(wave)

    # figure setup
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)

    # all three panels use the same map projection
    ax0 = fig.add_subplot(1, 3, 1, projection=map_in)
    ax1 = fig.add_subplot(1, 3, 2, projection=map_in)
    ax2 = fig.add_subplot(1, 3, 3, projection=map_in)

    try:
        cmap = plt.get_cmap(f"sdoaia{wave}")
    except Exception:
        # Fallback if the AIA colortable is unavailable
        cmap = cm.get_cmap("gray")

    # panel 1 - original SDO/AIA ultraviolet map
    vmin, vmax = drange
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    map_in.plot(axes=ax0, norm=norm, cmap=cmap, origin='lower')
    title_id = map_in.meta.get('id', f"SDO/AIA {wave} Å")
    title_time = map_in.meta.get('date-obs', getattr(map_in, "date", ""))
    ax0.set_title(f"{title_id}  {title_time} UT")
    ax0.set_xlabel("X (arcsec)")
    ax0.set_ylabel("Y (arcsec)")
    ax0.coords.grid(False)

    # panel 2 – same map, overplot cumulative flare pixels-
    map_in.plot(axes=ax1, norm=norm, cmap=cmap, origin='lower')
    ax1.set_title("Cumulative Flare Pixel Mask")
    ax1.set_xlabel("X (arcsec)")
    ax1.set_ylabel("Y (arcsec)")
    ax1.coords.grid(False)

    ny, nx = map_in.data.shape

    # plot flare pixels
    if n_flare > 0:
        flare_coords = linear_indices_to_coords_unravel(flare_pixel, nx, ny)
        x_pix_flare = flare_coords[0, :] * u.pixel
        y_pix_flare = flare_coords[1, :] * u.pixel

        # convert pixel → world coordinates
        flare_world = map_in.pixel_to_world(x_pix_flare, y_pix_flare)

        # plot in world coordinates (arcsec)
        ax1.plot_coord(
            flare_world,
            "o",
            c="deeppink",
            markersize=0.5,
            alpha=0.7,
        )

    # panel 3 – same map, overplot ndp pixels
    map_in.plot(axes=ax2, norm=norm, cmap=cmap, origin='lower')
    ax2.set_title("Newly Flaring Pixel")
    ax2.set_xlabel("X (arcsec)")
    ax2.set_ylabel("Y (arcsec)")
    ax2.coords.grid(False)

    if n_ndp > 0:
        ndp_coords = linear_indices_to_coords_unravel(ndp, nx, ny)
        x_pix_ndp = ndp_coords[0, :] * u.pixel
        y_pix_ndp = ndp_coords[1, :] * u.pixel

        ndp_world = map_in.pixel_to_world(x_pix_ndp, y_pix_ndp)

        ax2.plot_coord(
            ndp_world,
            "o",
            c="blue",
            markersize=0.5,
            alpha=0.7,
        )
        
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, label="[DN/s]")

    # save plot as PNG image
    i_str = f"{i:03d}"
    outfile = outimg / f"{event}_quicklook_{i_str}_{wave}.png"
    fig.savefig(outfile, format="png", dpi=150)
    plt.close(fig)

    print("Saved:", outfile)
