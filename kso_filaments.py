# Read-in and plot KSO filament masks together with SDO/AIA image
# (c) K. Dissauer, University of Graz, 2025

import sunpy.map
import matplotlib.pyplot as plt
#from sunpy.net import Fido, attrs as a
from astropy import units as u
from reproject import reproject_interp

from aiapy.calibrate import register, update_pointing
from aiapy.calibrate.util import get_pointing_table
import matplotlib.colors as colors

outpath='/Users/karindissauer/Downloads/'

# path to KSO FITS file (download from https://cesar.kso.ac.at/filament/filaments.php)
path='/Users/karindissauer/Documents/Projects/python/SOLERsources/example_data/'
fname = path+"/kanz_halph_ma_20251116_111735.fts.gz"

# load KSO file into a SunPy Map and correct angle keyword
kso_map = sunpy.map.Map(fname)
kso_map.meta['CROTA2'] = kso_map.meta['CROTA1']

#load co-temporal SDO/AIA image through fido
#res_aia= Fido.search(a.Time("2025-11-16T11:17:00", "2025-11-16T11:18:00"), 
#                      a.Instrument.aia, a.Wavelength(211 * u.AA))

#files_aia = Fido.fetch(res_aia[0,0])
#aia_map = sunpy.map.Map(files_aia[0])

#load co-temporal SDO/AIA image from path
fname_aia=path+"aia.lev1_euv_12s.2025-11-16T111659Z.211.image_lev1.fits"
aia_map = sunpy.map.Map(fname_aia)

# calibrate SDO/AIA data using recent pointing
pointing_table = get_pointing_table("JSOC", time_range=(aia_map.date - 12 * u.h, aia_map.date + 12 * u.h))
aia_map_updated_pointing = update_pointing(aia_map, pointing_table=pointing_table)
aia_map_reg = register(aia_map_updated_pointing)

# correct negative pixel values
aia_map_reg.data[aia_map_reg.data < 0.0] = 1.0

# co-align KSO filament mask to SDO/AIA image
new_data, new_wcs = reproject_interp(kso_map, aia_map_reg.wcs)
kso_map_reproj = sunpy.map.Map(new_data, aia_map_reg.meta)

#plot SDO/AIA image together with KSO filaments as contours
contours = kso_map_reproj.find_contours(0.5 * u.DN)

fig=plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=aia_map_reg)
norm= colors.LogNorm(vmin=10, vmax=12000)
aia_map_reg.plot(axes=ax, norm=norm)
ax.set_title(f"SDO/AIA 211 Å {kso_map_reproj.meta["date-obs"]}")
for contour in contours:
    ax.plot_coord(contour, color='cyan', linewidth=1.2)

aia_map_reg.draw_limb()
aia_map_reg.draw_grid()
fig.cbar=plt.colorbar()
fig.cbar.set_label("[DN]")

# save plot as PNG image
png_file = f"{outpath}kso_aia_filaments_{kso_map_reproj.meta["date-obs"]}.png"
fig.savefig(png_file, dpi=150)
plt.close(fig)
print("Saved kso/aia plot:", png_file)