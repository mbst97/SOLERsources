# main program to run dimming, flare ribbon and filament masks
# example data available at https://drive.google.com/drive/folders/10mBDcqnR-L61Pv91iwIcbS01wLKeC1i_?usp=share_link
# (c) K. Dissauer, University of Graz, 2025

import os
from dimming_detection import dimming_detection
from dimming_processing_aia import dimming_processing_aia
from flare_vector_field_map import flare_vector_field_map
from flare_processing_aia import flare_processing_aia
from flare_map_array import flare_map_array
from flare_saturation_correction import flare_saturation_correction
from flare_detection import flare_detection
from combine_flare_dimming_filament import combine_flare_dimming_filament
from pathlib import Path
import sys
import matplotlib.pyplot as plt

#event specifics
event='20240509_1653'
aia_dir = Path.home() / "Desktop" / "SOLER_example_data" / event

x0=450
y0=-280
dimension=2048

############################################################################################
#perform dimming detection and analysis
outdir= Path.home() / "Desktop" / "SOLER" / "out" / event
os.makedirs(outdir, exist_ok=True)
outpath_dimming = outdir / "dimming"
os.makedirs(outpath_dimming, exist_ok=True)

n_base=10
len_x_dimming=1000
len_y_dimming=1000
dimming_wave=211
dimming_cadence=1
dim_thresh = -0.19
angle_thresh=60.
drange = [10, 2000]

#processing of SDO/AIA data for dimming detection
dimming_processing_aia(event, outpath_dimming, aia_dir, dimension, n_base, dimming_wave, dimming_cadence)
#dimming detection
dimming_detection(event, outpath_dimming, dimming_wave, dim_thresh, drange, x0, y0, len_x_dimming, len_y_dimming, angle_thresh)

#sys.exit()
############################################################################################
#perform flare ribbon detection and analysis
outpath_flare = outdir / "flare"
os.makedirs(outpath_flare, exist_ok=True)

len_x_flare=300
len_y_flare=150
flare_wave=1600
flare_cadence=1

# flare start and end time (e.g. get from GOES flare catalog) 
fl_st_tm='2024-05-09T17:23:00.000'
fl_en_tm='2024-05-09T18:00:00.000'

#create SDO/HMI vector field map for reference
flare_vector_field_map(event, aia_dir, outpath_flare, x0, y0, len_x_flare, len_y_flare)
#processing of SDO/AIA data for flare ribbon detection
flare_processing_aia(event, flare_wave, flare_cadence, aia_dir, outpath_flare / "maps", dimension=dimension)
#create image array for time evolution analysis
flare_map_array(event, flare_wave, outpath_flare / "maps", outpath_flare)
#correct for image saturation
flare_saturation_correction(event, flare_wave, outpath_flare / "maps", outpath_flare)
#flare ribbon detection
flare_detection(event, flare_wave, fl_st_tm, fl_en_tm, outpath_flare, outpath_flare / "maps", outpath_flare / "saturation_corrected_maps")

#############################################################################################
#combine flare ribbon, coronal dimming and filment masks
#retrieve KSO filament data from https://cesar.kso.ac.at/filament/filaments.php 
kso_file   = aia_dir / "kanz_halph_ma_20240509_161055.fts.gz"
hmi_file  = aia_dir / "hmi.M_720s.20240509_164800_TAI.3.magnetogram.fits"
length=500

combine_flare_dimming_filament(event, dimming_wave, flare_wave, outdir, hmi_file, kso_file, fl_st_tm, x0, y0, length)

#plt.close()