# SOLERsources
This code suite includes Python code to detect and analyze coronal dimmings (based on Dissauer et al. 2018) and flare ribbons (based on Kazachenko et al. 2017) as well as combining both observations with filament detections (based on Pötzi et al. 2015) to create combined masks of all three phenomena for further scientific analysis.
Example Data can be found here: https://drive.google.com/drive/folders/10mBDcqnR-L61Pv91iwIcbS01wLKeC1i_?usp=share_link 

## Setup
SOLERwave requires a Python installation of 3.10 or newer. 

Further, required packages are

* jupyter 1.1.1 
* numpy 2.2.6
* astropy 7.2.0
* sunpy 7.1.2 
* aiapy 0.12.0

Alternatively the Jupyter notebook "Install_packages.ipynb" can be run after the installation of Python and Jupyter. 
It will check the availability of the packages required and download any missing. Note, it will not check the version
of packages already installed. 

## Demo Program

To demonstrate the functionality of the tool a Demo Jupyter Notebook (SOLERsources_demo.ipynb) is included, which calls the functions in the correct order