Ocean Tools
=============
This reposity contains oceanographic data analysis tools.

Major depedencies are (vast majority of functions work with only):
numpy

Minor dependencies are (some functions may require):
scipy
matplotlib
gsw

Modules
-------

* GM.py: functions for applying the Garrett and Munk internal wave spectra.
* TKED_parameterisations.py: functions for estimating turbulent kinetic energy dissipation from finescale observations. Including the finescale parameterisation, Thorpe scale estimates and the large eddy method.
* gravity_waves.py: linear inertia-gravity wave dynamics.
* my_savefig.py: format figures for saving to publication specifications. 
* sandwell.py: read data from the Smith and Sandwell bathymetric binary file into a numpy array efficiently.
* utils.py: miscellaneous functions. 
* window.py: functions for splitting data into chunks.

All very much a work in progress.
