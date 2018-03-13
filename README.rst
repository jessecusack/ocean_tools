Ocean Tools
=============
This repository contains oceanographic data analysis tools.

Modules
-------

* GM: functions for applying the Garrett and Munk internal wave spectra.
* TKED: functions for estimating turbulent kinetic energy dissipation from finescale observations. Including the finescale parameterisation, Thorpe scale estimates and the large eddy method.
* gravity_waves: linear inertia-gravity wave dynamics.
* sandwell: read data from the Smith and Sandwell bathymetric binary file into a numpy array efficiently.
* utils: miscellaneous functions.
* window: functions for splitting data into chunks.

All very much a work in progress.

Installation
------------

First clone or download the repository. Then install using pip:

``cd ocean_tools``

``pip install .``

Optionally, use the the -e flag, to make it editable.

``pip install -e .``
