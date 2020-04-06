Comprehensive Walkthrough
=========================
The following examples should form a comprehensive walkthough of downloading the package, getting experimental data into the right form for importing, running the inverse models to generate activation energy(ies), updating published kinetic values both manually and using new experimental data, generating predicted clumped isotope evolution for a given time-temperature history, and generating all necessary plots and tables for data analysis.

For detailed information on class attributes, methods, and parameters, consult the `Package Reference Documentation` or use the ``help()`` command from within Python.

Quick guide
-----------

Basic runthrough::

	#import modules
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import isoclump as ic

	#UPDATE WITH BASIC WALKTHROUGH!


Downloading the package
-----------------------

Using the ``pip`` package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``isoclump`` and the associated dependencies can be downloaded directly from the command line using ``pip``::

	$ pip install isoclump

You can check that your installed version is up to date with the latest release by doing::

	$ pip freeze


Downloading from source
~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, ``isoclump`` source code can be downloaded directly from `my github repo <http://github.com/FluvialSeds/isoclump>`_. Or, if you have git installed::

	$ git clone git://github.com/FluvialSeds/isoclump.git

And keep up-to-date with the latest version by doing::

	$ git pull

from within the isoclump directory.


Dependencies
~~~~~~~~~~~~
The following packages are required to run ``isoclump``:

* `python <http://www.python.org>`_ >= 2.7, including Python 3.x

* `matplotlib <http://matplotlib.org>`_ >= 1.5.2

* `numpy <http://www.numpy.org>`_ >= 1.11.1

* `pandas <http://pandas.pydata.org>`_ >= 0.18.1

* `scipy <http://www.scipy.org>`_ >= 0.18.0

If downloading using ``pip``, these dependencies (except python) are installed
automatically.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
The following packages are not required but are highly recommended:

* `ipython <http://www.ipython.org>`_ >= 4.1.1

Additionally, if you are new to the Python environment or programming using the command line, consider using a Python integrated development environment (IDE) such as:

* `wingware <http://wingware.com>`_

* `Enthought Canopy <https://store.enthought.com/downloads/#default>`_

* `Anaconda <https://www.continuum.io/downloads>`_

* `Spyder <https://github.com/spyder-ide/spyder>`_

Python IDEs provide a "MATLAB-like" environment as well as package management. This option should look familiar for users coming from a MATLAB or RStudio background.

Walkthrough (to be included later)
----------------------------------

