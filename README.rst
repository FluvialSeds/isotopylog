About isoclump
=================
``isoclump`` is a Python package for analyzing "clumped" isotope kinetic data; it is particularly suited for assessing carbonate clumped isotope (i.e., ∆:sub:`47`\) bond reordering and closure temperatures, but can be expanded to include clumped isotopes of other molecular species (e.g., sulfate). This package performs two basic functions: (1) to fit ∆:sub:`47`\ reordering data from carbonate heating experiments (inverse model) and (2) to predict geologic ∆:sub:`47`\ evolution given any time/temperature sample history (forward model). For both functions, the package can use any of the available clumped isotope kinetic models: Passey and Henkes (2012), Henkes et al. (2014), Stolper and Eiler (2015), or Hemingway and Henkes (202x).

This package allows users to quickly and easily assess whether their clumped isotope measurements reflect primary signatures, or if these values have been reset during diagenetic heating. It was developed as a summplement to Hemingway and Henkes (202x) "A distributed activation energy model for clumped isotope bond reordering in carbonates", *Earth and Planetary Science Letters*, **volume**, pages.


Package Information
-------------------
:Authors:
  Jordon D. Hemingway (jordon_hemingway@fas.harvard.edu)

:Version:
  0.0.1

:Release:
  6 April 2020

:License:
  GNU GPL v3 (or greater)

:url:
  http://github.com/FluvialSeds/isoclump
  http://pypi.python.org/pypi/isoclump

:doi:
  UPDATE THIS DOI!
  |doi|

How to Cite
-----------
When analyzing data with ``isoclump`` to be used in a peer-reviewed journal, please cite this package as:

* J.D. Hemingway. *isoclump*: open-source tools for clumped isotope kinetic data analysis, 2020-, http://pypi.python.org/pypi/isoclump [online; accessed |date|]

Additionally, please cite the following peer-reviewed manuscript describing the deveopment of the package and clumped isotope data treatment:

* J.D. Hemingway and G.A. Henkes (202x) A distributed activation energy model for clumped isotope bond reordering in carbonates. *Earth and Planetary Science Letters*, **volume**, pages.

If analyzing data with any of the previously published models, please also cite the relevant manuscript(s):

* B.H. Passey and G.A. Henkes (2012) Carbonate clumped isotope bond reordering and geospeedometry. *Earth and Planetary Science Letters*, **351**, 223--236.

*G.A. Henkes et al. (2014) Temperature limits for preservation of primary calcite clumped isotope paleotemperatures. *Geochimica et Cosmochimica Acta*, **139**, 362--382.

*D.A. Stolper and J.M. Eiler (2015) The kinetics of solid-state isotope exchange reactions for clumped isotopes: A study of inorganic calcites and apatites from natural and experimental samples. *American Journal of Science*, **315**, 363--411.


Documentation
-------------
The documentation for the latest release, including detailed package references as well as a comprehensive walkthrough for analyzing clumped isotope kinetic data, is available at:

	http://isoclump.readthedocs.io

Package features
----------------
``isoclump`` currently contains the following features relevant to clumped isotope kinetic analysis:

* Stores, culls, and plots experimental kinetic isotope data

  * Easily converts between multiple reference frames (e.g., Ghosh, CDES90)

* Estimates the activation energy(ies) of bond reordering kinetics for a given set of experimental results using any of the following models:

  * Pseudo-first-order model (Passey and Henkes, 2012)

  * Non-first-order model (Henkes et al., 2014)

  * Paired model (Stolper and Eiler, 2015)

  * Distributed activation energy model (Hemingway and Henkes, 202x)

    * Regularizes ("smoothes") p(E) using Tikhonov Regularization

    * Automated or user-defined regularization value

* Calculates and stores model performance metrics and goodness of fit statistics

* Ability to update published model kinetics, either by manually entering kinetic results or by fitting new experimental data

* Predicts clumped isotope evolution for a given geologic time/temperature history:

  * Allows users to assess if their results reflect primary signatures or diagenetic overprinting.

  * Generates simplified diagrams showing the time-temperature domains where primary signatures are predicted to be preserved (a la Hankes et al., 2014).

Future Additions
~~~~~~~~~~~~~~~~
Future versions of ``isoclump`` will aim to include:

* Additional models as they become available

* Kinetics of non-carbonate molecular species (e.g., sulfate) as they become available


How to Obtain
=============

Source code can be directly downloaded from GitHub:

	http://github.com/FluvialSeds/isoclump

Binaries can be installed through the Python package index::

	$ pip install isoclump

License
=======
This product is licensed under the GNU GPL license, version 3 or greater.

Bug Reports
===========
This software is still in active deveopment. Please report any bugs directly to me at:

	jordon_hemingway@fas.harvard.edu


.. |date| date::
.. |doi| image:: https://zenodo.org/badge/66090463.svg