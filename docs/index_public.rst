.. isoclump documentation master file, created by
   sphinx-quickstart on Mon Apr  6 14:24:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the isoclump documentation
=====================================

``isoclump`` is a Python package for analyzing "clumped" isotope kinetic data; it is particularly suited for assessing carbonate clumped isotope (i.e., ∆\ :sub:`47`\) bond reordering and closure temperatures, but will be expanded in the future to include clumped isotopes of other molecular species (e.g., sulfate). This package performs two basic functions: 

(1) it fits ∆\ :sub:`47`\ reordering data from carbonate heating experiments (inverse model) and 
(2) it predicts geologic ∆\ :sub:`47`\ evolution given any time/temperature sample history (forward model). 

For both functions, the package can use any of the available clumped isotope kinetic models: (1) the "pseudo-first-order" model (Passey and Henkes, 2012), (2) the "transient defect/equilibrium" model (Henkes et al., 2014), (3) the "paired raction/diffusion" model (Stolper and Eiler, 2015), and (4) the "distributed activation energy" model (Hemingway and Henkes, 2020).

This package allows users to quickly and easily assess whether their clumped isotope measurements reflect primary signatures, or if these values have been reset during diagenetic heating. Conversely, it also allows users to easily assess geologic cooling rates using the apparent "closure" or "blocking" temperatures recorded in carbonates that have been diagenetically heated.


Package Information
-------------------
:Authors:
  Jordon D. Hemingway (jordon_hemingway@fas.harvard.edu)

:Version:
  0.0.2

:Release:
  8 May 2020

:License:
  GNU GPL v3 (or greater)

:url:
  http://github.com/FluvialSeds/isoclump
  http://pypi.python.org/pypi/isoclump

:doi:
  |doi|

Bug Reports
-----------
This software is still in active deveopment. Please report any bugs directly to me at:

	jordon_hemingway@fas.harvard.edu

How to Cite
-----------
When analyzing data with ``isoclump`` to be used in a peer-reviewed journal, please cite this package as:

* J.D. Hemingway. *isoclump*: open-source tools for clumped isotope kinetic data analysis, 2020-, http://pypi.python.org/pypi/isoclump [online; accessed |date|]

Additionally, please cite the following peer-reviewed manuscript describing the deveopment of the package and clumped isotope data treatment:

* J.D. Hemingway and G.A. Henkes (2020) A distributed activation energy model for clumped isotope bond reordering in carbonates. *Earth and Planetary Science Letters*, **volume**, pages.

If analyzing data with any of the previously published models, please also cite the relevant manuscript(s):

* B.H. Passey and G.A. Henkes (2012) Carbonate clumped isotope bond reordering and geospeedometry. *Earth and Planetary Science Letters*, **351**, 223--236.

*G.A. Henkes et al. (2014) Temperature limits for preservation of primary calcite clumped isotope paleotemperatures. *Geochimica et Cosmochimica Acta*, **139**, 362--382.

*D.A. Stolper and J.M. Eiler (2015) The kinetics of solid-state isotope exchange reactions for clumped isotopes: A study of inorganic calcites and apatites from natural and experimental samples. *American Journal of Science*, **315**, 363--411.


Package features
----------------
``isoclump`` currently contains the following features relevant for clumped isotope data analysis:

* Stores, culls, and plots experimental kinetic isotope data

  * Easily converts between multiple reference frames (e.g., Ghosh, CDES90)
  * Plots forward-modeled predictions using rate parameters (i.e., k values) in order to visually assess goodness of fit

* Estimates the rate parameters (i.e., k values) of bond reordering kinetics for a given set of experimental results using any of the following models:

  * Pseudo-first-order model (Passey and Henkes, 2012)

  * Transient defect/equilibrium model (Henkes et al., 2014)

  * Paired reaction/diffusion model (Stolper and Eiler, 2015)

  * Distributed activation energy model (Hemingway and Henkes, 2020)

    * Regularizes ("smoothes") p(E) using Tikhonov Regularization

    * Automated or user-defined regularization value

* Determines activation energy values using an Arrhenius fit to rate parameters
  
  * Generates Arrhenius plots
  * Allows quick and easy importing of literature data

* Calculates and stores model performance metrics and goodness of fit statistics

* Ability to update published model kinetics, either by manually entering kinetic results or by fitting new experimental data

* Predicts clumped isotope evolution for a given geologic time/temperature history:

  * Allows users to assess if their results reflect primary signatures or diagenetic overprinting.

Future Additions
~~~~~~~~~~~~~~~~
Future versions of ``isoclump`` will aim to include:

* Additional models as they become available

* Kinetics of non-carbonate molecular species (e.g., sulfate) as they become available

License
-------
This product is licensed under the GNU GPL license, version 3 or greater.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   
   quick_guide
   examples
   package_reference

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`

.. * :ref:`modindex`


.. |date| date::
.. |doi| image:: https://zenodo.org/badge/89735636.svg