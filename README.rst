About isotopylog
================
``isotopylog`` is a Python package for analyzing "clumped" isotope kinetic data; it is particularly suited for assessing carbonate clumped isotope (i.e., ∆\ :sub:`47`\) bond reordering and closure temperatures, but will be expanded in the future to include clumped isotopes of other isotopologues (e.g., ∆\ :sub:`48`\) molecular species (e.g., sulfate). This package currently performs two basic functions: 

(1) it fits ∆\ :sub:`47`\ reordering data from carbonate heating experiments (inverse model) and 
(2) it predicts geologic ∆\ :sub:`47`\ evolution given any time/temperature sample history (forward model). 

For both functions, the package can use any of the available clumped isotope kinetic models: (1) the "pseudo-first-order" model (Passey and Henkes, 2012), (2) the "transient defect/equilibrium defect" model (Henkes et al., 2014), (3) the "pair-diffusion" model (Stolper and Eiler, 2015), and (4) the "disordered kinetic" model (Hemingway and Henkes, 2021).

This package allows users to quickly and easily assess whether their clumped isotope measurements reflect primary signatures, or if these values have been reset during diagenetic heating. Similarly, it also allows users to easily assess geologic cooling rates using the apparent "closure" or "blocking" temperatures recorded in carbonates that have been diagenetically heated.

Package Information
-------------------
:Authors:
  Jordon D. Hemingway (jordon.hemingway@erdw.ethz.ch)

:Version:
  0.0.7

:Release:
  25 August 2021

:License:
  GNU GPL v3 (or greater)

:url:
  http://github.com/FluvialSeds/isotopylog
  http://pypi.python.org/pypi/isotopylog

:doi:
  |doi|

How to Cite
-----------
When analyzing data with ``isotopylog`` to be used in a peer-reviewed journal, please cite this package as:

* J.D. Hemingway. *isotopylog*: open-source tools for clumped isotope kinetic data analysis, 2020-, http://pypi.python.org/pypi/isotopylog [online; accessed |date|]

Additionally, please cite the following peer-reviewed manuscript describing the deveopment of the package and clumped isotope data treatment:

* J.D. Hemingway and G.A. Henkes (2021) A disordered kinetic model for clumped isotope bond reordering in carbonates. *Earth and Planetary Science Letters*, **566**, 116962.

If analyzing data with any of the previously published models, please also cite the relevant manuscript(s):

* For the pseudo-first-order model: B.H. Passey and G.A. Henkes (2012) Carbonate clumped isotope bond reordering and geospeedometry. *Earth and Planetary Science Letters*, **351**, 223--236.

* For the transient defect/equilibrium defect model: G.A. Henkes et al. (2014) Temperature limits for preservation of primary calcite clumped isotope paleotemperatures. *Geochimica et Cosmochimica Acta*, **139**, 362--382.

* For the pair-diffusion model: D.A. Stolper and J.M. Eiler (2015) The kinetics of solid-state isotope exchange reactions for clumped isotopes: A study of inorganic calcites and apatites from natural and experimental samples. *American Journal of Science*, **315**, 363--411.


Documentation
-------------
The documentation for the latest release---including detailed package references and comprehensive examples detailing various steps of the data analysis process---is available at:

	http://isotopylog.readthedocs.io

Package features
----------------
``isotopylog`` currently contains the following features relevant for clumped isotope data analysis:

* Stores, culls, and plots experimental kinetic isotope data

  * Easily converts between multiple reference frames (e.g., Ghosh, CDES90, I-CDES)
  * Plots forward-modeled predictions using rate parameters (i.e., k values) in order to visually assess goodness of fit

* Estimates the rate parameters (i.e., k values) of bond reordering kinetics for a given set of experimental results using any of the following models:

  * Pseudo-first-order model (Passey and Henkes, 2012)

  * Transient defect/equilibrium defect model (Henkes et al., 2014)

  * Paired-diffusion model (Stolper and Eiler, 2015)

  * Disordered kinetic model (Hemingway and Henkes, 2021)

    * Calculates best-fit regularized ("smoothed") rate distributions using Tikhonov Regularization

      * Automated or user-defined regularization value
      * Can include negative solutions (e.g., for fitting aragonite data with transient ∆\ :sub:`47`\ increases)

    * Determines best-fit lognormal rate distributions

* Determines activation energy values using an Arrhenius fit to rate parameters
  
  * Generates Arrhenius plots (not possible for negative solutions)
  * Allows quick and easy importing of literature data

* Calculates and stores model performance metrics and goodness of fit statistics

* Ability to update published model kinetics, either by manually entering kinetic results or by fitting new experimental data

* Predicts clumped isotope evolution for a given geologic time/temperature history:

  * Allows users to assess if their results reflect primary signatures or diagenetic overprinting.

* Includes propagated uncertainty estimates for all heating experiment model fits, Arrhenius activation energy fits, and geologic history forward-model predictions.
  
  * Includes analytical uncertainty weighting factors when calculating heating experiment rate values.

  * Accounts for model parameter covariance using a numerical Jacobian approach.


Future Additions
~~~~~~~~~~~~~~~~
Future versions of ``isotopylog`` will aim to include:

* Additional models as they become available

* Kinetics of new isotopologues and non-carbonate molecular species (e.g., sulfate) as they become available


How to Obtain
=============

Source code can be directly downloaded from GitHub:

	http://github.com/FluvialSeds/isotopylog

Binaries can be installed through the Python package index::

	$ pip install isotopylog

License
=======
This product is licensed under the GNU GPL license, version 3 or greater.

Bug Reports
===========
This software is still in active deveopment. Please report any bugs directly to me at:

	jordon.hemingway@erdw.ethz.ch


.. |date| date::
.. |doi| image:: https://zenodo.org/badge/253549515.svg
   :target: https://zenodo.org/badge/latestdoi/253549515