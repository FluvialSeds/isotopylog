'''
This module contains helper functions for the kDistribution and EDistribution
classes.

Updated: 23/4/20
By: JDH
'''

#import from future for python 2
from __future__ import(
	division,
	print_function,
	)

#set magic attributes
__docformat__ = 'restructuredtext en'
__all__ = ['calc_L_curve',
		   'fit_Hea14',
		   'fit_HH20',
		   'fit_HH20inv',
		   'fit_PH12',
		   'fit_SE15',
		  ]

#import packages
import matplotlib.pyplot as plt
import numpy as np

#import necessary linear algebra functions
from numpy.linalg import (
	inv,
	norm,
	)

#import necessary optimization functions
from scipy.optimize import (
	curve_fit,
	nnls,
	)

# #import signal processing functions
from scipy.signal import argrelmax

#import necessary isoclump calculation and fitting functions
from .calc_funcs import(
	_calc_A,
	_calc_R,
	_calc_R_stoch,
	_calc_rmse,
	_calc_Rpeq,
	_fHea14,
	_fPH12,
	_fSE15,
	_Gaussian,
	_fHH20,
	)

#import necessary isoclump core functions
from .core_functions import(
	derivatize,
	)

#import necessary isoclump dictionaries
# from .dictionaries import(
# 	caleqs,
# 	)

#import necessary isoclump timedata helper functions
from .timedata_helper import(
	_calc_D_from_G,
	)

#define function for calculating best-fit omega using L-curve approach
def calc_L_curve(
	he,
	ax = None,
	kink = 1,
	lam_max = 10, 
	lam_min = -50, 
	nlam = 300,
	nom = 150,
	omega_max = 1e2, 
	omega_min = 1e-2,
	plot = False,
	ld = {},
	pd = {},
	):
	'''
	Function to choose the "best" omega value for regularization following
	the Tikhonov Regularization method. The best-fit omega is chosen as the
	value at the point of maximum curvature in a plot of log residual error
	vs. log roughness.
	
	Parameters
	----------

	he : isoclump.HeatingExperiment
		``ic.HeatingExperiment`` instance containing the D data to be modeled.

	ax : Non` or plt.axis
		Matplotlib axis to plot on, only relevant if ``plot = True``.

	kink : int
		Tells the funciton which L-curve "kink" to use; this is a required
		input since many L-curve solutions appear to have 2 "kinks"; input `1`
		for the lower kink and `2` for the upper kink. Without fail, the lower
		kink appears to be a significantly more robust fit.

	lam_max : scalar
		Maximum lambda value for distribution range; should be at least 4 sigma
		above the mean; defaults to `30`.

	lam_min : scalar
		Minimum lambda value for distribution range; should be at least 4 sigma
		below the mean; defaults to `-30`.

	nlam : int
		Number of nodes in lambda array.

	nom : int
		Number of nodes on omega array.

	omega_max : float
		Maximum omega value to consider, defaults to `1e3`.

	omega_min : float
		Minimum omega value to consider, defaults to `1e-3`.
			
	plot : Boolean
		Boolean telling the funciton whether or not to plot L-curve results.

	ld : dictionary
		Dictionary of keyward arguments to pass for plotting the L-curve line.
		Must contain keywords compatible with ``matplotlib.pyplot.plot``. 
		Defaults to empty dictionary. Only called if ``plot = True``.

	pd : dictionary
		Dictionary of keyward arguments to pass for plotting the best-fit omega
		point. Must contain keywords compatible with 
		``matplotlib.pyplot.scatter``. Defaults to empty dictionary. Only 
		called if ``plot = True``.

	Returns
	-------

	om_best : float
		The 'best fit' omega value.
	
	ax : plt.axis or None
		Updated Matplotlib axis containing L-curve plot.
	
	See Also
	--------

	isoclump.fit_HH20inv
		Method for fitting heating experiment data using the L-curve approach
		of Hemingway and Henkes (2020).

	kDistribution.invert_experiment
		Method for generating a `kDistribution` instance from experimental
		data; can generate L curve if `model = "HH20"` and `fit_reg = True`.

	Examples
	--------

	Basic implementation, assuming a `ic.HeatingExperiment` instance `he`
	exists::
		
		#import modules
		import isoclump as ic

		#assume he is a HeatingExperiment instance
		om_best = ic.calc_L_curve(he, plot = False)

	Similar implementation, but now also generating a plot of the resulting
	L-curve::

		#import modules
		import matplotlib.pyplot as plt

		#make axis instance
		fig,ax = plt.subplots(1,1)

		#assume he is a HeatingExperiment instance
		om_best, ax = ic.calc_L_curve(he, plot = True)

	References
	----------

	[1] Hansen (1994) *Numerical Algorithms*, **6**, 1-35.\n
	[2] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.\n
	[3] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
	'''
	
	#extract arrays
	tex = he.tex
	Gex = he.Gex
	lam = np.linspace(lam_min, lam_max, nlam)
	nt = len(tex)

	#define additional arrays
	log_om_vec = np.linspace(np.log10(omega_min), np.log10(omega_max), nom)
	om_vec = 10**log_om_vec

	res_vec = np.zeros(nom)
	rgh_vec = np.zeros(nom)

	#for each omega value in the vector, calculate the errors
	for i, w in enumerate(om_vec):

		#call the inverse fit parent function
		_, _, res_inv, rgh_inv = fit_HH20inv(
			he, 
			lam_max = lam_max,
			lam_min = lam_min,
			nlam = nlam,
			omega = w
			)

		#store results
		res_vec[i] = res_inv
		rgh_vec[i] = rgh_inv

	#convert to log space
	res_vec = np.log10(res_vec)
	rgh_vec = np.log10(rgh_vec)

	#remove noise after 6 sig figs
	res_vec = np.around(res_vec, decimals = 6)
	rgh_vec = np.around(rgh_vec, decimals = 6)

	#calculate derivatives and curvature
	dydx = derivatize(rgh_vec, res_vec)
	dy2d2x = derivatize(dydx, res_vec)

	#function for curvature
	k = np.abs(dy2d2x / ((1 + dydx**2)**1.5))
	
	#make any infs and nans into zeros just in case these exist
	k[k == np.inf] = 0
	k[k == np.nan] = 0

	#extract peak indices
	pkinds = argrelmax(k)[0]
	pki = np.argsort(k[pkinds])[::-1][:kink+1] #keep top 2 "kink" points
	ivals = pkinds[pki]

	#choose either lower or upper kink to keep
	i = np.sort(ivals)[kink-1]

	#extract om_best
	om_best = om_vec[i]

	#plot if necessary
	if plot is True:

		#create axis if necessary
		if ax is None:
			_, ax = plt.subplots(1, 1)

		#plot results
		ax.plot(
			res_vec,
			rgh_vec,
			# label = 'L-curve',
			**ld)

		ax.scatter(
			res_vec[i],
			rgh_vec[i],
			s = 250,
			# label = r'best-fit $\omega$',
			**pd)

		#set axis labels and text

		xlab = r'residual error, $\log_{10} \left( \frac{\|\|' \
			r'\mathbf{A}\cdot \mathbf{p} - \mathbf{g} \|\|}{\sqrt{n_{j}}}' \
			r'\right)$'
		ax.set_xlabel(xlab)

		ylab = r'roughness, $\log_{10} \left( \frac{\|\| \mathbf{R}' \
			r'\cdot\mathbf{p} \|\|}{\sqrt{n_{l}}} \right)$'

		ax.set_ylabel(ylab)

		label1 = r'best-fit $\omega$ = %.3f (kink = %.0f)' %(om_best, kink)

		label2 = (
			r'$log_{10}$ (resid. err.) = %.3f' %(res_vec[i]))

		label3 = (
			r'$log_{10}$ (roughness)  = %0.3f' %(rgh_vec[i]))
		
		ax.text(
			0.95,
			0.95,
			label1 + '\n' + label2 + '\n' + label3,
			verticalalignment = 'top',
			horizontalalignment = 'right',
			transform = ax.transAxes)

		return om_best, ax

	else:
		return om_best

#function to fit data using Hea14 model
def fit_Hea14(he, p0 = [-7., -7., -7.]):
	'''
	Fits D evolution data using the transient defect/equilibrium model of
	Henkes et al. (2014) (Equation 5).

	Parameters
	----------

	he : isoclump.HeatingExperiment
		`ic.HeatingExperiment` instance containing the D data to be modeled.

	p0 : array-like
		Array of paramter guess to initialize the fitting algorithm, in the
		order [ln(kc), ln(kd), ln(k2)]. Defaults to `[-7, -7, -7]`.

	Returns
	-------

	params : np.ndarray
		Array of resulting parameter values, in the order
		[ln(kc), ln(kd), ln(k2)].

	params_cov : np.ndarray
		Covariance matrix associated with the resulting parameter values, of
		shape [3 x 3]. The +/- 1 sigma uncertainty for each parameter can be 
		calculated as ``np.sqrt(np.diag(params_cov))``

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points included in the model solution.
	
	Notes
	-----

	Results are bounded to be non-negative. All calculations are done in lnG
	space and thus only depend on relative changes in D47.

	See Also
	--------

	isoclump.fit_HH20
		Method for fitting heating experiment data using the distributed
		activation energy model of Hemingway and Henkes (2020).

	isoclump.fit_PH12
		Method for fitting heating experiment data using the pseudo first-
		order method of Passey and Henkes (2012). Called to determine
		linear region.

	isoclump.fit_SE15
		Method for fitting heatinge experiment data using the paird diffusion
		model of Stolper and Eiler (2015).

	kDistribution.invert_experiment
		Method for generating a `kDistribution` instance from experimental
		data.

	Examples
	--------

	Basic implementation, assuming a `ic.HeatingExperiment` instance `he`
	exists::
		
		#import modules
		import isoclump as ic

		#assume he is a HeatingExperiment instance
		results = ic.fit_Hea14(he, p0 = [-7., -7., -7.])

	References
	----------

	[1] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
	'''

	#extract values
	x = he.tex
	y = he.Gex
	y_std = he.Gex_std
	npt = len(x)

	#solve the model
	params, params_cov = curve_fit(_fHea14, x, y, p0,
		sigma = y_std,
		absolute_sigma = True,
		bounds = (-np.inf, np.inf), #all lnk are unbounded
		)

	#calculate Ghat
	Ghat = _fHea14(x, *params)

	#convert to D47
	D47hat, _ = _calc_D_from_G(
		he.dex[0,0],
		Ghat,
		he.T,
		calibration = he.calibration,
		clumps = he.clumps,
		G_std = None,
		ref_frame = he.ref_frame
		)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47hat)

	return params, params_cov, rmse, npt

#function to fit data using HH20 lognormal model
def fit_HH20(he, lam_max = 10, lam_min = -50, nlam = 300, p0 = [-20, 5]):
	'''
	Fits D evolution data using the distributed activation energy model of
	Hemingway and Henkes (2020). This function solves for mu_lam and sig_lam,
	the mean and standard deviation of a Gaussian distribution in lnk space.
	See HH20 Eq. X for notation and details.

	Parameters
	----------

	he : isoclump.HeatingExperiment
		``ic.HeatingExperiment`` instance containing the D data to be modeled.

	lam_max : float
		The maximum lnk value to consider. Defaults to ``10``.

	lam_min : float
		The minimum lnk value to consider. Defaults to ``-50``.

	nlam : int
		The number of lam values in the array such that
		dlam = (lam_max - lam_min)/nlam. Defaults to ``300``.

	p0 : array-like
		Array of paramter guess to initialize the fitting algorithm, in the
		order [ln(k_mu), ln(k_sig)]. Defaults to ``[-20, 5]``.

	pcov : np.array
		Array of outputted parameter covariance. To be passed as hidden
		attribute to kDistribution and used for calculating forward-modeling
		uncertainty.

	Returns
	-------

	params : np.ndarray
		Array of resulting parameter values, in the order [ln(k_mu), ln(k_sig)].

	params_cov : np.ndarray
		Covariance matrix associated with the resulting parameter values, of
		shape [3 x 3]. The +/- 1 sigma uncertainty for each parameter can be 
		calculated as ``np.sqrt(np.diag(params_cov))``

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points included in the model solution.

	lam : np.ndarray
		The array of lam values used for calcualtions, of length `nlam` and
		ranging from `lam_min` to `lam_max`.

	rho_lam : np.ndarray
		The array of corresponding Gaussian rho_lam values.

	Notes
	-----

	Results are bounded such that mu_lam is between lam_min and lam_max; sig_lam
	<= (lam_max - lam_min)/2. All calculations are done in G space and thus
	only depend on relative changes in D47.

	See Also
	--------

	isoclump.fit_Hea14
		Method for fitting heating experiment data using the transient defect/
		equilibrium model of Henkes et al. (2014). 'Hea14' can be considered
		an updated version of the present method.

	isoclump.fit_HH20inv
		Method for fitting heating experiment data using the L-curve approach
		of Hemingway and Henkes (2020).

	isoclump.fit_PH12
		Method for fitting heating experiment data using the pseudo first-
		order method of Passey and Henkes (2012). Called to determine
		linear region.

	isoclump.fit_SE15
		Method for fitting heatinge experiment data using the paird diffusion
		model of Stolper and Eiler (2015).

	kDistribution.invert_experiment
		Method for generating a `kDistribution` instance from experimental
		data.

	Examples
	--------

	Basic implementation, assuming a `ic.HeatingExperiment` instance `he`
	exists::
		
		#import modules
		import isoclump as ic

		#assume he is a HeatingExperiment instance
		results = ic.fit_HH20(he)

	References
	----------

	[1] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
	'''

	#extract values to fit
	x = he.tex
	y = he.Gex
	y_std = he.Gex_std
	npt = len(x)

	#make lam array
	# dlam = (lam_max - lam_min)/nlam
	lam = np.linspace(lam_min, lam_max, nlam)

	#fit model to lambda function to allow inputting constants
	lamfunc = lambda x, mu_lam, sig_lam: _fHH20(
		x, 
		mu_lam, 
		sig_lam, 
		lam_max,
		lam_min,
		nlam
		)

	#solve
	sig_max = (lam_max - lam_min)/2
	params, params_cov = curve_fit(lamfunc, x, y, p0,
		sigma = y_std, 
		absolute_sigma = True,
		bounds = ([lam_min, 0.],[lam_max, sig_max]), #mu, sig must be in range
		)

	#calculate rho_lam array
	rho_lam = _Gaussian(lam, *params)

	#calculate Gex_hat
	Ghat = lamfunc(x, *params)

	#convert to D47
	D47hat, _ = _calc_D_from_G(
		he.dex[0,0],
		Ghat,
		he.T,
		calibration = he.calibration,
		clumps = he.clumps,
		G_std = None,
		ref_frame = he.ref_frame
		)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47hat)

	return params, params_cov, rmse, npt, lam, rho_lam

#function to fit data using the HH20 inverse model
def fit_HH20inv(
	he,
	lam_max = 10,
	lam_min = -50,
	nlam = 300,
	omega = 'auto',
	**kwargs
	):
	'''
	Fits D evolution data using the distributed activation energy model of
	Hemingway and Henkes (2020). This function solves for rho_lam, the
	regularized distribution of rates in lnk space. See HH20 Eq. X for
	notation and details. This function can estimate best-fit omega using
	Tikhonov regularization.
	
	Parameters
	----------

	he : isoclump.HeatingExperiment
		`ic.HeatingExperiment` instance containing the D data to be modeled.

	lam_max : float
		The maximum lnk value to consider. Defaults to `10`.

	lam_min : float
		The minimum lnk value to consider. Defaults to `-50`.

	nlam : int
		The number of lam values in the array such that
		dlam = (lam_max - lam_min)/nlam. Defaults to `300`.

	omega : str or float
		The "smoothing parameter" to use. This can be a number or `auto`; if 
		'auto', the function uses Tikhonov regularization to calculate the
		optimal omega value. Defaults to `auto`.
	
	Returns
	-------

	rho_lam_inv : array-like
		Resulting regularized rho distribution, of length `n_lam`.

	omega : float
		If inputed `omega = 'auto'`, then this is the best-fit omega value.
		If inputted omega was a number, this is simply same as the inputted
		value.

	res_inv : float
		Root mean square error of the inverse model fit.

	rgh_inv : float
		Roughness norm of the inverse model fit.

	Raises
	------

	TypeError
		If `omega` is not 'Auto' or float or int type.

	TypeError
		If unexpected keyword arguments are passed to `calc_L_curve`.

	See Also
	--------

	isoclump.fit_HH20
		Method for fitting heating experiment data using the lognormal model
		of Hemingway and Henkes (2020).

	kDistribution.invert_experiment
		Method for generating a `kDistribution` instance from experimental
		data.

	Examples
	--------

	Basic implementation, assuming a `ic.HeatingExperiment` instance `he`
	exists::
		
		#import modules
		import isoclump as ic

		#assume he is a HeatingExperiment instance
		results = ic.fit_HH20inv(he, omega = 'auto')

	Same implementation, but if best-fit `omega` is known a priori::

		#import modules
		import isoclump as ic

		#assume best-fit omega is 3
		omega = 3

		#assume he is a HeatingExperiment instance
		results = ic.fit_HH20inv(he, omega = 3)

	References
	----------

	[1] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.\n
	[2] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
	'''

	#extract variables
	tex = he.tex
	Gex = he.Gex
	lam = np.linspace(lam_min, lam_max, nlam)
	nt = len(tex)

	#calculate A matrix
	A = _calc_A(tex, lam)
	
	#calculate regularization matrix, R
	R = _calc_R(nlam)
	
	#calculate omega using L curve if necessary:
	if omega in ['auto', 'Auto']:

		#run L curve function to calculate best-fit omega
		omega = calc_L_curve(
			he,
			lam_max = lam_max,
			lam_min = lam_min,
			nlam = nlam,
			plot = False,
			**kwargs
			)

	#make sure omega is a scalar
	elif not isinstance(omega, float) and not isinstance(omega, int):

		omt = type(omega).__name__

		raise TypeError(
			'Attempting to input `omega` of type %s. Must be `int`, `float`'
			' or "auto".' % omt)

	#ensure it's float
	else:
		omega = float(omega)

	#concatenate A+R and Gex+zeros
	A_reg = np.concatenate(
		(A, R*omega))

	Gex_reg = np.concatenate(
		(Gex, np.zeros(nlam + 1)))

	#calculate inverse results and estimated G
	rho_lam_inv, _ = nnls(A_reg, Gex_reg)
	Ghat = np.inner(A, rho_lam_inv)
	rgh = np.inner(R, rho_lam_inv)

	#calculate errors
	res_inv = norm(Gex - Ghat)/nt**0.5
	rgh_inv = norm(rgh)/nlam**0.5

	return rho_lam_inv, omega, res_inv, rgh_inv

#function to fit data using PH12 model
def fit_PH12(he, p0 = [-7., 0.5], thresh = 1e-6):
	'''
	Fits D evolution data using the first-order model approximation of Passey
	and Henkes (2012). The function uses curvature in t vs. ln(G) space to
	extract the linear region and only fits this region.

	Parameters
	----------

	he : isoclump.HeatingExperiment
		`ic.HeatingExperiment` instance containing the D data to be modeled.

	p0 : array-like
		Array of paramter guess to initialize the fitting algorithm, in the
		order [ln(k), intercept]. Defaults to ``[-7, 0.5]``.

	thresh : float
		Curvature threshold to use for extracting the linear region. *All*
		points after the first point that drops below this threshold are
		considered to be in the linear region. Defaults to ``1e-6``.

	Returns
	-------

	params : np.ndarray
		Array of resulting parameter values, in the order
		[ln(k), intercept].

	params_cov : np.ndarray
		Covariance matrix associated with the resulting parameter values, of
		shape [2 x 2]. The +/- 1 sigma uncertainty for each parameter can be 
		calculated as ``np.sqrt(np.diag(params_cov))``

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit. Only 
		includes data points that are deemed to be in the linear region.

	npt : int
		Number of data points deemed to be in the linear region.

	Raises
	------

	ValueError
		If the curvature in t vs. ln(G) space never drops below the inputted
		`thresh` value.

	Notes
	-----

	Results are bounded such that k and intercept are non-negative; intercept 
	value in ``params`` is the intercept in G vs. t space. All calculations 
	are done in G space and thus only depend on relative changes in D47.

	See Also
	--------

	isoclump.fit_Hea14
		Method for fitting heating experiment data using the transient defect/
		equilibrium model of Henkes et al. (2014). 'Hea14' can be considered
		an updated version of the present method.

	isoclump.fit_HH20
		Method for fitting heating experiment data using the distributed
		activation energy model of Hemingway and Henkes (2020).

	isoclump.fit_SE15
		Method for fitting heatinge experiment data using the paird diffusion
		model of Stolper and Eiler (2015).

	isoclump.kDistribution.invert_experiment
		Method for generating a ``kDistribution`` instance from experimental
		data.

	Examples
	--------

	Basic implementation, assuming a ``ic.HeatingExperiment`` instance ``he``
	exists::
		
		#import modules
		import isoclump as ic

		#assume he is a HeatingExperiment instance
		results = ic.fit_PH12(he, thresh = 1e-6)

	References
	----------

	[1] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
	'''

	#extract values to fit
	x = he.tex
	y = he.Gex
	y_std = he.Gex_std

	#calculate curvature in ln(y) space
	dydx = derivatize(np.log(y), x)
	dy2d2x = derivatize(dydx, x)
	k = np.abs(dy2d2x / ((1 + dydx**2)**1.5))

	#retain all points after curvature drops below threshold
	try:
		i0 = np.where(k < thresh)[0][0]

	except IndexError:
		raise ValueError(
			't vs. lnG curvature never goes below inputted value of %.1e;'
			' cannot extract linear region. Raise `thresh` value.' % thresh)

	xl = x[i0:]
	yl = y[i0:]
	yl_std = y_std[i0:]
	npt = len(xl)

	#calculate statistics with linear fit to linear region
	params, params_cov = curve_fit(_fPH12, xl, yl, p0,
		sigma = yl_std,
		absolute_sigma = True,
		bounds = ([-np.inf,0],[np.inf,1]), #lnk unbounded; 0 < int. < 1
		)

	#calculate Ghat
	Ghat = _fPH12(xl, *params)

	#convert to D47
	D47hat, _ = _calc_D_from_G(
		he.dex[0,0],
		Ghat,
		he.T,
		calibration = he.calibration,
		clumps = he.clumps,
		G_std = None,
		ref_frame = he.ref_frame
		)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[i0:,0], D47hat)

	return params, params_cov, rmse, npt

#function to fit data using SE15 model
def fit_SE15(he, p0 = [-7., -9., 1.0001], z = 6):
	'''
	Fits D evolution data using the paired diffusion model of Stolper and
	Eiler (2015). The function solves for both k1 and k_dif_single as well
	as the initial pair concentration, p0/peq, by solving a modified version
	of SE15 Eq. 9-10. Note that p0/peq can be estimated from SE15 Eq. 17.

	Parameters
	----------

	he : isoclump.HeatingExperiment
		`ic.HeatingExperiment` instance containing the D data to be modeled.

	p0 : array-like
		Array of paramter guess to initialize the fitting algorithm, in the
		order [ln(k1), ln(k_dif_single), p0/peq]. Defaults to 
		`[-7, -9, 1.0001]`.

	z : int
		The mineral lattice coordination number to use for calculating the
		concentration of pairs. Defaults to `6` following Stolper and Eiler
		(2015).

	Returns
	-------

	params : np.ndarray
		Array of resulting parameter values, in the order
		`[ln(k1), ln(k_dif_single), p0/peq]`.

	params_cov : np.ndarray
		Covariance matrix associated with the resulting parameter values, of
		shape [3 x 3]. The +/- 1 sigma uncertainty for each parameter can be 
		calculated as ``np.sqrt(np.diag(params_cov))``

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points included in the model solution.

	Notes
	-----

	This function uses the average of all experimental d13C and d18O values
	when calculating stochastic statistics. If d13C and d18O values change
	considerably throughout the course of an experiment, this could cause
	slight inconsistencies in results.

	This function uses the average of SE15 Eq. 13a and Eq. 13b when calculating
	pair concentrations. According to SE15, the relative difference between
	these equations is ~1-2 percent, so this should be an arbitrary choice.

	Results are bounded such that k values are non-negative and p0/peq >= 1.
	Calculations depend on stochastic 'pair' concentrations, which are a
	function of the chosen isotope parameters and thus might be sensitive to
	this choice. See Daëron et al. (2016) and the `calc_R` and `calc_d`
	functions for details.

	As mentioned in Stolper and Eiler (2015), results appear to be sensitive
	to the choice of initial conditions; for this reason, the user can pass
	different choices of p0 when solving.

	See Also
	--------

	isoclump.fit_Hea14
		Method for fitting heating experiment data using the transient defect/
		equilibrium model of Henkes et al. (2014). 'Hea14' can be considered
		an updated version of the present method.

	isoclump.fit_HH20
		Method for fitting heating experiment data using the distributed
		activation energy model of Hemingway and Henkes (2020).

	isoclump.fit_PH12
		Method for fitting heating experiment data using the pseudo first-
		order method of Passey and Henkes (2012). Called to determine
		linear region.

	kDistribution.invert_experiment
		Method for generating a `kDistribution` instance from experimental
		data.

	Examples
	--------

	Basic implementation, assuming a `ic.HeatingExperiment` instance `he`
	exists::
		
		#import modules
		import isoclump as ic

		#assume some a priori guess at p0; results are sensitive to choice of
		# p0 as described in SE15
		p0 = [-7., -9., 1.00014]

		#assume he is a HeatingExperiment instance
		results = ic.fit_SE15(he, p0 = p0)

	References
	----------

	[1] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.\n
	[2] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.
	'''

	#extract values to fit
	x = he.tex
	y = he.dex[:,0] #in standard D47 notation
	y_std = he.dex_std[:,0] #in standard D47 notation
	npt = len(he.tex)

	#calculate additional constants for model fit
	#calculate constants: D0, Deq
	D0 = y[0]
	Deq = he.caleq(he.T)

	#calculate constants: Dppeq
	#calculate R45_stoch, R46_stoch, R47_stoch
	d13C = np.mean(he.dex[:,1]) #use average of all experimental points
	d18O = np.mean(he.dex[:,2]) #use average of all experimental points

	R45_stoch, R46_stoch, R47_stoch = _calc_R_stoch(d13C, d18O, he.iso_params)

	#calculate Rpeq and convert to Dppeq
	Rpeq = _calc_Rpeq(R45_stoch, R46_stoch, R47_stoch, z)
	Dppeq = Rpeq/R47_stoch

	#combine constants into list
	cs = [D0, Deq, Dppeq, he]

	#fit model to lambda function to allow inputting constants
	lamfunc = lambda t, lnk1f, lnkds, p0peq: _fSE15(
		t, 
		lnk1f, 
		lnkds, 
		p0peq, 
		*cs
		)

	#solve
	params, params_cov = curve_fit(lamfunc, x, y, p0,
		sigma = y_std, 
		absolute_sigma = True,
		bounds = ([-np.inf,-np.inf,1.],[np.inf,np.inf,np.inf]), #k unbounded
		)

	#calculate Dex_hat
	D47hat = lamfunc(x, *params)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47hat)

	return params, params_cov, rmse, npt
