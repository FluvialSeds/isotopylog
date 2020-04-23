'''
This module contains helper functions for the kDistribution and EDistribution
classes.

Updated: 22/4/20
By: JDH
'''

#import from future for python 2
from __future__ import(
	division,
	print_function,
	)

#set magic attributes
__docformat__ = 'restructuredtext en'
__all__ = ['fit_Hea14',
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

#import signal processing functions
from scipy.signal import argrelmax

#import necessary isoclump calculation and fitting functions
from .calc_funcs import(
	_calc_A,
	_calc_R,
	_calc_R_stoch,
	_calc_rmse,
	_calc_Rpeq,
	_fexp,
	_fexp_const,
	_fHea14,
	_flin,
	_fSE15,
	_Gaussian,
	_lognormal_decay,
	)

#import necessary isoclump core functions
from .core_functions import(
	derivatize,
	)

#import necessary isoclump dictionaries
from .dictionaries import(
	caleqs,
	)

#import necessary isoclump timedata helper functions
from .timedata_helper import(
	_calc_D_from_G,
	)
	
#function to fit data using Hea14 model
def fit_Hea14(he, p0 = [-7., -7., -7.], thresh = 1e-6):
	'''
	Fits D evolution data using the transient defect/equilibrium model of
	Henkes et al. (2014). The function first solves the first-order linear
	approximation of Passey and Henkes (2012) then solves for the remaining
	kinetic parameters using Eq. 5 of Henkes et al. (2014).

	Parameters
	----------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	p0 : array-like
		Array of paramter guess to initialize the fitting algorithm, in the
		order [ln(kc), ln(kd), ln(k2)]. Defaults to `[-7, -7, -7]`.

	thresh : float
		Curvature threshold to use for extracting the linear region. *All*
		points after the first point that drops below this threshold are
		considered to be in the linear region. Defaults to `1e-6`.

	Returns
	-------
	params : np.ndarray
		Array of resulting parameter values, in the order
		[ln(kc), ln(kd), ln(k2)].

	params_std : np.ndarray
		Uncertainty associated with resulting parameter values; in +- 1 sigma.

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points deemed to be in the linear region.

	Raises
	------
	ValueError
		If the curvature in t vs. ln(G) space never drops below the inputted
		`thresh` value.

	References
	----------
	[1] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
	[2] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
	
	Notes
	-----
	Results are bounded to be non-negative. All calculations are done in lnG
	space and thus only depend on relative changes in D47.
	'''

	#convert to log space
	x = he.tex
	y = np.log(he.Gex)
	y_std = he.Gex_std/he.Gex

	#get p0 into the right format
	p0 = np.array(p0)
	PH12p0 = [p0[0], np.exp(p0[2])/np.exp(p0[1])]

	#run PH12 model to get first-order results
	pfo, pfo_std, rmsefo, nptfo = fit_PH12(he, thresh = thresh, p0 = PH12p0)

	#plug back into full equation and propagate error
	A = y + np.exp(pfo[0])*x
	A_std = np.sqrt(y_std**2 + (x*pfo_std[0]*np.exp(pfo[0]))**2)

	#calculate statistics with full equation and exponential fit
	Hea14p0 = [-np.exp(p0[2]), pfo[1]] #[-k2, kd/k2] in Hea14 notation
	p, pcov = curve_fit(_fexp_const, x, A, Hea14p0,
		sigma = A_std,
		absolute_sigma = True,
		bounds = ([-np.inf, 0],[0, np.inf]), #-k < 0; f.o. intercept > 0
		)

	#extract variables to export

	#k values
	kc = np.exp(pfo[0])
	kd = -p[0]*p[1]
	k2 = -p[0]

	par = np.array([kc, kd, k2]) #[lnkc, lnkd, lnk2]; Hea14 notation
	params = np.log(par)

	#k uncertainty
	kc_std = pfo_std[0]*kc
	kd_std = np.sqrt(k2**2*pcov[1,1] + p[1]**2*pcov[0,0] + 2*kd*pcov[1,0])
	k2_std = pcov[0,0]**0.5

	perr = np.array([kc_std, kd_std, k2_std]) #[kc, kd, k2]; Hea14 notation
	params_std = perr/par

	#calculate Gex_hat
	Ghat = _fHea14(x, kc, kd, k2)

	#convert to D47
	D47hat, _ = _calc_D_from_G(
		he.calibration, 
		he.clumps, 
		he.dex[0,0], 
		Ghat, 
		0, #just pass Gex_hat_std = 0 since we won't use it 
		he.ref_frame, 
		he.T)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47hat)

	return params, params_std, rmse, nptfo

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
		`ic.HeatingExperiment` instance containing the D data to be modeled.

	lam_max : float
		The maximum lnk value to consider. Defaults to `10`.

	lam_min : float
		The minimum lnk value to consider. Defaults to `-50`.

	nlam : int
		The number of lam values in the array such that
		dlam = (lam_max - lam_min)/nlam. Defaults to `300`.

	p0 : array-like
		Array of paramter guess to initialize the fitting algorithm, in the
		order [mu_lam, sig_lam]. Defaults to `[-20, 5]`.

	Returns
	-------
	params : np.ndarray
		Array of resulting parameter values, in the order [mu_lam, sig_lam].

	params_std : np.ndarray
		Uncertainty associated with resulting parameter values; in +- 1 sigma.

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points deemed to be in the linear region.

	lam : np.ndarray
		The array of lam values used for calcualtions, of length `nlam` and
		ranging from `lam_min` to `lam_max`.

	rho_lam : np.ndarray
		The array of corresponding Gaussian rho_lam values.

	References
	----------
	[1] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.

	Notes
	-----
	Results are bounded such that mu_lam is between lam_min and lam_max; sig_lam
	<= (lam_max - lam_min)/2. All calculations are done in lnG space and thus
	only depend on relative changes in D47.
	'''

	#extract values to fit
	x = he.tex
	y = he.Gex
	y_std = he.Gex_std

	#make lam array
	# dlam = (lam_max - lam_min)/nlam
	lam = np.linspace(lam_min, lam_max, nlam)

	#fit model to lambda function to allow inputting constants
	lamfunc = lambda x, mu_lam, sig_lam: _lognormal_decay(
		x, 
		mu_lam, 
		sig_lam, 
		lam_max,
		lam_min,
		nlam
		)

	#solve
	sig_max = (lam_max - lam_min)/2
	p, pcov = curve_fit(lamfunc, x, y, p0,
		sigma = y_std, 
		absolute_sigma = True,
		bounds = ([lam_min, 0.],[lam_max, sig_max]), #mu, sig must be in range
		)

	#extract variables to export
	params = p
	params_std = np.sqrt(np.diag(pcov))
	npt = len(x)

	#calculate rho_lam array
	rho_lam = _Gaussian(lam, *p)

	#calculate Gex_hat
	Ghat = lamfunc(x, *p)

	#convert to D47
	D47hat, _ = _calc_D_from_G(
		he.calibration, 
		he.clumps, 
		he.dex[0,0], 
		Ghat, 
		0, #just pass Gex_hat_std = 0 since we won't use it 
		he.ref_frame, 
		he.T)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47hat)

	return params, params_std, rmse, npt, lam, rho_lam

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

	References
	----------
	[1] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
	[2] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.
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
		omega = _calc_L_curve(
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
		order [ln(k), -ln(intercept)]. Defaults to `[-7, 0.5]`.

	thresh : float
		Curvature threshold to use for extracting the linear region. *All*
		points after the first point that drops below this threshold are
		considered to be in the linear region. Defaults to `1e-6`.

	Returns
	-------
	params : np.ndarray
		Array of resulting parameter values, in the order
		[ln(k), -ln(intercept)].

	params_std : np.ndarray
		Uncertainty associated with resulting parameter values; in +- 1 sigma.

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

	References
	----------
	[1] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.

	Notes
	-----
	Results are bounded such that k is non-negative and intercept is negative;
	intercept value in `params` is the negative of the intercept in lnG vs. t
	space. All calculations are done in lnG space and thus only depend on 
	relative changes in D47.
	'''

	#convert to log space
	x = he.tex
	y = np.log(he.Gex)
	y_std = he.Gex_std/he.Gex

	#calculate curvature
	dydx = derivatize(y, x)
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

	#get initial guess into the right format
	# right format = [-k, -intercept]
	PH12p0 = np.array([-np.exp(p0[0]), -p0[1]])

	#calculate statistics with linear fit to linear region
	p, pcov = curve_fit(_flin, xl, yl, PH12p0,
		sigma = yl_std,
		absolute_sigma = True,
		bounds = ([-np.inf, -np.inf],[0,0]), #-k and -intercept < 0
		)
	
	#extract variables to export
	#get rate and intercept to be positive, and transform rate to lnk
	params = np.array([np.log(-p[0]), -p[1]])

	#get uncertainty
	params_std = np.diag(pcov)**0.5
	params_std[0] = params_std[0]/np.exp(params[0]) #convert to lnk space

	npt = len(xl)

	#calculate Ghat
	Ghat = _fexp(xl, p[0], np.exp(p[1]))

	#convert to D47
	D47hat, _ = _calc_D_from_G(
		he.calibration, 
		he.clumps, 
		he.dex[0,0], 
		Ghat, 
		0, #just pass Ghat_std = 0 since we won't use it 
		he.ref_frame, 
		he.T)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[i0:,0], D47hat)

	return params, params_std, rmse, npt

#function to fit data using SE15 model
def fit_SE15(he, p0 = [-7., -9., 1.0001], z = 6):
	'''
	Fits D evolution data using the paired diffusion model of Stolper and
	Eiler (2015). The function solves for both k1 and k_dif_single as well
	as the initial pair concentration, p0/peq, by solving a modified version
	of SE15 Eq. 9-10. Note that p0/peq can be estimated from SE15 Eq. 17.

	Parameters
	----------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

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

	params_std : np.ndarray
		Uncertainty associated with resulting parameter values; in +- 1 sigma.

	rmse : float
		Root Mean Square Error (in D47 permil units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points included in the model solution.

	References
	----------
	[1] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
	[2] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.

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
	'''

	#extract values to fit
	x = he.tex
	y = he.dex[:,0] #in standard D47 notation
	y_std = he.dex_std[:,0] #in standard D47 notation
	npt = len(he.tex)

	#convert to Dp notation for solving
	yp = y/1000 + 1
	yp_std = y_std/1000

	#calculate constants: Dp470, Dp47eq
	Dp470 = yp[0]
	D47eq = caleqs[he.calibration][he.ref_frame](he.T) #extract from dict.
	Dp47eq = D47eq/1000 + 1 #convert to Dp notation

	#calculate constants: Dppeq
	#calculate R45_stoch, R46_stoch, R47_stoch
	d13C = np.mean(he.dex[:,1]) #use average of all experimental points
	d18O = np.mean(he.dex[:,2]) #use average of all experimental points

	R45_stoch, R46_stoch, R47_stoch = _calc_R_stoch(d13C, d18O, he.iso_params)

	#calculate Rpeq
	Rpeq = _calc_Rpeq(R45_stoch, R46_stoch, R47_stoch, z)

	#combine
	Dppeq = Rpeq/R47_stoch

	#combine constants into list
	cs = [Dppeq, Dp470, Dp47eq]

	#fit model to lambda function to allow inputting constants
	lamfunc = lambda x, k1f, k2f, Dpp0: _fSE15(x, k1f, k2f, Dpp0, *cs)

	#get inputted k0 into the right format
	#k1f
	p00 = np.exp(p0[0])

	#k2f
	p01 = np.exp(p0[1])*(R45_stoch - Rpeq)*(R46_stoch - Rpeq)/Rpeq

	#Dpp0
	p02 = p0[2]*Dppeq

	SE15p0 = [p00, p01, p02]

	#solve
	p, pcov = curve_fit(lamfunc, x, yp, SE15p0,
		sigma = yp_std, 
		absolute_sigma = True,
		bounds = ([0.,0.,1.],[np.inf, np.inf, np.inf]), #k > 0; p0/peq >= 1
		)

	#calculate rmse
	yphat = lamfunc(x, *p)
	yhat = (yphat - 1)*1000 #get back into D47 notation
	rmse = _calc_rmse(y, yhat)

	#Get results into SE15 notation:
	#	k1 = k1f
	#	k_dif_single = k2f*R45_s_eq*R46_s_eq/Rp_eq
	#	p0/peq = Dpp0/Dppeq

	#extract values
	k1 = p[0]
	k_dif_single = p[1]*Rpeq/((R45_stoch - Rpeq)*(R46_stoch - Rpeq))
	p0peq = p[2]/Dppeq

	#combine parameters into array
	params = np.array([np.log(k1), np.log(k_dif_single), p0peq])

	#extract uncertainty
	pstd = np.sqrt(np.diag(pcov))
	k1_std = pstd[0]
	k_dif_single_std = pstd[1]*(R45_stoch - Rpeq)*(R46_stoch - Rpeq)/Rpeq
	p0peq_std = pstd[2]/Dppeq

	params_std = np.array([
		k1_std/k1, 
		k_dif_single_std/k_dif_single, 
		p0peq_std]) #combine into array

	return params, params_std, rmse, npt

#define function for calculating best-fit omega using L-curve approach
def _calc_L_curve(
	he,
	ax = None,
	kink = 1,
	lam_max = 10, 
	lam_min = -50, 
	nlam = 300,
	nom = 150,
	omega_max = 1e2, 
	omega_min = 1e-2,
	plot = False
	):
	'''
	Function to choose the "best" omega value for regularization following
	the Tikhonov Regularization method. The best-fit omega is chosen as the
	value at the point of maximum curvature in a plot of log residual error
	vs. log roughness.
	
	Parameters
	----------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	ax : `None` or plt.axis
		Matplotlib axis to plot on, only relevant if `plot = True`.

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
	
	Returns
	-------
	om_best : np.ndarray
		Array of best-fit omega values, of length `n_peaks`.
	
	ax : plt.axis
		Updated Matplotlib axis containing L-curve plot.
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
			linewidth = 2,
			color = 'k',
			label = 'L-curve')

		ax.scatter(
			res_vec[i],
			rgh_vec[i],
			s = 250,
			facecolor = 'k',
			edgecolor = 'w',
			linewidth = 1.5,
			label = r'best-fit $\omega$')

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
			0.3,
			0.95,
			label1 + '\n' + label2 + '\n' + label3,
			verticalalignment = 'top',
			horizontalalignment = 'left',
			transform = ax.transAxes)

		return om_best, ax

	else:
		return om_best