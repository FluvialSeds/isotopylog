'''
This module contains helper functions for the RateData classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_fit_PH12',
			'_fit_Hea14',
			'_fit_SE15',
			'_fit_HH20',
			]

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

# from .dictionaries import(
# 	caleqs
# 	)

from .core_functions import(
	derivatize,
	)


#inverse model fitting functions
def _fit_PH12(he, thresh):
	'''
	Fits D evolution data using the first-order model approximation of Passey
	and Henkes (2012). The function uses curvature in t vs. ln(G) space to
	extract the linear region and only fits this region.

	Paramters
	---------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	thresh : float
		Curvature threshold to use for extracting the linear region. *All*
		points after the first point that drops below this threshold are
		considered to be in the linear region.

	Returns
	-------
	k : np.ndarray
		Array of resulting k values, in the order [k, ln(intercept)].

	k_std : np.ndarray
		Uncertainty associated with resulting k values.

	rmse : float
		Root Mean Square Error uncertainty (in G units) of the model fit. Only
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
	'''

	#convert to log space
	x = he.tex
	y = np.log(he.Gex)

	#assume error is symmetric in log space, which isn't strictly true but is
	# a good approximation for small numbers
	y_std = (np.log(he.Gex + he.Gex_std) - np.log(he.Gex - he.Gex_std))/2

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

	#calculate statistics with linear fit to linear range
	p0 = [-1e-3,-0.5] #initial guess
	p, pcov = curve_fit(_flin, xl, yl, p0,
		sigma = yl_std,
		absolute_sigma = True
		)
	
	#extract variables to export
	k = -p #get rate and intercept to be positive
	k_std = np.diag(pcov)**0.5
	npt = len(xl)

	#calculate rmse in G space
	Gex_hat = _fexp(xl, p[0], np.exp(p[1]))
	rmse = _calc_rmse(np.exp(yl), Gex_hat)

	return k, k_std, rmse, npt
	
def _fit_Hea14(he, thresh):
	'''
	Fits D evolution data using the transient defect/equilibrium model of
	Henkes et al. (2014). The function first solves the first-order linear
	approximation of Passey and Henkes (2012) then solves for the remaining
	kinetic parameters using Eq. 5 of Henkes et al. (2014).

	Paramters
	---------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	thresh : float
		Curvature threshold to use for extracting the linear region. *All*
		points after the first point that drops below this threshold are
		considered to be in the linear region.

	Returns
	-------
	k : np.ndarray
		Array of resulting k values, in the order [kc, kd, k2].

	k_std : np.ndarray
		Uncertainty associated with resulting k values.

	rmse : float
		Root Mean Square Error uncertainty (in G units) of the model fit.
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
	'''

	#convert to log space
	x = he.tex
	y = np.log(he.Gex)

	#assume error is symmetric in log space, which isn't strictly true but is
	# a good approximation for small numbers
	y_std = (np.log(he.Gex + he.Gex_std) - np.log(he.Gex - he.Gex_std))/2

	#run PH12 model to get first-order results
	kfo, kfo_std, rmsefo, nptfo = _fit_PH12(he, thresh)

	#plug back into full equation and propagate error
	A = y + kfo[0]*x
	A_std = np.sqrt(y_std**2 + (x*kfo_std[0])**2)

	#calculate statistics with full equation and exponential fit
	p0 = [-1e-3, kfo[1]] #[-k2, kd/k2] in Hea14 notation
	p, pcov = curve_fit(_fexp_const, x, A, p0,
		sigma = A_std,
		absolute_sigma = True
		)

	#extract variables to export

	#k values
	kc = kfo[0]
	kd = -p[0]*p[1]
	k2 = -p[0]

	k = np.array([kc, kd, k2]) #[kc, kd, k2] in Hea14 notation

	#k uncertainty
	kc_std = kfo_std[0]
	kd_std = np.sqrt(k2**2*pcov[1,1] + p[1]**2*pcov[0,0] + 2*kd*pcov[1,0])
	k2_std = pcov[0,0]**0.5

	k_std = np.array([kc_std, kd_std, k2_std]) #[kc, kd, k2] in Hea14 notation

	#calculate rmse
	Gex_hat = _fHea14(x, *k)
	rmse = _calc_rmse(np.exp(y), Gex_hat)

	# npt = len(x) #retained all points

	return k, k_std, rmse, nptfo

def _fit_SE15(heatingexperiment):

	return k

def _fit_HH20(heatingexperiment):

	return k, pk


#linear function for curve fitting
def _flin(x, c0, c1):
	'''
	Defines a straight line.

	Parameters
	----------
	x : array-like
		The x values.

	c0 : float
		The slope.

	c1 : float
		The intercept.

	Returns
	-------
	yhat : array-like
		Resulting array of y values.
	'''

	return c0*x + c1

#exponential function for curve fitting
def _fexp(x, c0, c1):
	'''
	Defines an exponential decay.

	Parameters
	----------
	x : array-like
		The x values.

	c0 : float
		The exponential value; e.g., the rate constant.

	c1 : float
		The pre-exponential factor.

	Returns
	-------
	yhat : array-like
		Resulting array of y values.
	'''
	return np.exp(x*c0)*c1

#exponential function for curve fitting
def _fexp_const(x, c0, c1):
	'''
	Defines an exponential decay with a constant. used for Hea14 model fit.

	Parameters
	----------
	x : array-like
		The x values.

	c0 : float
		The exponential value; e.g., the rate constant.

	c1 : float
		The pre-exponential factor.

	Returns
	-------
	yhat : array-like
		Resulting array of y values.
	'''
	return (np.exp(x*c0) - 1)*c1

#rmse function for curve fitting
def _calc_rmse(y,yhat):
	'''
	Defines a straight line.

	Parameters
	----------
	y : array-like
		The true y values.

	yhat : array-like
		The model-estimated y values.

	Returns
	-------
	rmse : float
		Resulting root mean square error value.
	'''
	return np.sqrt(np.sum((y-yhat)**2)/len(y))



def _fHea14(t, kc, kd, k2):
	'''
	Estimates G using the "transient defect/equilibrium" model of Henkes et
	al. (2014) (Eq. 5).

	Parameters
	----------
	t : array-like
		The array of time points.

	kc : float
		The first-order rate constant.

	kd : float
		The transient defect rate constant.

	k2 : float
		The transient defect disappearance rate constant.

	Returns
	-------
	Ghat : array-like
		Resulting estimated G values.

	References
	----------
	[1] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
	'''

	lnGhat = -kc*t + (kd/k2)*(np.exp(-k2*t) - 1)

	return np.exp(lnGhat)