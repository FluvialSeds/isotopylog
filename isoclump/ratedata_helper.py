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

from numpy.linalg import inv
from numpy import eye
from scipy.optimize import curve_fit

# from .dictionaries import(
# 	caleqs
# 	)

from .core_functions import(
	derivatize,
	)

from .dictionaries import(
	caleqs,
	d47_isoparams,
	)

#TO DO:
# * SHOULD I REPORT PH12 AND HEA14 RMSE IN D47 NOTATION RATHER THAN G??


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

def _fit_SE15(he, z):
	'''
	Add docstring

	Notes: 
	1. uses average of all experimental d13C and d18O for calculating
	stochastic statistics.
	2. Uses the average of both pair calculation equations (13a/b).
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
	lamfunc = lambda x, k1f, k2f, Dpp0: _SE15_fin_diff(x, k1f, k2f, Dpp0, *cs)

	#set initial guess
	p0 = [1e-5, 1e-5, 0.99*Dppeq]

	#solve
	p, pcov = curve_fit(lamfunc, x, yp, p0,
		sigma = yp_std, 
		absolute_sigma = True
		)

	#calculate rmse
	yphat = lamfunc(x, *p)
	yhat = (yphat - 1)*1000 #get back into D47 notation
	rmse = _calc_rmse(y, yhat)

	#Get results into SE15 notation:
	#	k1 = k1f
	#	k_dif_single = k2f*R45_s_eq*R46_s_eq/Rp_eq
	#	Rp0 = Dpp0*R47_stoch

	#extract values
	k1 = p[0]
	k_dif_single = p[1]*(R45_stoch - Rpeq)*(R46_stoch - Rpeq)/Rpeq
	Rp0 = p[2]*R47_stoch

	k = np.array([k1, k_dif_single, Rp0]) #combine into list

	#extract uncertainty
	pstd = np.sqrt(np.diag(pcov))
	k1_std = pstd[0]
	k_dif_single_std = pstd[1]*(R45_stoch - Rpeq)*(R46_stoch - Rpeq)/Rpeq
	Rp0_std = pstd[2]*R47_stoch

	k_std = np.array([k1_std, k_dif_single_std, Rp0_std]) #combine into list

	return k, k_std, rmse, npt

def _fit_HH20(he):

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

#function to fit complete Hea14 model
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

#function to fit SE15 model using backward Euler
def _SE15_fin_diff(t, k1f, k2f, Dpp0, Dppeq, Dp470, Dp47eq):
	'''
	Function for solving the Stolper and Eiler (2015) paired diffusion model
	using a backward Euler finite difference approach.

	Paramters
	---------
	t : array-like
		Array of time points, in minutes.

	k1f : float
		Forward k value for the [44] + [47] <-> [pair] equation (SE15 Eq. 8a).
		To be estimated using `curve_fit`.

	k2f : float
		Forward k value for the [pair] <-> [45]s + [46]s equation (SE15 Eq. 8b).
		To be estimated using `curve_fit`.

	Dpp0 : float
		Initial pair composition, written in 'prime' notation:

			Dpp = Rp/R47_stoch

		To be estimated using `curve_fit`.

	Dppeq : float
		Equilibrium pair composition, written in 'prime' notation (as above).
		Calculated using measured d18O and d13C values (SE15 Eq. 13 a/b).

	Dp470 : float
		Initial D47 value of the experiment, written in 'prime' notation:

			Dp47 = R47/R47_stoch = D47/1000 + 1

		Measured using mass spectrometry and inputted into function.

	Dp47eq : float
		Equilibrium D47 value for a given temperature, written in 'prime'
		notation (as above). Calculated using one of the T-D47 calibration
		curves.

	Returns
	-------
	Dp47 : np.ndarray
		Array of calculated Dp47 values at each time point. To be used for
		'curve_fit' solving. 

	References
	----------
	[1] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.

	Notes
	-----
	Because of the requirements for 'curve_fit', this funciton is only for
	solving the inverse problem for heating experiment data. Geological
	history forward-model solution is in a separate function.
	'''

	#make A matrix and B array

	#values for each entry
	a = k1f
	b = k1f*Dp47eq/Dppeq
	c = k2f
	d = k2f*Dppeq

	#combine into arrays
	A = np.array([[-a, b],
				  [a, -(b+c)]])

	B = np.array([0, d])

	#extract constants, pre-allocate arrays, and set initial conditions
	nt = len(t)
	x = np.zeros([nt, 2])
	x[0,:] = [Dp470, Dpp0]

	#loop through each time points and solve backward Euler problem
	for i in range(nt-1):

		#calculate inverted A
		Ai = inv(eye(2) - (t[i+1] - t[i])*A)

		#calculate x at next time step
		x[i+1,:] = np.dot(Ai, (x[i,:] + (t[i+1] - t[i])*B))

	#extract Dp47 for curve-fit purposes
	Dp47 = x[:,0]

	return Dp47

#function to calcualte stochastic R45, R46, and R47
def _calc_R_stoch(d13C, d18O, iso_params):
	'''
	Calculates stochastic R45, R46, and R47 distributions for a given set of
	d13C, d18O, and isotope parameters.

	Parameters
	----------
	d13C : float
		13C composition, in permil VPDB.

	d18O : float
		18O composition, in permil VPDB.

	iso_params : string
		String of the isotope parameters to use.

	Returns
	-------
	R45_stoch : float
		Stochastic R45 value.

	R46_stoch : float
		Stochastic R46 value.

	R47_stoch : float
		Stochastic R47 value.

	References
	----------
	[1] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.
	'''

	#extract iso_params
	R13_vpdb, R18_vpdb, R17_vpdb, lam17 = d47_isoparams[iso_params]

	#convert d13C and d18O into R13, R18
	R13 = (d13C/1000 + 1)*R13_vpdb
	R18 = (d18O/1000 + 1)*R18_vpdb

	#calculate R17 using R18 and lam17
	R17 = R17_vpdb * (R18 / R18_vpdb)**lam17

	#convert R values to fractional abundances
	f12 = 1/(1+R13)
	f13 = R13*f12

	f16 = 1/(1+R17+R18)
	f17 = R17*f16
	f18 = R18*f16

	#combine into stochastic distributions
	# [44] = [12][16][16]
	# [45] = [13][16][16] + 2*[12][17][16]
	# [46] = 2*[12][16][18] + 2*[13][17][16] + [12][17][17]
	# [47] = 2*[13][16][18] + 2*[12][17][18] + [13][17][17]

	f44 = f12*f16*f16
	f45 = f13*f16*f16 + 2*f12*f17*f16
	f46 = 2*f12*f16*f18 + 2*f13*f17*f16 + f12*f17*f17
	f47 = 2*f13*f16*f18 + 2*f12*f17*f18 + f13*f17*f17

	#convert to R values
	R45_stoch = f45/f44
	R46_stoch = f46/f44
	R47_stoch = f47/f44

	return R45_stoch, R46_stoch, R47_stoch

#function to calcualte equilibrium pair concentrations
def _calc_Rpeq(R45_stoch, R46_stoch, R47_stoch, z):
	'''
	Function to calculate the equilibrium pair concentration ratio.

	Parameters
	----------
	R45_stoch : float
		Stochastic R45 value.

	R46_stoch : float
		Stochastic R46 value.

	R47_stoch : float
		Stochastic R47 value.

	z : int
		The mineral coordination number; z = 6 according to SE15.

	Returns
	-------
	Rpeq : float
		The equilibrium pair concentration, normalied to [44].

	References
	----------
	[1] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.

	Notes
	-----
	This function uses the average of both methods for calcuating [p] (i.e.,
	Eqs. 13a and 13b in SE15). The difference between the two functions is ~1-2
	percent relative, so this choice is essentially arbitrary.
	'''

	#calcualte f values
	f44 = 1/(1 + R45_stoch + R46_stoch + R47_stoch)
	f45 = R45_stoch*f44
	f46 = R46_stoch*f44

	#calculate equilibrium pair concentration (SE15 Eq. 13a/b)
	# Note: Use the average of both calculations
	pa = f46*(1 - (1 - f45)**z)
	pb = f45*(1 - (1 - f46)**z)
	p = (pa+pb)/2

	#convert to ratio
	Rpeq = p/f44

	return Rpeq










	# D47_eq = _calc_Deq(he.T,
	# 	calibration = he.calibration,
	# 	clumps = he.clumps,
	# 	ref_frame = he.calibration
	# 	)

	# #extract R45 and R46 from the first row of he.d (assume constant in time?)
	# R45, R46, _ = calc_R(he.d[0,:],
	# 	clumps = he.clumps,
	# 	iso_params = he.iso_params,
	# 	sig_figs = 15
	# 	)

	# #calculate singleton concentrations (SE15 Eq. 12)
	# R45_s_eq = R45*((1 - R46)**z)
	# R46_s_eq = R46*((1 - R45)**z)

	# #calculate equilibrium pair concentration (SE15 Eq. 13a)
	# Rp_eq = R46 - R46_s_eq

	# #compile constants
	# cs = [D47_eq, Rp_eq, R45_s_eq, R46_s_eq]



# def _fSE15(x, k1, kdp, Rp0, D47_eq, Rp_eq, R45_s_eq, R46_s_eq):
# # 	'''
# # 	Add docstring
# # 	'''


# 	return D47







