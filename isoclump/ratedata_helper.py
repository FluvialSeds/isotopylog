'''
This module contains helper functions for the RateData classes.
'''

#import from future for python 2
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

#import packages
import numpy as np
import pandas as pd

from numpy.linalg import inv
from numpy import eye
from scipy.optimize import (
	curve_fit,
	nnls,
	)

#import necessary isoclump dictionaries and functions
from .dictionaries import(
	caleqs,
	d47_isoparams,
	)

from .core_functions import(
	derivatize,
	)

from .timedata_helper import(
	_calc_D_from_G,
	)

# TODO:
# * FINISH WRITING L-CURVE FUNCTION
# * CHANGE CALL STRUCTURE TO INCLUDE **KWARGS
# * MOVE FUNCTIONS INTO MORE APPROPRIATE MODULES; RENAME ACCORDINGLY
# * 



#function to fit data using PH12 model
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
		Root Mean Square Error uncertainty (in D47 units) of the model fit.
		Only includes data points that are deemed to be in the linear region.

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
	Results are bounded such that k is non-negative; reported intercept is the
	negative of the intercept in lnG vs. t space. All calculations are done in
	lnG space and thus only depend on relative changes in D47.
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
		absolute_sigma = True,
		bounds = ([-np.inf, -np.inf],[0,0]), #-k and -intercept < 0
		)
	
	#extract variables to export
	k = -p #get rate and intercept to be positive
	k_std = np.diag(pcov)**0.5
	npt = len(xl)

	#calculate Gex_hat
	Gex_hat = _fexp(xl, p[0], np.exp(p[1]))

	#convert to D47
	D47_hat, _ = _calc_D_from_G(
		he.calibration, 
		he.clumps, 
		he.dex[0,0], 
		Gex_hat, 
		0, #just pass Gex_hat_std = 0 since we won't use it 
		he.ref_frame, 
		he.T)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[i0:,0], D47_hat)

	return k, k_std, rmse, npt
	
#function to fit data using Hea14 model
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
	
	Notes
	-----
	Results are bounded to be non-negative. All calculations are done in lnG
	space and thus only depend on relative changes in D47.
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
		absolute_sigma = True,
		bounds = ([-np.inf, 0],[0, np.inf]), #-k < 0; f.o. intercept > 0
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

	#calculate Gex_hat
	Gex_hat = _fHea14(x, *k)

	#convert to D47
	D47_hat, _ = _calc_D_from_G(
		he.calibration, 
		he.clumps, 
		he.dex[0,0], 
		Gex_hat, 
		0, #just pass Gex_hat_std = 0 since we won't use it 
		he.ref_frame, 
		he.T)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47_hat)

	return k, k_std, rmse, nptfo

#function to fit data using SE15 model
def _fit_SE15(he, k0, z):
	'''
	Fits D evolution data using the paired diffusion model of Stolper and
	Eiler (2015). The function solves for both k1 and k_dif_single as well
	as the initial pair concentration, p0/peq, by solving a modified version
	of SE15 Eq. 9-10. Note that p0/peq can be estimated from SE15 Eq. 17.

	Paramters
	---------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	k0 : array-like
		Initial guess at k parameters, in the order [k1, k_dif_single, p0/peq].
		This is taken as an input in order to allow the user to adjust these
		values to exactly match SE15 results.

	z : float
		The mineral lattice coordination number to use for calculating the
		concentration of pairs. According to Stolper and Eiler (2015), this
		should default to `6`.

	Returns
	-------
	k : np.ndarray
		Array of resulting k values, in the order [k1, k_dif_single, p0/peq].

	k_std : np.ndarray
		Uncertainty associated with resulting k values.

	rmse : float
		Root Mean Square Error uncertainty (in D47 units) of the model fit.
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
	different choices of k0 when solving.
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

	#get inputted k0 into the right format
	#k1f
	p00 = k0[0]

	#k2f
	p01 = k0[1]*(R45_stoch - Rpeq)*(R46_stoch - Rpeq)/Rpeq

	#Dpp0
	p02 = k0[2]*Dppeq

	p0 = [p00, p01, p02]

	#solve
	p, pcov = curve_fit(lamfunc, x, yp, p0,
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

	k = np.array([k1, k_dif_single, p0peq]) #combine into list

	#extract uncertainty
	pstd = np.sqrt(np.diag(pcov))
	k1_std = pstd[0]
	k_dif_single_std = pstd[1]*(R45_stoch - Rpeq)*(R46_stoch - Rpeq)/Rpeq
	p0peq_std = pstd[2]/Dppeq

	k_std = np.array([k1_std, k_dif_single_std, p0peq_std]) #combine into list

	return k, k_std, rmse, npt

#function to fit data using the HH20 inverse model
def _fit_HH20inv(he, lam_max, lam_min, nlam, omega, **kwargs):
	'''
	Fits D evolution data using the distributed activation energy model of
	Hemingway and Henkes (2020). This function solves for rho_lam, the
	regularized distribution of rates in lnk space. See HH20 Eq. X for
	notation and details. This function can estimate best-fit omega using
	Tikhonov regularization.
	
	Parameters
	----------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	lam_max : float
		The maximum lnk value to consider.

	lam_min : float
		The minimum lnk value to consider.

	nlam : int
		The number of lam values in the array such that
		dlam = (lam_max - lam_min)/nlam.

	omega : str or float
		The "smoothing parameter" to use. This can be a number or 'auto'; if 'auto',
		the function uses Tikhonov regularization to calculate the optimal omega
		value.
	
	Returns
	-------
	rho_lam : array-like
		Resulting regularized rho distribution, of length `n_lam`.
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
		omega = _calc_L_curve(
			tex, 
			Gex, 
			lam_max = lam_max,
			lam_min = lam_min,
			nlam = nlam,
			plot = False,
			**kwargs
			)

	#make sure omega is a scalar
	elif type(omega) not in [float, int]:
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
	rho, _ = nnls(A_reg, Gex_reg)
	Gex_hat = np.inner(A, rho)
	rgh = np.inner(R, rho)

	#calculate errors
	resid = norm(Gex - Gex_hat)/nt**0.5
	rgh = norm(rgh)/nlam**0.5

	return rho_lam, resid, rgh

#function to fit data using HH20 lognormal model
def _fit_HH20(he, lam_max, lam_min, nlam):
	'''
	Fits D evolution data using the distributed activation energy model of
	Hemingway and Henkes (2020). This function solves for mu_lam and sig_lam,
	the mean and standard deviation of a Gaussian distribution in lnk space.
	See HH20 Eq. X for notation and details.

	Paramters
	---------
	he : ic.HeatingExperiment
		HeatingExperiment instance containing the D data to be modeled.

	lam_max : float
		The maximum lnk value to consider.

	lam_min : float
		The minimum lnk value to consider.

	nlam : int
		The number of lam values in the array such that
		dlam = (lam_max - lam_min)/nlam.

	Returns
	-------
	k : np.ndarray
		Array of resulting k values, in the order [mu_lam, sig_lam].

	k_std : np.ndarray
		Uncertainty associated with resulting k values.

	rmse : float
		Root Mean Square Error uncertainty (in D47 units) of the model fit.
		Includes model fit to all data points.

	npt : int
		Number of data points included in the model solution.

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
	# lam = np.linspace(lam_min, lam_max, nlam)

	#fit model to lambda function to allow inputting constants
	lamfunc = lambda x, mu_lam, sig_lam: _lognormal_decay(
		x, 
		mu_lam, 
		sig_lam, 
		lam_max,
		lam_min,
		nlam
		)

	#make initial guess
	sig0 = (lam_max-lam_min)/4 #quarter of the inputted width
	mu0 = lam_min + 2*sig0 #middle of the inputted range
	p0 = [mu0, sig0]

	#solve
	p, pcov = curve_fit(lamfunc, x, y, p0,
		sigma = y_std, 
		absolute_sigma = True,
		bounds = ([lam_min, 0.],[lam_max, sig0]), #mu, sig must be in range
		)

	#extract variables to export
	k = p
	k_std = np.sqrt(np.diag(pcov))
	npt = len(x)

	#calculate Gex_hat
	Gex_hat = lamfunc(x, *p)

	#convert to D47
	D47_hat, _ = _calc_D_from_G(
		he.calibration, 
		he.clumps, 
		he.dex[0,0], 
		Gex_hat, 
		0, #just pass Gex_hat_std = 0 since we won't use it 
		he.ref_frame, 
		he.T)

	#calcualte RMSE
	rmse = _calc_rmse(he.dex[:,0], D47_hat)

	return k, k_std, rmse, npt

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

#function to predict decay given an inputted lognormal k distribution
def _lognormal_decay(t, mu_lam, sig_lam, lam_max, lam_min, nlam):
	'''
	Function to calculate G as a function of time assuming a lognormal 
	distribution of decay rates described by mu and sigma.

	Parameters
	----------
	t : array-like
		Array of time, in seconds; of length `n_t`.

	mu_lam : scalar
		Mean of lam, the lognormal rate distribution.
		
	sig_lam : scalar
		Standard deviation of lam, the lognormal rate distribution.

	lam_max : scalar
		Maximum lambda value for distribution range; should be at least 4 sigma
		above the mean. 

	lam_min : scalar
		Minimum lambda value for distribution range; should be at least 4 sigma
		below the mean.
		
	nlam : int
		Number of nodes in lam array.

	Returns
	-------
	G : array-like
		Array of resulting G values at each time point.
	'''

	#setup arrays
	nt = len(t)
	lam = np.linspace(lam_min, lam_max, nlam)
	dlam = lam[1] - lam[0]
	rho = _Gaussian(lam, mu_lam, sig_lam)

	#make matrices
	t_mat = np.outer(t, np.ones(nlam))
	lam_mat = np.outer(np.ones(nt), lam)
	rho_mat = np.outer(np.ones(nt), rho)

	#solve
	x = rho_mat * np.exp(- np.exp(lam_mat) * t_mat) * dlam
	G = np.inner(x, np.ones(nlam))

	return G

#function for a Gaussian distribution
def _Gaussian(x, mu, sigma):
	'''
	Function to make a Gaussian (normal) distribution.
	
	Parameters
	----------
	x : scalar or array-like
		Input x value(s).
	
	mu : scalar
		Gaussian mean.
		
	sigma : scalar
		Gaussian standard deviation.
	
	Returns
	-------
	y : scalar or array-like
		Output y value(s).
	'''
	
	s = 1/(2*np.pi*sigma**2)**0.5
	y = s * np.exp(-(x - mu)**2/(2*sigma**2))

	return y

#define function for calculating HH20 inverse A matrix
def _calc_A(t, lam):
	'''
	Function for calculating A matrix for HH20 data inversion.

	Parameters
	----------
	t : array-like
		Array of time points, of length `nt`.

	lam : array-like
		Array of lambda points, of lnegth `nlam`.

	Returns
	-------
	A : np.ndarray
		2-D array A matrix, of shape [`n_t` x `n_lam`]

	References
	----------
	[1] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.
	[2] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
	'''

	#extract constants
	nt = len(t)
	nlam = len(lam)
	dlam = lam[1] - lam[0]

	#define matrices
	tmat = np.outer(t, np.ones(nlam))
	lam_mat = np.outer(np.ones(nt), lam)

	A = np.exp(- np.exp(lam_mat) * t_mat) * dlam
	
	return A

#define function for calculating HH20 inverse R matrix
def _calc_R(n):
	'''
	Calculates smoothing matrix, R.

	Parameters
	----------
	n : int
		The number of points.

	Returns
	-------
	R : np.ndarray
		2-D array Tikhonov regularization matrix, of shape [`n+1` x `n`]

	References
	----------
	[1] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.
	'''

	#pre-allocate matrix
	R = np.zeros([n + 1, n])

	#ensure pdf = 0 outside of E range specified
	R[0, 0] = 1.0
	R[-1, -1] = -1.0

	#1st derivative operator
	c = [-1, 1]

	#populate matrix
	for i, row in enumerate(R):
		if i != 0 and i != n:
			row[i - 1:i + 1] = c

	return R

#FINISH UPDATING THIS FUNCTION!
#define function for calculating best-fit omega using L-curve approach
def _calc_L_curve(
	tex,
	Gex,
	omega_max = 1e3, 
	omega_min = 1e-3,
	nom = 150,
	lam_max = 10, 
	lam_min = -50, 
	nlam = 300,
	kink = 1,
	ax = None,
	plot = False
	):
	'''
	Function to choose the "best" omega value for regularization following
	the Tikhonov Regularization method. The best-fit omega is chosen as the
	value at the point of maximum curvature in a plot of log residual error
	vs. log roughness.
	
	Parameters
	----------
	t_e : array-like
		Array of experimental time points, in seconds; of length `n_t`.
	
	alpha_e : array-like
		Array of measured alpha values; of length `n_lam`.
	
	omega_max : float
		Maximum omega value to consider, defaults to `1e3`.

	omega_min : float
		Minimum omega value to consider, defaults to `1e-3`.
		
	nom : int
		Number of nodes on omega array.

	lam_max : scalar
		Maximum lambda value for distribution range; should be at least 4 sigma
		above the mean; defaults to `30`.

	lam_min : scalar
		Minimum lambda value for distribution range; should be at least 4 sigma
		below the mean; defaults to `-30`.
		
	nlam : int
		Number of nodes in lambda array.
	
	kink : int
		Tells the funciton which L-curve "kink" to use; this is a required
		input since many L-curve solutions appear to have 2 "kinks"; input `1`
		for the lower kink and `2` for the upper kink. Without fail, the lower
		kink appears to be a significantly more robust fit.
	
	ax : `None` or plt.axis
		Matplotlib axis to plot on, only relevant if `plot = True`.
	
	plot : Boolean
		Boolean telling the funciton whether or not to plot L-curve results.
	
	Returns
	-------
	om_best : np.ndarray
		Array of best-fit omega values, of length `n_peaks`.
	
	ax : plt.axis
		Updated Matplotlib axis containing L-curve plot.
	'''
	
	#define arrays
	log_om_vec = np.linspace(np.log10(omega_min), np.log10(omega_max), nom)
	om_vec = 10**log_om_vec

	res_vec = np.zeros(nom)
	rgh_vec = np.zeros(nom)

	#for each omega value in the vector, calculate the errors
	for i, w in enumerate(om_vec):

		#THIS NEEDS TO BE UPDATED TO CONFORM TO FUNCTION INPUTS!!!!
		_, resid, rgh = _fit_HH20inv(t_e,
			alpha_e,
			w,
			lam_max = lam_max,
			lam_min = lam_min,
			nlam = nlam
			)

		# _fit_HH20inv(he, lam_max, lam_min, nlam, omega,

		res_vec[i] = res
		rgh_vec[i] = rgh

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
	
	#make any infs and nans into zeros
	k[np.abs(k) == np.inf] = 0
	k[k == np.nan] = 0

	#extract peak indices
	pkinds = argrelmax(k)[0]
	pki = np.argsort(k[pkinds])[::-1][:n_peaks]
	i = pkinds[pki]

	#extract om_best
	om_best = om_vec[i]

	#plot if necessary
	if plot:

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
			label = r'best-fit $\lambda$')

		#set axis labels and text

		xlab = r'residual error, $\log_{10} \left( \frac{\|\|' \
			r'\mathbf{A}\cdot \mathbf{p} - \mathbf{g} \|\|}{\sqrt{n_{j}}}' \
			r'\right)$'
		ax.set_xlabel(xlab)

		ylab = r'roughness, $\log_{10} \left( \frac{\|\| \mathbf{R}' \
			r'\cdot\mathbf{p} \|\|}{\sqrt{n_{l}}} \right)$'

		ax.set_ylabel(ylab)

		if n_peaks == 1:
			label1 = r'best-fit $\omega$ = %.3f' %(om_best)

			label2 = (
				r'$log_{10}$ (resid. err.) = %.3f' %(res_vec[i]))

			label3 = (
				r'$log_{10}$ (roughness)  = %0.3f' %(rgh_vec[i]))
		
		else:
			label1 = r'best-fit $\omega$ = %.3f, %.3f' %(om_best[0], om_best[1])

			label2 = (
				r'$log_{10}$ (resid. err.) = %.3f, %.3f' %(res_vec[i[0]], res_vec[i[1]]))

			label3 = (
				r'$log_{10}$ (roughness)  = %0.3f, %.3f' %(rgh_vec[i[0]], rgh_vec[i[1]]))

		ax.text(
			0.3,
			0.95,
			label1 + '\n' + label2 + '\n' + label3,
			verticalalignment='top',
			horizontalalignment='left',
			transform=ax.transAxes)

		return om_best, ax

	else:
		return om_best






	# return om_best, ax
