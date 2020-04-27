'''
This module contains assorted fitting and calculation functions for all
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
__all__ = ['_calc_A',
		   '_calc_R',
		   '_calc_R_stoch',
		   '_calc_rmse',
		   '_calc_Rpeq',
		   # '_fexp_const',
		   '_fHea14',
		   '_fHH20',
		   # '_flin',
		   '_fPH12',
		   '_fSE15',
		   '_Gaussian',
		   '_Jacobian',
		  ]

#import packages
import numpy as np

#import linear algebra functions
from numpy import eye
from numpy.linalg import (
	inv,
	norm,
	)

#import necessary isoclump core functions
from .core_functions import(
	derivatize,
	)

#import necessary isoclump dictionaries
from .dictionaries import(
	d47_isoparams,
	)

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

	[1] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.\n
	[2] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
	'''

	#extract constants
	nt = len(t)
	nlam = len(lam)
	dlam = lam[1] - lam[0]

	#define matrices
	t_mat = np.outer(t, np.ones(nlam))
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
	# return np.sqrt(np.sum((y-yhat)**2)/len(y))
	return norm(y - yhat)/(len(y)**0.5)

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

	Notes
	-----

	This function uses the average of both methods for calcuating [p] (i.e.,
	Eqs. 13a and 13b in SE15). The difference between the two functions is ~1-2
	percent relative, so this choice is essentially arbitrary.

	References
	----------
	[1] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
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

#function to fit complete Hea14 model
def _fHea14(t, lnkc, lnkd, lnk2):
	'''
	Estimates G using the "transient defect/equilibrium" model of Henkes et
	al. (2014) (Eq. 5).

	Parameters
	----------

	t : array-like
		The array of time points.

	lnkc : float
		The latural log of the first-order rate constant.

	lnkd : float
		The natural log of the transient defect rate constant.

	lnk2 : float
		The natural log of the transient defect disappearance rate constant.

	Returns
	-------

	Ghat : array-like
		Resulting estimated G values.

	References
	----------

	[1] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
	'''

	#get into un-logged format
	kc = np.exp(lnkc)
	kd = np.exp(lnkd)
	k2 = np.exp(lnk2)

	#calculate lnGhat
	lnGhat = -kc*t + (kd/k2)*(np.exp(-k2*t) - 1)

	return np.exp(lnGhat)

#function to fit data to lognormal decay k distribution for HH20  model
def _fHH20(t, mu_lam, sig_lam, lam_max, lam_min, nlam):
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

#function to fit complete PH12 model
def _fPH12(t, lnk, intercept):
	'''
	Defines the pseudo-first order model of Passey and Henkes (2012).

	Parameters
	----------

	t : array-like
		The t values.

	lnk : float
		The natural log of the rate constant.

	intercept : float
		The pre-exponential factor; i.e., the intercept in t vs. G space.

	Returns
	-------

	Ghat : array-like
		Resulting estimated G values.

	References
	----------
	[1] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
	'''
	return intercept*np.exp(-t*np.exp(lnk))

#function to fit SE15 model using backward Euler
def _fSE15(t, lnk1f, lnkds, p0peq, D0, Deq, Dppeq, he):
	'''
	Function for solving the Stolper and Eiler (2015) paired diffusion model
	using a backward Euler finite difference approach.

	Paramters
	---------

	t : array-like
		Array of time points, in minutes.

	lnk1f : float
		Natural log of the forward k value for the [44] + [47] <-> [pair] 
		equation (SE15 Eq. 8a). To be estimated using ``curve_fit``.

	lnkds : float
		Natural log of the backward k value for the [pair] <-> [45]s + [46]s
		equation (SE15 Eq. 8b). To be estimated using ``curve_fit``.

	p0peq : float
		Ratio of initial pair composition relative to equilibrium pair
		composition. To be estimated using ``curve_fit``.

	D0 : float
		Initial D47 value of the experiment.

	Deq : float
		Equilibrium D47 value for a the experimental temperature.

	Dppeq : float
		Equilibrium pair composition, written in 'prime' notation. Calculated 
		using measured d18O and d13C values (SE15 Eq. 13 a/b).

	he : isoclump.HeatingExperiment
		The HeatingExperiment instance containing data to fit; to be used to
		convert kdp to kds for ``curve_fit``.

	Returns
	-------

	D : np.ndarray
		Array of calculated D values at each time point. To be used for
		``curve_fit`` solving. 

	Notes
	-----

	Because of the requirements for ``curve_fit``, this funciton is only for
	solving the inverse problem for heating experiment data. Geological
	history forward-model solution is in a separate function.

	References
	----------

	[1] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
	'''

	#first, need to get lnkds into kdp format (consistent with Stolper)
	# kdp = kds*(R45_s_eq * R46_s_eq)/Rpeq

	#calculate R45_stoch, R46_stoch, R47_stoch
	d13C = np.mean(he.dex[:,1]) #use average of all experimental points
	d18O = np.mean(he.dex[:,2]) #use average of all experimental points

	R45_stoch, R46_stoch, R47_stoch = _calc_R_stoch(d13C, d18O, he.iso_params)
	Rpeq = Dppeq*R47_stoch

	#exponentiate and convert to kdf
	kds = np.exp(lnkds)
	kdp = kds*((R45_stoch - Rpeq) * (R46_stoch - Rpeq))/Rpeq

	#get other unknowns into right format
	k1f = np.exp(lnk1f)
	Dpp0 = p0peq*Dppeq

	#get constants into the right format
	Dp470 = D0/1000 + 1
	Dp47eq = Deq/1000 + 1

	#make A matrix and B array

	#values for each entry
	a = k1f
	b = k1f*Dp47eq/Dppeq
	c = kdp
	d = kdp*Dppeq

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

	#convert back to D
	D = 1000*(Dp47 - 1)

	return D

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

#function for estimating Jacobian matrices for error propagation
def _Jacobian(f, t, p, eps = 1e-6):
	'''
	Estimates the Jacobian matrix of derivatives by perturbing each paramter
	in p by some small amount eps.

	Parameters
	----------
	f : function
		Function to calculate the Jacobian on; e.g., the model function for each
		model type.

	t : np.array
		Array of time points. Length ``nt``.

	p : array-like
		An array containing all the parameters that are inputted to f. 
		Length ``np``.

	eps : float
		The amount to perturb each parameter by. Defaults to ``1e-10``.

	Returns
	-------
	J : np.array
		A 2d array containing the Jacobian matrix. Shape [``nt`` x ``np``].
	'''

	#extract constants and pre-allocate array
	npar = len(p)
	nt = len(t)
	J = np.zeros([nt, npar], dtype = np.float)

	#loop through each parameter and estimate derivative when perturbed
	for i in range(npar):

		#parameter values plus perturbation
		pp = p.copy()
		pp[i] += eps

		#parameter values minus perturbation
		pm = p.copy()
		pm[i] -= eps

		#store in J matrix
		J[:,i] = (f(t, *pp) - f(t, *pm)) / (2*eps)

	return J












#function for calculating HH20 model Jacobian; used for propagating error
# def _Jacobian_HH20(t, mu_lam, sig_lam, lam_max, lam_min, nlam):
# 	'''
# 	Function to calculate dG/dln(k_mu) and dG/dln(k_sig) and combine into
# 	Jacobian matrix to propagate uncertainty when forward-modeling HH20
# 	results.

# 	Parameters
# 	----------

# 	t : array-like
# 		Array of time, in seconds; of length `n_t`.

# 	mu_lam : scalar
# 		Mean of lam, the lognormal rate distribution.
		
# 	sig_lam : scalar
# 		Standard deviation of lam, the lognormal rate distribution.

# 	lam_max : scalar
# 		Maximum lambda value for distribution range; should be at least 4 sigma
# 		above the mean. 

# 	lam_min : scalar
# 		Minimum lambda value for distribution range; should be at least 4 sigma
# 		below the mean.
		
# 	nlam : int
# 		Number of nodes in lam array.

# 	Returns
# 	-------

# 	J : np.array
# 		Resulting Jacobian matrix, of shape [``nt`` x 2]
# 	'''

# 	#setup arrays
# 	nt = len(t)
# 	lam = np.linspace(lam_min, lam_max, nlam)
# 	dlam = lam[1] - lam[0]
# 	rho = _Gaussian(lam, mu_lam, sig_lam)

# 	#derivative multiplier arrays
# 	m1 = (lam - mu_lam) / (sig_lam**2)
# 	m2 = (lam - mu_lam) / (sig_lam**3) - 1/sig_lam

# 	#make matrices
# 	t_mat = np.outer(t, np.ones(nlam))
# 	lam_mat = np.outer(np.ones(nt), lam)
# 	rho_mat = np.outer(np.ones(nt), rho)

# 	#derivative multiplier matrices
# 	m1_mat = np.outer(np.ones(nt), m1)
# 	m2_mat = np.outer(np.ones(nt), m2)

# 	#solve
# 	x = rho_mat * np.exp(- np.exp(lam_mat) * t_mat) * dlam
	
# 	xm1 = x*m1_mat
# 	Gpp0 = np.inner(xm1, np.ones(nlam))

# 	xm2 = x*m2_mat
# 	Gpp1 = np.inner(xm2, np.ones(nlam))

# 	#combine into jacobian matrix
# 	J = np.column_stack((Gpp0, Gpp1))

# 	return J

# #exponential function for curve fitting
# def _fexp_const(x, c0, c1):
# 	'''
# 	Defines an exponential decay with a constant. used for Hea14 model fit.

# 	Parameters
# 	----------

# 	x : array-like
# 		The x values.

# 	c0 : float
# 		The exponential value; e.g., the rate constant.

# 	c1 : float
# 		The pre-exponential factor.

# 	Returns
# 	-------

# 	yhat : array-like
# 		Resulting array of y values.
# 	'''
# 	return (np.exp(x*c0) - 1)*c1

# #linear function for curve fitting
# def _flin(x, c0, c1):
# 	'''
# 	Defines a straight line.

# 	Parameters
# 	----------

# 	x : array-like
# 		The x values.

# 	c0 : float
# 		The slope.

# 	c1 : float
# 		The intercept.

# 	Returns
# 	-------

# 	yhat : array-like
# 		Resulting array of y values.
# 	'''

# 	return c0*x + c1

