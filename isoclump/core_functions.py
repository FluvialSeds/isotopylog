'''
Module to store all the core functions for ``isoclump``.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = [
			# 'Arrhenius_plot',
			# 'assert_len',
			# 'change_ref_frame',
			# 'calc_cooling_rate',
			# 'calc_d',
			# 'calc_R',
			# 'calc_Teq',
			# 'calc_Deq',
			'derivatize',
			# 'resetting_plot',
			# 'cooling_plot',
			]

import matplotlib.pyplot as plt
import numpy as np

# from collections import Sequence

# #import exceptions
# from .exceptions import(
# 	ArrayError,
# 	LengthError,
# 	)

#import timedata helper functions
# from .timedata_helper import(
# 	_assert_clumps,
# 	_assert_ref_frame,
# 	_assert_iso_params,
# 	)

#import dictionaries with conversion information
from .dictionaries import(
	caleqs,
	d47_isoparams,
	)

# #define funtction for making Arrhenius plots
# def Arrhenius_plot(rate_list, ax = None, xaxis = 'Tinv', yaxis = 'mu'):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to assert length of array
# def assert_len(data, n):
# 	'''
# 	Asserts that an array has length `n` and `float` datatypes.

# 	Parameters
# 	----------
# 	data : scalar or array-like
# 		Array to assert has length n. If scalar, generates an np.ndarray
# 		with length n.

# 	n : int
# 		Length to assert

# 	Returns
# 	-------
# 	array : np.ndarray
# 		Updated array, now of class np.ndarray and with length n.

# 	Raises
# 	------
# 	ArrayError
# 		If inputted data not int or array-like (excluding string).

# 	LengthError
# 		If length of the array is not n.
# 	'''

# 	#assert that n is int
# 	n = int(n)

# 	#assert data is in the right form
# 	if isinstance(data, (int, float)):
# 		data = data*np.ones(n)
	
# 	elif isinstance(data, Sequence) or hasattr(data, '__array__'):
		
# 		if isinstance(data, str):
# 			raise ArrayError(
# 				'Data cannot be a string')

# 		elif len(data) != n:
# 			raise LengthError(
# 				'Cannot create array of length %r if n = %r' \
# 				% (len(data), n))

# 	else:
# 		raise ArrayError('data must be scalar or array-like')

# 	return np.array(data).astype(float)

# #define function to change reference frame
# def change_ref_frame(ds, ref_frame, clumps = 'CO47'):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to calculate estimated cooling rate
# def calc_cooling_rate(ds, EDistribution):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to calculate d values from fractional abundances
# def calc_d(R, clumps = 'CO47', iso_params = 'Brand', sig_figs = 3):
# 	'''
# 	Calculates the delta values for a sample with a given set of ratios.

# 	Paramters
# 	---------
# 	R : array-like
# 		Array of isotopologue ratios, written from lowest to highest a.m.u.
# 		(e.g., d45, d46, d47 for CO47). All ratios are assumed to be relative
# 		to their commonly used standards:

# 			C : Vienna PeeDee Belemnite
# 			O : Vienna PeeDee Belemnite (for carbonates; solid CaCO3 not CO2!)
		
# 		Note, for 'CO47', R17 is assumed to be mass-dependent. Shape `n` x 3.

# 	clumps : string
# 		The clumped isotope system under consideration. Currently only accepts
# 		'CO47' for D47 clumped isotopes, but will include other isotope systems
# 		as they become more widely used and data become available. Defaults to
# 		'CO47'.

# 	iso_params: string
# 		The isotope parameters to use for calculations. If `clumps` is 'CO47',
# 		possible isotope parameter options are:

# 			'Brand' : Brand et al. (2010)
# 			'Gonfiantini' : Gonfiantini et al. (1995)
# 			'Craig + Assonov' : Craig (1957), Assonov and Brennenkmeijer (2003)
# 			'Chang + Li' : Chang and Li (1990), Li et al. (1988)
# 			'Craig + Li' : Craig (1957), Li et al. (1988)
# 			'Barkan' : Chang and Li (1990), Barkan et al. (2015)
# 			'Passey' : Chang and Li (1990), Passey et al. (2014)

# 		See discussion in Daëron et al. (2016) for further details. Defaults to
# 		'Brand'.

# 	sig_figs : int
# 		The number of significant figures to retain for returned d values.
# 		Defaults to '3'.

# 	Returns
# 	-------
# 	d : np.ndarray
# 		Array of resulting isotope values, written as [D, d1, d2] where D is the
# 		clumped isotope measurement (e.g., D47) and d1 and d2 are the 
# 		corresponding major isotope values, listed from lowest to highest a.m.u.
# 		(e.g., d13C, d18O). 

# 	Raises
# 	------
# 	TypeError
# 		If either 'clumps' or 'iso_params' is not a string.

# 	StringError
# 		If either 'clumps' or 'iso_params' is not an acceptable string.

# 	Notes
# 	-----
# 	This calculation assumes that major isotope compositions conform to the
# 	"stochastic" definition of Santrock et al. (1985) and elaborated further
# 	in Daëron et al. (2016). If this is not true, then resulting R values
# 	will be spurrious!

# 	References
# 	----------
# 	[1] Craig (1957) *Geochim. Cosmochim. Ac.*, **12**, 133--149.
# 	[2] Santrock et al. (1985) *Anal. Chem.*, **57**, 7444--7448.
# 	[3] Li et al. (1988) *Chin. Sci. Bull.*, **33**, 1610--1613.
# 	[3] Chang and Li (1990) *Chin. Sci. Bull.*, **35**, 290.
# 	[4] Gonfiantini et al. (1995) *IAEA Technical Document*.
# 	[5] Assonov and Brennenkmeijer (2003) *Rapid. Comm. Mass Spec.*, 
# 		**17**, 1017--1029.
# 	[6] Brand et al. (2010) *Pure Appl. Chem.*, **82**, 1719--1733.
# 	[7] Passey et al. (2014) *Geochim. Cosmochim. Ac.*, **141**, 1--25.
# 	[8] Barkan et al. (2015) *Rapid Comm. Mass. Spec.*, **29**, 2219--2224.
# 	[9] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.
# 	'''

# 	#check clumps are right
# 	clumps = _assert_clumps(clumps)

# 	#check iso_params are right
# 	iso_params = _assert_iso_params(clumps, iso_params)

# 	#the following steps are specific to each clumped isotope system:
# 	if clumps == 'CO47':

# 		#extract iso params from dictionary
# 		R13vpdb, R18vpdb, R17vpdb, lam17 = d47_isoparams[iso_params]

# 		#extract R values from inputted R array
# 		R45, R46, R47 = R

# 		#do some math!
# 		#calculate K, A-D from Daëron et al. (2016) appendix A Taylor expansion
# 		K = R17vpdb*(R18vpdb**-lam17)
# 		A = -3*(K**2)*(R18vpdb**(2*lam17))
# 		B = 2*K*R45*(R18vpdb**lam17)
# 		C = 2*R18vpdb
# 		D = -R46

# 		#calculate a-c
# 		a = A*lam17*(2*lam17 - 1) + B*lam17*(lam17-1)/2
# 		b = 2*A*lam17 + B*lam17 + C
# 		c = A + B + C + D

# 		#calculate R18, R17, and R13
# 		x = (-b + (b**2 - 4*a*c)**0.5)/(2*a)

# 		R18 = (1+x)*R18vpdb
# 		R17 = K*(R18**lam17)
# 		R13 = R45 - 2*R17

# 		#calculate d13C and d18O
# 		d13C = (R13/R13vpdb - 1)*1000
# 		d18O = (R18/R18vpdb - 1)*1000

# 		#calculate R47* and D47

# 		#calculate R47 from D47 and R47stoch
# 		# [47]* = [13][17][17] + 2*[13][16][18] + 2*[12][17][18]
# 		# [44]* = [12][16][16]
# 		# R47* = R13*R17**2 + 2*R13*R18 + 2*R17*R18
# 		R47stoch = R13*R17*R17 + 2*R13*R18 + 2*R17*R18
# 		D47 = (R47/R47stoch - 1)*1000
		
# 		d = [D47, d13C, d18O]

# 	return np.around(d, sig_figs)

# #define function to calculate fractional abundances from d values
# def calc_R(d, clumps = 'CO47', iso_params = 'Brand', sig_figs = 15):
# 	'''
# 	Calculates the isotopologue ratios for a sample with a given delta
# 	composition.

# 	Paramters
# 	---------
# 	d : array-like
# 		Array of isotope values, written as [D, d1, d2] where D is the clumped
# 		isotope measurement (e.g., D47) and d1 and d2 are the corresponding 
# 		major isotope values, listed from lowest to highest a.m.u. (e.g., d13C,
# 		d18O). All isotope values are assumed to be relative to their commonly 
# 		used standards:

# 			C : Vienna PeeDee Belemnite
# 			O : Vienna PeeDee Belemnite (for carbonates; solid CaCO3 not CO2!)
		
# 		Note, for 'CO47', d17O is assumed to be mass-dependent. Shape
# 		`n` x 3.

# 	clumps : string
# 		The clumped isotope system under consideration. Currently only accepts
# 		'CO47' for D47 clumped isotopes, but will include other isotope systems
# 		as they become more widely used and data become available. Defaults to
# 		'CO47'.

# 	iso_params: string
# 		The isotope parameters to use for calculations. If `clumps` is 'CO47',
# 		possible isotope parameter options are:

# 			'Brand' : Brand et al. (2010)
# 			'Gonfiantini' : Gonfiantini et al. (1995)
# 			'Craig + Assonov' : Craig (1957), Assonov and Brennenkmeijer (2003)
# 			'Chang + Li' : Chang and Li (1990), Li et al. (1988)
# 			'Craig + Li' : Craig (1957), Li et al. (1988)
# 			'Barkan' : Chang and Li (1990), Barkan et al. (2015)
# 			'Passey' : Chang and Li (1990), Passey et al. (2014)

# 		See discussion in Daëron et al. (2016) for further details. Defaults to
# 		'Brand'.

# 	sig_figs : int
# 		The number of significant figures to retain for returned R values.
# 		Defaults to '15'.

# 	Returns
# 	-------
# 	R : np.ndarray
# 		Array of resulting R values, in order of lowest to highest mass (e.g.,
# 		R45, R46, R47 for 'CO47' clumps).

# 	Raises
# 	------
# 	TypeError
# 		If either 'clumps' or 'iso_params' is not a string.

# 	StringError
# 		If either 'clumps' or 'iso_params' is not an acceptable string.

# 	Notes
# 	-----
# 	This calculation assumes that major isotope compositions conform to the
# 	"stochastic" definition of Santrock et al. (1985) and elaborated further
# 	in Daëron et al. (2016). If this is not true, then resulting R values
# 	will be spurrious!

# 	References
# 	----------
# 	[1] Craig (1957) *Geochim. Cosmochim. Ac.*, **12**, 133--149.
# 	[2] Santrock et al. (1985) *Anal. Chem.*, **57**, 7444--7448.
# 	[3] Li et al. (1988) *Chin. Sci. Bull.*, **33**, 1610--1613.
# 	[3] Chang and Li (1990) *Chin. Sci. Bull.*, **35**, 290.
# 	[4] Gonfiantini et al. (1995) *IAEA Technical Document*.
# 	[5] Assonov and Brennenkmeijer (2003) *Rapid. Comm. Mass Spec.*, 
# 		**17**, 1017--1029.
# 	[6] Brand et al. (2010) *Pure Appl. Chem.*, **82**, 1719--1733.
# 	[7] Passey et al. (2014) *Geochim. Cosmochim. Ac.*, **141**, 1--25.
# 	[8] Barkan et al. (2015) *Rapid Comm. Mass. Spec.*, **29**, 2219--2224.
# 	[9] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.
# 	'''

# 	#check clumps are right
# 	clumps = _assert_clumps(clumps)

# 	#check iso_params are right
# 	iso_params = _assert_iso_params(clumps, iso_params)

# 	#the following steps are specific to each clumped isotope system:
# 	if clumps == 'CO47':

# 		#extract iso params from dictionary
# 		R13vpdb, R18vpdb, R17vpdb, lam17 = d47_isoparams[iso_params]

# 		#extract delta values from inputted d array
# 		D47, d13Cvpdb, d18Ovpdb = d

# 		#do some math!
# 		#calculate R13, R18 from d13C, d18O
# 		R13 = (d13Cvpdb/1000 + 1)*R13vpdb
# 		R18 = (d18Ovpdb/1000 + 1)*R18vpdb

# 		#calculate K, R17 from d18O, lambda
# 		K = R17vpdb*(R18vpdb**-lam17)
# 		R17 = K*(R18**lam17)

# 		#calculate R45, R46 from R13, R17, R18 assuming stochastic (Daëron Eq. 4)
# 		R45 = R13 + 2*R17
# 		R46 = -3*(K**2)*(R18**(2*lam17)) + 2*K*R45*(R18**lam17) + 2*R18

# 		#calculate R47 from D47 and R47stoch
# 		# [47]* = [13][17][17] + 2*[13][16][18] + 2*[12][17][18]
# 		# [44]* = [12][16][16]
# 		# R47* = R13*R17**2 + 2*R13*R18 + 2*R17*R18

# 		R47stoch = R13*R17*R17 + 2*R13*R18 + 2*R17*R18
# 		R47 = (D47/1000 + 1)*R47stoch

# 		R = [R45, R46, R47]

# 	return np.around(R, sig_figs)

# #define function calculate Tikhonov regularization "L-curve"
# def calc_L_curve(HeatingExperiment, kmin = 1e-50, kmax = 1e20, nk = 300):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to calculate equilibrium T given D
# def calc_Teq(ds, calibration = 'PH12', clumps = 'CO47', ref_frame = 'CDES90'):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to calculate equilibrium D given T
# def calc_Deq(T, calibration = 'PH12', clumps = 'CO47', ref_frame = 'CDES90'):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to generate resetting plot a la Henkes et al. (2014)
# def resetting_plot(EDistribution):
# 	'''
# 	ADD DOCSTRING
# 	'''

# #define function to generate cooling plot a la Henkes et al. (2014)
# def cooling_plot(EDistribution):
# 	'''
# 	ADD DOCSTRING
# 	'''


#define function to derivatize an array w.r.t. another array
def derivatize(num, denom):
    '''
    Method for derivatizing numerator, `num`, with respect to denominator, 
    `denom`.

    Parameters
    ----------

    num : int or array-like
        The numerator of the numerical derivative function.

    denom : array-like
        The denominator of the numerical derivative function. Length `n`.

    Returns
    -------

    derivative : np.array
        An `np.array` instance of the derivative. Length `n`.

    Raises
    ------

    ArrayError
        If `denom` is not array-like.

    See Also
    --------

    numpy.gradient
        The method used to calculate derivatives

    Notes
    -----

    This method uses the ``np.gradient`` method to calculate derivatives. If
    `denom` is a scalar, resulting array will be all ``np.inf``. If both `num`
    and `denom` are scalars, resulting array will be all ``np.nan``. If 
    either `num` or `denom` are 1d and the other is 2d, derivative will be
    calculated column-wise. If both are 2d, each column will be derivatized 
    separately.
    '''

    #calculate separately for each dimensionality case
    if num.ndim == denom.ndim == 1:
        dndd = np.gradient(num)/np.gradient(denom)

    elif num.ndim == denom.ndim == 2:
        dndd = np.gradient(num)[0]/np.gradient(denom)[0]

    #note recursive list comprehension when dimensions are different
    elif num.ndim == 2 and denom.ndim == 1:
        col_der = [derivatize(col, denom) for col in num.T]
        dndd = np.column_stack(col_der)

    elif num.ndim == 1 and denom.ndim == 2:
        col_der = [derivatize(num, col) for col in denom.T]
        dndd = np.column_stack(col_der)

    return dndd





