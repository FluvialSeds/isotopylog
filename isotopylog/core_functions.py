'''
Module to store core functions for isoclump package-level methods.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = [
			'derivatize',
			'geologic_history'
			]

import matplotlib.pyplot as plt
import numpy as np

#import necessary calculation functions
from .calc_funcs import(
	_ghHea14,
	_ghHH20,
	_ghPH12,
	_ghSE15,
	_Jacobian,
	)

#import dictionaries with conversion information
from .dictionaries import(
	caleqs,
	)

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

#define function to predict D47 evolution along geologic history
def geologic_history(
	t, 
	T, 
	ed, 
	d0,
	d0_std = [0.,0.,0.],
	calibration = 'Bea17', 
	iso_params = 'Gonfiantini',
	ref_frame = 'CDES90',
	nlam = 400,
	z = 6,
	**kwargs
	):
	'''
	Predicts the D47 evolution when a given ``ic.EDistribution`` model is 
	subjected to any arbitrary time-temperature history.
	
	Parameters
	----------

	t : array-like
		Array of time points, in the same temporal units used to calculate
		the ``isoclump.EDistribution`` object passed to this function. Of
		length ``nt``.

	T : array-like
		Array of temperatures at each time point, in Kelvin. Of length ``nt``.

	ed : isoclump.EDistribution
		The ``ic.EDistribution`` object containing the activation energy
		parameters used for forward modeling.

	d0 : array-like
		Array of initial isotope composition, in the order [D47, d13C, d18O],
		with d13C and d18O both reported relative to VPDB. Note that d13C and
		d18O are only used if ``ed.model = 'SE15'``; for other model types,
		these are unused and arbitrary values can be passed.

	d0_std : array-like
		Uncertainty associated with the values in d0, as +/- 1 standard
		deviation. Defaults to array of zeros.

	calibration : str
		The D-T calibration equation to use for forward modeling. Defaults to
		``'Bea17'`` for Bonifacie et al. (2017).

	iso_params : string
		The isotope parameters used to calculate clumped data. For example, if
		``clumps = 'CO47'``, then isotope parameters are R13_vpdb, R17_vpdb,
		R18_vpdb, and lam17. Following Daëron et al. (2016) nomenclature,
		options are: \n
			``'Barkan'``: for Barkan and Luz (2005) lam17\n
			``'Brand'`` (equivalent to ``'Chang+Assonov'``): for Brand (2010)\n
			``'Chang+Li'``: for Chang and Li (1990) + Li et al. (1988) \n
			``'Craig+Assonov'``: for Craig (1957) + Assonov and Brenninkmeijer 
			(2003)\n
			``'Craig+Li'``: for Craig (1957) + Li et al. (1988)\n
			``'Gonfiantini'``: for Gonfiantini et al. (1995)\n
			``'Passey'``: for Passey et al. (2014) lam17\n
		Defaults to ``'Gonfiantini'``. Only used if ``ed.model = 'SE15'``.

	ref_frame : str
		The reference frame used to generate D47 values. Defaults to
		``'CDES90'``.

	nlam : int
		The number of points to use in the lambda array. Only applies if
		``ed.model = 'HH20'``; for other model types, this is unused. Defaults 
		to ``400``.

	z : int
		The mineral coordination number. Only applies if ``ed.model = 'SE15'``;
		for other model types, this is unused. Defaults to ``6`` as suggested
		in Stolper and Eiler (2015).

	Returns
	-------

	D : np.array
		Array of resulting D47 values. Of length ``nt``.

	D_std : np.array
		Array of corresponding uncertainty for resulting D values. Of length 
		``nt``.

	Raises
	------

	TypeError
		If inputted 'calibration' and/or 'ref_frame' are not strings.

	ValueError
		If inputted t and T arrays are not the same length.

	ValueError
		If inputted 'calibration' and/or 'ref_frame' arrays are not acceptable
		strings.

	See Also
	--------

	isoclump.EDistribution
		The class that contains the activation energy parameters that are
		to be modeled.

	Examples
	--------

	Estimate resetting temperatures during heating for an arbitrarily chosen
	starting isotope composition. This example creates an ``ic.EDistribution``
	instance by importing literature values of the 'SE15' model type, and 
	plots results::

		#import packages
		import isoclump as ic
		import matplotlib.pyplot as plt

		#generate EDistribution instance
		ed = ic.EDistribution.from_literature(
			mineral = 'calcite', 
			reference = 'SE15', 
			Tref = 700)

		#define the initial composition and the time-temperature evolutions
		d0 = [0.55, 0, 0] #starting D47 = 0.55, d13C and d18O both zero
		d0_std = [0.010, 0, 0] #assume some reasonable D47 uncertainty

		T0 = 25 + 273.15 #assume starting at 25C, ending at 350C
		Tf = 350 + 273.15
		beta = 100/(1e6*365*24*3600) #100C/million years, converted to seconds

		t0 = 0
		tf = (Tf-T0)/beta
		nt = 500

		T = np.linspace(T0, Tf, nt)
		t = np.linspace(t0, tf, nt)

		#now calculate D at each time point
		D, Dstd = ic.geologic_history(t, T, ed, d0, d0_std = d0_std)

		#plot results, along with equilibrium D at each time point
		Deq = ic.Deq_from_T(T)
		tmyr = t/(1e6*365*24*3600) #getting t in Myr for plotting

		fig,ax = plt.subplots(1,1)
		ax.plot(tmyr, D, label = 'forward-modeled data')
		ax.fill_between(tmyr, D - Dstd, D + Dstd, alpha = 0.5)
		ax.plot(tmyr,Deq, label = 'equilibrium values at each time point')

		ax.set_xlabel('time (Myr)')
		ax.set_ylabel('D47 (‰)')
		ax.legend(loc = 'best')

	Note the non-monotonic behavior that arises from the intermediate "pair"
	reservoir (see Stolper and Eiler 2015, Lloyd et al. 2018, and Chen et al.,
	2019 for further details). 

	.. image:: ../_images/gh_1.png

	Similarly, one can estimate cooling closure temperatures. This is identical
	to the above example, only the temperature axis is reversed and D is
	assumed to be in equilibrium at T0::

		#reverse T and Deq arrays
		T = T[::-1]
		Deq = Deq[::-1]

		#make D0 in equilibrium
		D0 = ic.Deq_from_T(T[0])
		d0 = [D0, 0, 0] #still d13C and d18O of zero

		#fit the new t-T trajectory
		D, Dstd = ic.geologic_history(t, T, ed, d0, d0_std = d0_std)

		#plot the results
		fig,ax = plt.subplots(1,1)
		ax.plot(tmyr, D, label = 'forward-modeled cooling data')
		ax.fill_between(tmyr, D - Dstd, D + Dstd, alpha = 0.5)
		ax.plot(tmyr,Deq, label = 'equilibrium values at each time point')

		ax.set_xlabel('time (Myr)')
		ax.set_ylabel('D47 (‰)')
		ax.legend(loc = 'best')

	.. image:: ../_images/gh_2.png

	References
	----------

	[1] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.\n
	[2] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.\n
	[3] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.\n
	[4] Lloyd et al. (2018) *Geochim. Cosmochim. Ac.*, **242**, 1--20.\n
	[5] Chen et al. (2019) *Geochim. Cosmochim. Ac.*, **258**, 156--173.\n
	[7] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.

	'''

	#check inputs are correct
	if len(T) != len(t):
		raise ValueError(
			'unexpected length of T array %n. Must be same length as t array.' 
			% len(T)
			)

	if calibration not in ['PH12', 'SE15', 'Bea17']:
		raise ValueError(
			"unexpected calibration %s. Must be 'PH12', 'SE15', or 'Bea17'."
			% calibration
			)

	elif not isinstance(calibration, str):
		ct = type(calibration).__name__
		raise TypeError(
			'unexpected calibration of type %s. Must be string.' % ct
			)

	if ref_frame not in ['Ghosh25', 'Ghosh90', 'CDES25', 'CDES90']:
		raise ValueError(
			"unexpected ref_frame %s. Must be 'Ghosh25', 'Ghosh90', 'CDES25',"
			" or 'CDES90'." % ref_frame
			)

	elif not isinstance(ref_frame, str):
		rft = type(ref_frame).__name__
		raise TypeError(
			'unexpected calibration of type %s. Must be string.' % rft
			)

	#calculate array of D47eq
	Deq = caleqs[calibration][ref_frame](T)
	D0 = d0[0]
	D0_cov = d0_std[0]**2
	Tref = ed.Tref

	#calculate D depending on model type

	#Henkes et al. 2014 model
	if ed.model == 'Hea14':

		#extract relevant parameters and uncertainty in the order:
		# Ec, lnkcref, Ed, lnkdref, E2, lnk2ref
		p = ed.Eparams.T.flatten()
		pcov = ed.Eparams_cov

		#append D0 to params and params_cov to include those in uncertainty
		p = np.append(p, D0)

		npt = len(pcov)
		pcov = np.append(pcov, [np.zeros(npt)], 0)
		pcov = np.append(pcov, np.append(np.zeros(npt), D0_cov).reshape(-1,1),1)

		#solve for D evolution
		D = _ghHea14(t, *p, Deq, T, Tref)

		#define lambda function for uncertainty propagation
		lamfunc = lambda t,Ec,lnkcref,Ed,lnkdref,E2,lnk2ref,D0 : _ghHea14(
			t, 
			Ec,
			lnkcref,
			Ed,
			lnkdref,
			E2,
			lnk2ref,
			D0,
			Deq,
			T,
			Tref)

	#Hemingway and Henkes 2020 model
	elif ed.model == 'HH20':

		#extract relevant parameters and uncertainty in the order:
		# Emu, lnkmuref, Esig, lnksigref
		p = ed.Eparams.T.flatten()
		pcov = ed.Eparams_cov

		#append D0 to params and params_cov to include those in uncertainty
		p = np.append(p, D0)

		npt = len(pcov)
		pcov = np.append(pcov, [np.zeros(npt)], 0)
		pcov = np.append(pcov, np.append(np.zeros(npt), D0_cov).reshape(-1,1),1)

		#solve for D evolution
		D = _ghHH20(t, *p, Deq, T, Tref)

		#define lambda function for uncertainty propagation
		lamfunc = lambda t, Emu, lnkmuref, Esig, lnksigref, D0 : _ghHH20(
			t, 
			Emu, 
			lnkmuref, 
			Esig, 
			lnksigref,
			D0,
			Deq,
			T,
			Tref,
			nlam = nlam)

	#Passey and Henkes 2012 model
	elif ed.model == 'PH12':

		#extract relevant parameters and uncertainty in the order: 
		# E, lnkref
		p = ed.Eparams[:,0]
		pcov = ed.Eparams_cov[:2,:2]

		#append D0 to params and params_cov to include those in uncertainty
		p = np.append(p, D0)

		npt = len(pcov)
		pcov = np.append(pcov, [np.zeros(npt)], 0)
		pcov = np.append(pcov, np.append(np.zeros(npt), D0_cov).reshape(-1,1),1)

		#solve for D evolution
		D = _ghPH12(t, *p, Deq, T, Tref)

		#define lambda function for uncertainty propagation
		lamfunc = lambda t, E, lnkref, D0 : _ghPH12(
			t, 
			E, 
			lnkref, 
			D0, 
			Deq,
			T, 
			Tref)

	#Stolper and Eiler 2015 model
	elif ed.model == 'SE15':

		#extract relevant parameters and uncertainty in the order:
		# E1, lnk1ref, Eds, lnkdsref, Emp, mpref
		p = ed.Eparams.T.flatten()
		pcov = ed.Eparams_cov

		#append D0 to params and params_cov to include those in uncertainty
		p = np.append(p, D0)

		npt = len(pcov)
		pcov = np.append(pcov, [np.zeros(npt)], 0)
		pcov = np.append(pcov, np.append(np.zeros(npt), D0_cov).reshape(-1,1),1)

		#solve for D evolution
		D = _ghSE15(
			t, 
			*p, 
			d0[1], 
			d0[2], 
			T, 
			Tref, 
			calibration = calibration,
			iso_params = iso_params,
			ref_frame = ref_frame,
			z = z)[0]

		#define lambda function for uncertainty propagation
		lamfunc = lambda t, E1, lnk1ref, Eds, lnkdsref, Emp, mpref, D0 : _ghSE15(
			t, 
			E1, 
			lnk1ref, 
			Eds, 
			lnkdsref, 
			Emp, 
			mpref,
			D0,
			d0[1], 
			d0[2], 
			T, 
			Tref, 
			calibration = calibration,
			iso_params = iso_params,
			ref_frame = ref_frame,
			z = z)[0]

	#calculate Jacobian and D uncertainty
	J = _Jacobian(lamfunc, t, p, **kwargs)
	Dcov = np.dot(J, np.dot(pcov, J.T))
	D_std = np.sqrt(np.diag(Dcov))

	return D, D_std


if __name__ == '__main__':
	import isoclump as ic