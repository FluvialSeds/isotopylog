'''
This module contains helper functions for the TimeData classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_assert_calib',
			'_assert_clumps',
			'_assert_ref_frame',
			]

import numpy as np
import pandas as pd

from types import LambdaType

#import exceptions
from .exceptions import(
	StringError,
	)

from .dictionaries import(
	caleqs
	)

#THESE FUNCTIONS ARE GOOD:

#function to calcualte a D value for a given reaction progress, D0, and T
def _calc_D_from_G(
	D0,
	G,
	Teq,
	calibration = 'Bea17',
	clumps = 'CO47',
	G_std = None,
	ref_frame = 'CDES90'
	):
	'''
	Calculates the clumped isotope value, D, for a given initial composition,
	equilibrium temperature, and reaction progress remaining.

	Parameters
	----------
	D0 : float
		The initial clumped isotope composition.

	G : array-like
		The array of inputted reaction progress remaining data. Length ``nd``.

	Teq : int or float
		The equilibrium temperature (in Kelvin) used to calculate reaction
		progress.

	calibration : string
		The T-D calibration to use for calculating equilibrium D. Defaults to
		``'Bea17'``.

	clumps : string
		The clumped isotope system being analyzed. Defaults to ``'CO47'``.

	G_std : None or array-like
		Uncertainty for each entry in G. If ``None``, assumes no uncertainty;
		defaults to ``None``. If not none, must have length ``nd``.

	ref_frame : string
		The reference frame to use for calculating equilibrium D. Defaults to
		``'CDES90'``.

	Returns
	-------

	D : np.ndarray
		The resulting clumped isotope values. Length ``nd``.


	D_std : np.ndarray
		Uncertainty associated with each entry in D. Length ``nd``.
	'''

	#do some clump-specific math
	if clumps == 'CO47':

		#calculate equilibrium D value
		Deq = caleqs[calibration][ref_frame](Teq)

		#calculate D values
		D = G*(D0 - Deq) + Deq

		#calcualte D_std, assuming no uncertainty in D0 and Deq
		# if it exists
		try:
			D_std = (D0 - Deq)*G_std

		except TypeError:
			D_std = np.zeros(len(D))

	return D, D_std

#function to calculate reaction progress for a given D and Teq
def _calc_G_from_D(
	d, 
	Teq, 
	calibration = 'Bea17', 
	clumps = 'CO47', 
	d_std = None,
	ref_frame = 'CDES90'
	):

	'''
	Calculates the reaction progress remaining, G, for a set of clumped isotope
	measurements and an inputted equilibrium temperature.

	Parameters
	----------

	d : array-like
		The array of inputted isotope data, with clumped isotope data in
		the first column. Reaction progress will be calculated relative to the
		first row of d (i.e., D0). Length ``nd``.

	Teq : int or float
		The equilibrium temperature (in Kelvin) used to calculate reaction
		progress.

	calibration : string
		The T-D calibration to use for calculating equilibrium D. Defaults to
		``'Bea17'``.

	clumps : string
		The clumped isotope system being analyzed. Defaults to ``'CO47'``.

	d_std : None or array-like
		Analytical uncertainty for each entry in d. If ``None``, assumes no
		uncertainty; defaults to ``None``. If not none, must have length ``nd``.

	ref_frame : string
		The reference frame to use for calculating equilibrium D. Defaults to
		``'CDES90'``.

	Returns
	-------

	G : np.ndarray
		The resulting reaction progress remaining. Length ``nd``.


	G_std : np.ndarray
		Uncertainty associated with each entry in G. Length ``nd``.
	'''

	#do some clump-specific math
	if clumps == 'CO47':

		#calcualte equilibrium D value
		Deq = caleqs[calibration][ref_frame](Teq)

		#convert to G
		D = d[:,0]
		
		#get uncertainty if it exists
		try:
			sigD = d_std[:,0]

		except TypeError:
			sigD = np.zeros(len(D))

		#extract initial data and calculate G
		D0 = D[0]
		sigD0 = sigD[0]
		G = 1 - (D0-D)/(D0-Deq)

		#calculate G_std, assuming Deq is known perfectly
		G_std = ((sigD0*(D-Deq)/((D0-Deq)**2))**2 + (sigD/(D0-Deq))**2)**0.5

	return G, G_std


#FUNCTIONS BELOW HERE NEED TO BE REASSESSED:

#data importing functions
def _read_csv(file):
	'''
	Add docstring
	'''

	#check data format and raise appropriate errors
	if isinstance(file, str):
		#import as dataframe
		file = pd.read_csv(
			file,
			# index_col = 0,
			)

	elif not isinstance(file, pd.DataFrame):
		raise FileError(
			'file must be pd.DataFrame instance or path string')

	#check the clumped isotope system
	if 'D47' in file.columns:

		#set parameters
		clumps = 'CO47'
		iso_params = list(set(file['iso_params']))[0]
		ref_frame = list(set(file['ref_frame']))[0]
		T = list(set(file['T_C']))[0] + 273.15

		#extract the necessary data
		dex = file[['D47',
				    'd13C_vpdb', 
				    'd18O_vpdb']].values

		dex_std = file[['D47_std',
					    'd13C_std',
					    'd18O_std']].values

		tex = file['t_min'].values

	else:
		raise ValueError('Must be CO47 clumps!')

	return clumps, dex, dex_std, iso_params, ref_frame, tex, T

def _cull_data(calibration, clumps, dex, dex_std, ref_frame, T, tex):
	'''
	Add docstring
	'''

	#extract clumped values
	D = dex[:,0]
	D_std = dex_std[:,0]

	#check which clumped isotoep system
	if clumps == 'CO47':

		#calcualte equilibrium D value
		Deq = caleqs[calibration][ref_frame](T)

		#see which data are not within 1 sigma of Deq
		ind = np.abs(D - Deq) > D_std

		#TODO: exclude all data after the first instance where this 
		# fails (explicitly stated in PH12 Appendix A.2. to remove bias)

		#retain only values at those indices
		dex = dex[ind]
		dex_std = dex_std[ind]
		tex = tex[ind]

	return dex, dex_std, tex

#forward modeling functions
def _forward_PH12(kdistribution):

	return d, d_std

def _forward_Hea14(kdistribution):

	return d, d_std

def _forward_SE15(kdistribution):

	return d, d_std

def _forward_HH20(kdistribution):

	return d, d_std







#define hidden function for asserting calibration is okay
def _assert_calib(clumps, calibration, ref_frame):
	'''
	Asserts that the 'calibration' input is an acceptable string or lambda
	function and generates a calibration lambda equation.

	Paramters
	---------
	clumps : string
		String of clumped isotope system being measured

	calibration : string or lambda
		String or lambda function of T-D calibration equation to use. If str,
		automatically ensures that the equation is in the right ref. frame.

	ref_frame : string
		String of the reference frame being used

	Returns
	-------
	calibration : string
		String of T-D calibration to use

	calib_eq : lambda function
		The actual T-D calibration equation

	Raises
	------
	TypeError
		If 'calibration' is not a string or lambda function

	StringError
		If 'calibration' is not an acceptable string
	'''

	#check if string or lambda function
	cal_type = type(calibration).__name__

	#check if string or lambda
	if not isinstance(calibration, (str, LambdaType)):
		raise TypeError(
			'Attempting to input "calibration" of type %r. Must be a string'
			' or a lambda function' % cal_type)

	#if string, make sure it's an acceptable string and store
	if isinstance(calibration, str):

		if clumps == 'CO47':

			#compile all PH12 variants
			if calibration in ['PH12', 'ph12', 'Passey', 'passey']:
				calibration = 'PH12'

			#compile all SE15 variants
			elif calibration in ['SE15', 'se15', 'Stolper', 'stolper']:
				calibration = 'SE15'

			#compile all Bea17 variants
			elif calibration in ['Bea17', 'bea17', 'BEA17' 'Bonifacie', 
								 'bonifacie']:
				calibration = 'Bea17'

			else:
				raise StringError(
					'Attempting to input calibration %r for CO47 measurements.'
					' Must be one of "PH12", "SE15" or "Bea17"' % calibration)

		#store calibration equation
		calib_eq = caleqs[calibration][ref_frame]

	#if lambda function, store directly
	else:
		calib_eq = calibration
		calibration = 'User Inputted Lambda Function'

	return calibration, calib_eq

#define hidden function for asserting clumped isotope system is okay
def _assert_clumps(clumps):
	'''
	Asserts that the 'clumps' input is an acceptable string.

	Paramters
	---------
	clumps : string
		String of clumped isotope system being measured

	Returns
	-------
	clumps : string
		String of clumped isotope system being measured

	Raises
	------
	TypeError
		If 'clumps' is not a string

	StringError
		If 'clumps' is not an acceptable string
	'''

	#check if string
	cl_type = type(clumps).__name__

	if not isinstance(clumps, str):
		raise TypeError(
			'Attempting to input "clumps" value of type %r. Must be string.'
			% cl_type)

	#check if acceptable string
	#compile all variants into a single string
	if clumps in ['CO47', 'co47', 'D47']:
		clumps = 'CO47'

	else:
		raise StringError(
			'Attempting to calculate the %r clumped isotope system. Model'
			' currently only calculates C-O clumped isotopes. Please enter'
			' "CO47"' % clumps)

	return clumps

#define hidden function for asserting reference frame is okay
def _assert_ref_frame(clumps, ref_frame):
	'''
	Asserts the 'ref_frame' input is an acceptbale string.

	Paramters
	---------
	clumps : string
		String of clumped isotope system being measured

	ref_frame : string
		String of the reference frame being used

	Returns
	-------
	ref_frame : string
		String of the reference frame being used

	Raises
	------
	TypeError
		If 'ref_frame' is not a string

	StringError
		If 'ref_frame' is not an acceptable string

	Warnings
	--------
	UserWarning
		If 'CDES' is inputted but no tmperature is specified. Assumes 90 C.
	'''

	#check if string
	rf_type = type(ref_frame).__name__

	if not isinstance(ref_frame, str):
		raise TypeError(
			'Attempting to input "ref_frame" value of type %r. Must be string.'
			% rf_type)

	#check if acceptable string
	
	#check if clumps is 'CO47'
	if clumps == 'CO47':
		
		#compile all Ghosh variants
		if ref_frame in ['Ghosh', 'ghosh', 'Gea06', 'Ghosh 2006']:
			ref_frame = 'Ghosh'

		#compile all CDES90 variants
		elif ref_frame in ['CDES90', 'cdes90', 'Cdes90', 'CDES 90', 'cdes 90']:
			ref_frame = 'CDES90'

		#compile all CDES25 variants
		elif ref_Frame in ['CDES25', 'cdes25', 'Cdes25', 'CDES 25', 'cdes 25']:
			ref_frame = 'CDES25'

		#warn if CDES but no temperature defined
		elif ref_frame in ['cdes', 'CDES', 'Cdes']:
			warnings.warn(
				'Reference frame "CDES" being used but no temperature is'
				' specified. Assuming 90 C!')

			ref_frame = 'CDES90'

		else:
			raise StringError(
				'Attempting to input reference frame %r for CO47 measurements.'
				' Must be one of: "Ghosh", "CDES25", or "CDES90"' % ref_frame)

	return ref_frame

#define hidden function for asserting iso_params is okay
def _assert_iso_params(clumps, iso_params):
	'''
	Asserts the 'iso_params' is an acceptable string.

	Paramters
	---------
	clumps : string
		String of clumped isotope system being measured

	iso_params : string
		String of the isotopic parameters beting used

	Returns
	-------
	iso_params : string
		String of the isotopic parameters being used

	Raises
	------
	TypeError
		If 'iso_params' is not a string

	StringError
		If 'iso_params' is not an acceptable string
	'''

	#check if string
	ip_type = type(iso_params).__name__

	if not isinstance(iso_params, str):
		raise TypeError(
			'Attempting to input "iso_params" value of type %r. Must be string.'
			% ip_type)

	#check if acceptable string
	
	#check if clumps is 'CO47'
	if clumps == 'CO47':
		
		#compile all Gonfiantini
		if iso_params in ['Gonfiantini','gonfiantini']:
			iso_params = 'Gonfiantini'

		#compile all Brand
		elif iso_params in ['Brand','brand']:
			iso_params = 'Brand'

		#compile all Craig + Assonov
		elif iso_params in ['Craig + Assonov', 'craig + assonov', 
						 'Craig+Assonov', 'craig+assonov',
						]:
			iso_params = 'Craig + Assonov'

		#compile all Chang + Li
		elif iso_params in ['Chang + Li', 'chang + li', 'Chang+Li', 'chang+li']:
			iso_params = 'Chang + Li'

		#compile all Craig + Li
		elif iso_params in ['Craig + Li', 'craig + li', 'Craig+Li', 'craig+li']:
			iso_params = 'Craig + Li'

		#compile all Barkan
		elif iso_params in ['Barkan', 'barkan']:
			iso_params = 'Barkan'

		#compile all Passey
		elif iso_params in ['Passey', 'passey']:
			iso_params = 'Passey'

		else:
			raise StringError(
				'Attempting to input isotope parameters %r for CO47 measurements.'
				' Must be one of: "Brand", "Gonfiantini", "Craig + Assonov",'
				' "Chang + Li", "Craig + Li", "Barkan", or "Passey".' % iso_params)

	return iso_params


