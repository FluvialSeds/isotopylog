'''
Module to store all the core functions for ``isoclump``.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['Arrhenius_plot',
			'assert_len',
			'change_ref_frame',
			'calc_cooling_rate',
			'calc_ds',
			'calc_fs',
			'calc_L_curve',
			'calc_Teq',
			'calc_Deq',
			'resetting_plot',
			'cooling_plot',
			]

import matplotlib.pyplot as plt
import numpy as np

from collections import Sequence

#import exceptions
from .exceptions import(
	ArrayError,
	LengthError,
	)

#define funtction for making Arrhenius plots
def Arrhenius_plot(rate_list, ax = None, xaxis = 'Tinv', yaxis = 'mu'):
	'''
	ADD DOCSTRING
	'''

#define function to assert length of array
def assert_len(data, n):
	'''
	Asserts that an array has length `n` and `float` datatypes.

	Parameters
	----------
	data : scalar or array-like
		Array to assert has length n. If scalar, generates an np.ndarray
		with length n.

	n : int
		Length to assert

	Returns
	-------
	array : np.ndarray
		Updated array, now of class np.ndarray and with length n.

	Raises
	------
	ArrayError
		If inputted data not int or array-like (excluding string).

	LengthError
		If length of the array is not n.
	'''

	#assert that n is int
	n = int(n)

	#assert data is in the right form
	if isinstance(data, (int, float)):
		data = data*np.ones(n)
	
	elif isinstance(data, Sequence) or hasattr(data, '__array__'):
		
		if isinstance(data, str):
			raise ArrayError(
				'Data cannot be a string')

		elif len(data) != n:
			raise LengthError(
				'Cannot create array of length %r if n = %r' \
				% (len(data), n))

	else:
		raise ArrayError('data must be scalar or array-like')

	return np.array(data).astype(float)

#define function to change reference frame
def change_ref_frame(ds, ref_frame, clumps = 'CO47'):
	'''
	ADD DOCSTRING
	'''

#define function to calculate estimated cooling rate
def calc_cooling_rate(ds, EDistribution):
	'''
	ADD DOCSTRING
	'''

#define function to calculate d values from fractional abundances
def calc_ds(fs, clumps = 'CO47', ref_frame = 'CDES90'):
	'''
	ADD DOCSTRING
	'''

#define function to calculate fractional abundances from d values
def calc_fs(ds, clumps = 'CO47', ref_frame = 'CDES90'):
	'''
	ADD DOCSTRING
	'''

#define function calculate Tikhonov regularization "L-curve"
def calc_L_curve(HeatingExperiment, kmin = 1e-50, kmax = 1e20, nk = 300):
	'''
	ADD DOCSTRING
	'''

#define function to calculate equilibrium T given D
def calc_Teq(ds, calibration = 'PH12', clumps = 'CO47', ref_frame = 'CDES90'):
	'''
	ADD DOCSTRING
	'''

#define function to calculate equilibrium D given T
def calc_Deq(T, calibration = 'PH12', clumps = 'CO47', ref_frame = 'CDES90'):
	'''
	ADD DOCSTRING
	'''

#define function to generate resetting plot a la Henkes et al. (2014)
def resetting_plot(EDistribution):
	'''
	ADD DOCSTRING
	'''

#define function to generate cooling plot a la Henkes et al. (2014)
def cooling_plot(EDistribution):
	'''
	ADD DOCSTRING
	'''









