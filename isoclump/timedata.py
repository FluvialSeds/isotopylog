'''
This module contains the TimeData superclass and all corresponding subclasses.
'''

#for python 2 compatibility
from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['HeatingExperiment', 'GeologicHistory']

#import modules
import matplotlib.pyplot as plt
import numpy as np
import warnings

#import exceptions
# from .exceptions import(
# 	)

# #import helper functions
# from .core_functions import(
# 	)

# from .plotting_helper import(
# 	)

# from .summary_helper import(
# 	)

# from .ratedata_helper import(
# 	)

# from .timedata_helper import (
# 	)


class TimeData(object):
	'''
	Class to store time-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, t, T, calibration = 'PH12', clumps = 'CO47', d = None,
		d_std = None, ref_frame = 'CDES90', T_std = None):
		'''
		ADD DOCSTRING
		'''

	#define classmethod to import from csv file
	@classmethod
	def from_csv(cls):
		NotImplementedError

	#define method to change calibration
	def change_calibration(self, calibration):
		'''
		ADD DOCSTRING
		'''

	#define method to change reference frame
	def change_ref_frame(self, ref_frame):
		'''
		ADD DOCSTRING
		'''

	#define method to forward model RateData instance
	def forward_model(self):
		NotImplementedError

	#define function to input estimated values
	def input_estimated(self, d, d_std):
		'''
		ADD DOCSTRING
		'''

	#define function to plot
	def plot(self, ax = None, labs = None, md = None, rd = None):
		'''
		ADD DOCSTRING
		'''

	#define function to print summary
	def summary(self):
		'''
		ADD DOCSTRING
		'''


class HeatingExperiment(TimeData):
	__doc__='''
	Class for inputting and storing reordering experiment true (observed)
	and estimated (forward-modelled) clumped isotope data, calculating
	goodness of fit statistics, and reporting summary tables. This class can
	currently handle carbonate D47 only, but will expand to other isotope
	systems as they become available and more widely used.

	Parameters
	----------

	Raises
	------

	Warnings
	--------

	Notes
	-----

	See Also
	--------

	Examples
	--------

	**Attributes**

	References
	----------

	'''

	def __init__(self, t, T, calibration = 'PH12', clumps = 'CO47', d = None,
		d_std = None, dex = None, dex_std = None, ref_frame = 'CDES90', 
		tex = None, T_std = None):

		#adding a dummy line here to circumvent indentation error
		self.t = t

	
	#define classmethod to import from csv file
	@classmethod
	def from_csv(cls, file, calibration = 'PH12', clumps = 'CO47',
		culled = True, ref_frame = 'CDES90'):
		'''
		ADD DOCSTRING
		'''

	#define method to change calibration
	def change_calibration(self, calibration):
		'''
		ADD DOCSTRING
		'''

	#define method to change reference frame
	def change_ref_frame(self, ref_frame):
		'''
		ADD DOCSTRING
		'''

	#define method to cull data
	def cull_data(self, sigma, Teq):
		'''
		ADD DOCSTRING
		'''

	#define method to generate fit summary
	def fit_summary(self):
		'''
		ADD DOCSTRING
		'''

	#define method to forward mode rate data
	def forward_model(self, kDistribution):
		'''
		ADD DOCSTRING
		'''

	#define function to input estimated values
	def input_estimated(self, d, d_std):
		'''
		ADD DOCSTRING
		'''

	#define function to plot
	def plot(self, ax = None, yaxis = 'D', logx = False, logy = False):
		'''
		ADD DOCSTRING
		'''

	#define function to print summary
	def summary(self):
		'''
		ADD DOCSTRING
		'''


class GeologicHistory(TimeData):
	__doc__='''
	Class for inputting geologic time-temperature histories and estimating
	their clumped isotope evolution using a forward implementation of any of
	the kinetic models. This class can currently handle carbonate D47 only,
	but will expand to other isotope systems as they become available.

	Parameters
	----------

	Raises
	------

	Warnings
	--------

	Notes
	-----

	See Also
	--------

	Examples
	--------

	**Attributes**

	References
	----------

	'''

	def __init__(self, t, T, calibration = 'PH12', clumps = 'CO47', d = None,
		d_std = None, ref_frame = 'CDES90', T_std = None):

		#adding a dummy line here to circumvent indentation error
		self.t = t

	#define classmethod to import from csv file
	@classmethod
	def from_csv(cls, file, calibration = 'PH12', clumps = 'CO47',
		ref_frame = 'CDES90'):
		'''
		ADD DOCSTRING
		'''

	#define method to change calibration
	def change_calibration(self, calibration):
		'''
		ADD DOCSTRING
		'''

	#define method to change reference frame
	def change_ref_frame(self, ref_frame):
		'''
		ADD DOCSTRING
		'''

	#define method to forward model RateData instance
	def forward_model(self, EDistribution):
		'''
		ADD DOCSTRING
		'''

	#define function to input estimated values
	def input_estimated(self, d, d_std):
		'''
		ADD DOCSTRING
		'''

	#define function to plot
	def plot(self, ax = None, xaxis = 'time', yaxis = 'D', logx = False,
		logy = False):
		'''
		ADD DOCSTRING
		'''

	#define function to print summary
	def summary(self):
		'''
		ADD DOCSTRING
		'''







