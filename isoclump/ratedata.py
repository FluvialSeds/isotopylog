'''
This module contains the RateData superclass and all corresponding subclasses.
'''

#for python 2 compatibility
from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['kDistribution', 'EDistribution']

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

class RateData(object):
	'''
	Class to store rate-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, model = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define method for plotting results
	def plot(self, ax = None, xaxis = 'k', yaxis = 'pk', logx = True):
		'''
		ADD DOCSTRING
		'''

	#define method for printing results summary
	def summary(self):
		'''
		ADD DOCSTRING
		'''


class kDistribution(RateData):
	__doc__='''
	Class for inputting and storing clumped isotope rate data, whether entered
	manually or derived from an inverse fit to heating experiment data,
	calculating goodness-of-fit statistics, reporting summary tables, and
	generating plots. This class can handle any of the published clumped isotope
	reordering kinetic models: Passey and Henkes (2012), Henkes et al. (2014),
	Stolper and Eiler (2015), or Hemingway and Henkes (2020). This class can 
	currently handle carbonate D47 only, but will expand to other isotope 
	systems as they become available and more widely used.

	Parameters
	----------

	Raises
	------

	Warnings
	--------
	UserWarning
		Foo bar baz

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

	def __init__(self, kvals, model = 'HH20'):

		#adding a dummy line here to circumvent indentation error
		self.model = model

	#define classmethod for inputting experiment inverse-model results
	@classmethod
	def invert_experiment(cls, HeatingExperiment, model = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define method for plotting results
	def plot(self, ax = None, xaxis = 'k', yaxis = 'pk', logx = True):
		'''
		ADD DOCSTRING
		'''

	#define method for printing results summary
	def summary(self):
		'''
		ADD DOCSTRING
		'''


class EDistribution(RateData):
	__doc__='''
	Class for inputting and storing clumped isotope activation energy data, E,
	whether from the literature or from a set of inverse fits to heating
	experiment data, calculating goodness-of-fit statistics, reporting summary
	tables, and generating plots and Arrhenius plots. This class can handle any
	of the published clumped isotope reordering kinetic models: Passey and
	Henkes (2012), Henkes et al. (2014), Stolper and Eiler (2015), or Hemingway
	and Henkes (2020). This class can currently handle carbonate D47 only, but
	will expand to other isotope systems as they become available and more 
	widely used.

	Parameters
	----------

	Raises
	------

	Warnings
	--------
	UserWarning
		Foo bar baz

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

	def __init__(self, Evals, model = 'HH20'):

		#adding a dummy line here to circumvent indentation error
		self.model = model

	#define classmethod for defining instance directly from literature values
	@classmethod
	def from_literature(cls, clumps = 'CO47', mineral = 'calcite',
		paper = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define classmethod for defining instance directly from a set of rates
	@classmethod
	def from_rates(cls, rate_list):
		'''
		ADD DOCSTRING
		'''

	#define method for making Arrhenius plots
	def Arrhenius_plot(self, ax = None, xaxis = 'Tinv', yaxis = 'mu'):
		'''
		ADD DOCSTRING
		'''

	#define method for plotting results
	def plot(self, ax = None, xaxis = 'E', yaxis = 'pE'):
		'''
		ADD DOCSTRING
		'''

	#define method for printing results summary
	def summary(self):
		'''
		ADD DOCSTRING
		'''

	#define function to update E distribution using new results
	def update(self, ratedata_list):
		'''
		ADD DOCSTRING
		'''








