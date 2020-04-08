'''
This module contains the RateData superclass and all corresponding subclasses.
'''

#for python 2 compatibility
from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['Rates', 'Energies']

#import modules
import matplotlib.pyplot as plt
import numpy as np
import warnings

#import exceptions
from .exceptions import(
	)

#import helper functions
from .core_functions import(
	)

from .plotting_helper import(
	)

from .summary_helper import(
	)

from .ratedata_helper import(
	)

from .timedata_helper import (
	)

class RateData(object):
	'''
	Class to store rate-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(model = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define classmethod for defining instance directly from literature values
	@classmethod
	def from_literature(clumps = 'CO47', mineral = 'calcite', paper = 'PH12'):
		raise NotImplementedError

	#define method for plotting results
	def plot(ax = ax, xaxis = 'k', yaxis = 'pk', logx = True):
		'''
		ADD DOCSTRING
		'''

	#define method for printing results summary
	def summary():
		'''
		ADD DOCSTRING
		'''


class Rates(RateData):
	__doc__='''
	Class for inputting and storing clumped isotope rate data, whether derived
	from the literature or from an inverse fit to heating experiment data,
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

	def __init__():
		'''
		ADD DOCSTRING
		'''

	#define classmethod for defining instance directly from literature values
	@classmethod
	def from_literature(T, clumps = 'CO47', mineral = 'calcite', paper = 'PH12'):
		'''
		ADD DOCSTRING
		'''

	#define function for inputting experiment inverse-model results
	def invert_experiment(HeatingExperiment, model = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define method for plotting results
	def plot(ax = ax, xaxis = 'k', yaxis = 'pk', logx = True):
		'''
		ADD DOCSTRING
		'''

	#define method for printing results summary
	def summary():
		'''
		ADD DOCSTRING
		'''


class Energies(RateData):
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

	def __init__(E, pE, model = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define classmethod for defining instance directly from literature values
	@classmethod
	def from_literature(clumps = 'CO47', mineral = 'calcite', paper = 'HH20'):
		'''
		ADD DOCSTRING
		'''

	#define classmethod for defining instance directly from a set of rates
	@classmethod
	def from_rates(rate_list):
		'''
		ADD DOCSTRING
		'''

	#define method for making Arrhenius plots
	def Arrhenius_plot(ax = ax, xaxis = 'Tinv', yaxis = 'mu'):
		'''
		ADD DOCSTRING
		'''

	#define method for plotting results
	def plot(ax = ax, xaxis = 'E', yaxis = 'pE'):
		'''
		ADD DOCSTRING
		'''

	#define method for printing results summary
	def summary():
		'''
		ADD DOCSTRING
		'''

	#define function to update E distribution using new results
	def update(ratedata_list):
		'''
		ADD DOCSTRING
		'''








