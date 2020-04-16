'''
This module contains the RateData superclass and all corresponding subclasses.
'''

#for python 2 compatibility
from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = [
	'kDistribution',
	'EDistribution'
	]

#import modules
import matplotlib.pyplot as plt
import numpy as np

#import helper functions
from .ratedata_helper import(
	_fit_PH12,
	_fit_Hea14,
	_fit_SE15,
	_fit_HH20,
	)

def kDistribution(object):
	__doc__='''
	Add docstring here
	'''

	def __init__(self, k, k_std = None, model = 'HH20', pk = None, RMSE = None):
		'''
		Initializes the class
		'''

		#input all attributes
		self.k = k
		self.k_std = k_std
		self.model = model
		self.RMSE = RMSE

		if pk is None:
			self.pk = np.ones(len(k))
		else:
			self.pk = pk #only used for HH20 model

	@classmethod
	def invert_experiment(cls, heatingexperiment, model = 'HH20'):
		'''
		Inverst a HeatingExperiment instance to generate rates
		'''

		#check which model and run the inversion
		if model == 'PH12':

			k, k_std, RMSE = _fit_PH12(heatingexperiment)
			pk = None

		elif model == 'Hea14':

			k = _fit_Hea14(heatingexperiment)
			pk = None

		elif model == 'SE15':

			k = _fit_SE15(heatingexperiment)
			pk = None

		elif model == 'HH20':

			k, pk = _fit_HH20(heatingexperiment)

		else:
			raise ValueError('Invalid model string.')

		#run __init__ and return class instance
		return cls(k, k_std = k_std, model = model, pk = pk, RMSE = RMSE)

	def plot(ax = None, **kwargs):
		'''
		Method for plotting data
		'''

		#make axis if necessary
		if ax is None:
			_, ax = plt.subplots(1,1)

		#plot data
		ax.plot(self.pk, self.k, **kwargs)

		#return result
		return ax

	def summary(self):
		'''
		Method for printing summary information
		'''

		#make a summary table
		sum_vars = {
			'Model' : self.model,
			'k' : self.k
			}

		#make into a table
		sum_table = pd.Series(sum_vars)

		#return table
		return sum_table


class EDistribution(object):
	__doc__='''
	Add docstring here
	'''

	def __init__():
		'''
		Initializes the class
		'''

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