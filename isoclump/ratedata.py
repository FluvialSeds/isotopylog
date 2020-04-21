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
	# _fit_HH20,
	)

class kDistribution(object):
	__doc__='''
	Add docstring here
	'''

	def __init__(self, k, 
		k_std = None, 
		kvec = None,
		model = 'HH20', 
		npt = None, 
		pkvec = None, 
		rmse = None
		):
		'''
		Initializes the class
		'''

		#input all attributes
		self.k = k
		self.k_std = k_std
		self.kvec = kvec
		self.model = model
		self.npt = npt
		self.pkvec = pkvec
		self.rmse = rmse

	@classmethod
	def invert_experiment(cls,
		heatingexperiment,
		fit_regularized = True, #for HH20inv
		k0 = [1e-3, 1e-4, 1.0001], #for SE15
		L_curve_kink = 1, #for HH20inv
		lam_max = 10, #for HH20
		lam_min = -50, #for HH20
		nlam = 300, #for HH20
		model = 'HH20',
		omega = 'auto', #for HH20inv
		thresh = 1e-6, #for PH12 and Hea14
		z = 6, #for SE15
		):
		'''
		Inverst a HeatingExperiment instance to generate rates
		'''

		#check which model and run the inversion
		#Passey and Henkes 2012
		if model == 'PH12':

			#fit the model
			k, k_std, rmse, npt = _fit_PH12(heatingexperiment, thresh)

			#this model has no kvec and pkvec
			kvec = pkvec = None

		#Henkes et al. 2014
		elif model == 'Hea14':

			#fit the model
			k, k_std, rmse, npt = _fit_Hea14(heatingexperiment, thresh)

			#this model has no kvec and pkvec
			kvec = pkvec = None

		#Stolper and Eiler 2015
		elif model == 'SE15':

			#fit the model
			k, k_std, rmse, npt = _fit_SE15(heatingexperiment, k0, z)

			#this model has no kvec and pkvec
			kvec = pkvec = None

		#Hemingway and Henkes 2020
		elif model == 'HH20':

			#fit the model
			k, k_std, rmse, npt = _fit_HH20(
				heatingexperiment, 
				lam_max, 
				lam_min, 
				nlam
				)

			#include regularized data if necessary
			if fit_regularized is True:

				kvec, pkvec = _fit_HH20inv(
					heatingexperiment, 
					L_curve_kink,
					lam_max,
					lam_min,
					nlam,
					omega
					)

			else:
				
				#this model has no kvec and pkvec
				kvec = pkvec = None

		else:
			raise ValueError('Invalid model string.')

		#run __init__ and return class instance
		return cls(
			k, 
			k_std = k_std,
			kvec = kvec,
			model = model, 
			npt = npt, 
			pkvec = pkvec,
			rmse = rmse
			)

	#DOES THIS PLOTTING METHOD MAKE SENSE? IT IS USELESS FOR ALL MODELS EXCEPT
	# FOR HH20 INVERSE! PROBABLY GET RID OF IT?
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