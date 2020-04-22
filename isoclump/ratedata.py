'''
This module contains the kDistribution and EDistribution classes.

Updated: 22/4/20
By: JDH
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
import inspect
import matplotlib.pyplot as plt
import numpy as np

#import helper functions
from .ratedata_helper import(
	fit_Hea14,
	fit_HH20,
	fit_HH20inv,
	fit_PH12,
	fit_SE15,
	)

# TODO: 
# * Update fit_HH20inv to throw error if unexpected **kwargs passed to L-curve
# * Define @property functions
# * Update __repr__ to output summary table
# * Customize other magic method behavior??
# * Write docstring

class kDistribution(object):
	__doc__='''
	Class description synopsis.

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

	References
	----------
	
	**Attributes**

	'''

	#define all the possible attributes for __init__ using _attrs
	_attrs = [
		'lam', 
		'model', 
		'npt', 
		'omega', 
		'params',
		'params_std', 
		'rho_lam', 
		'rho_lam_inv',
		'res_inv',
		'rgh_inv',
		'rmse',
		]

	#initialize the object
	def __init__(self, params, model, **kwargs):
		'''
		Initilizes the object.

		Parameters
		----------
		params : array-like
			The kinetic parameters associated with this instance; the exact
			length and values of `params` depends on the model used.

		model : string
			The type of model to use for fitting. Must be one of:

				"PH12", \n
				"Hea14", \n
				"SE15", \n
				"HH20"

		Returns
		-------
		kd : ic.kDistribution
			The returned `kDistribution` instance.
		'''

		#first make everything in _attrs = None
		for k in self._attrs:
			setattr(self, k, None)

		#then set arguments
		self.params = params
		self.model = model

		#finally set all attributes in kwargs and raise exception if unknown
		for k, v in kwargs.items():

			if k in self._attrs:
				setattr(self, k, v)

			else:
				raise TypeError(
					'__init__() got an unexpected keyword argument %s' % k)

	#define classmethod for generating kDistribution instance from data
	@classmethod
	def invert_experiment(cls, he, model = 'HH20', fit_reg = False, **kwargs):
		'''
		Method description synopsis.

		Parameters
		----------
		he : isoclump.HeatingExperiment
			The `ic.HeatingExperiment` instance to fit.

		model : string
			The type of model to use for fitting. Must be one of:

				"PH12", \n
				"Hea14", \n
				"SE15", \n
				"HH20"

		fit_reg : boolean
			Tells the function whether or not to find the regularized inverse
			solution in addition to the lognormal solution. This only applies
			if `model = 'HH20'`.

		Returns
		-------
		kd : isoclump.kDistribution
			The resutling `ic.kDistribution` instance containing the fit
			parameters.

		Raises
		------
		ValueError
			If `model` is not an acceptable string.

		TypeError
			If `model` is not a string.

		Warnings
		--------

		Notes
		-----

		See Also
		--------

		Examples
		--------

		References
		----------
		[1] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
		[2] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
		[3] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
		[4] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
		'''

		#check which model and run the corresponding inversion method:

		#Passey and Henkes 2012
		if model == 'PH12':

			#fit the model
			params, params_std, rmse, npt = fit_PH12(he, **kwargs)

		#Henkes et al. 2014
		elif model == 'Hea14':

			#fit the model
			params, params_std, rmse, npt = fit_Hea14(he, **kwargs)

		#Stolper and Eiler 2015
		elif model == 'SE15':

			#fit the model
			params, params_std, rmse, npt = fit_SE15(he, **kwargs)

		#Hemingway and Henkes 2020
		elif model == 'HH20':

			#extract appropriate kwargs to pass
			a = [k for k, v in inspect.signature(fit_HH20).parameters.items()]
			kwa = {k : kwargs[k] for k in dict(kwargs) if k in a}

			#fit the model
			params, params_std, rmse, npt, lam, rho_lam = fit_HH20(he, **kwa)

			#include regularized data if necessary
			if fit_reg is True:

				#fit the model using the inverse function
				rho_lam_inv, omega, res_inv, rgh_inv = fit_HH20inv(he, **kwargs)

			else:
				
				#this model has no rho_lam_inv and associated statistics
				rho_lam_inv = omega = res_inv = rgh_inv = None

		#raise exception if it's not an acceptable string
		elif isinstance(model, str):
			raise ValueError(
				'%s is an invalid model string. Must be one of: "PH12",'
				'"Hea14", "SE15", or "HH20"' % model)

		#raise different exception if it's not a string
		else:

			mdt = type(mdodel).__name__

			raise TypeError(
				'Unexpected model of type %s. Must be string.' % mdt)

		#set HH20 specific attributes to not if model is not HH20
		if model != 'HH20':
			lam = omega = rho_lam = rho_lam_inv = res_inv = rgh_inv = None

		#return class instance
		return cls(
			params,
			lam = lam,
			model = model,
			npt = npt,
			omega = omega,
			params_std = params_std,
			rho_lam = rho_lam,
			rho_lam_inv = rho_lam_inv,
			res_inv = res_inv,
			rgh_inv = rgh_inv,
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


	#customize __repr__ method for printing summary
	def __repr__(self):
		return str(self.params)

	#TODO: Customize other magic method behavior?
	#TODO: Customize @property behavior?



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


# if __name__ == __main__: