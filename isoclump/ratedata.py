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
import pandas as pd

#import helper functions
from .ratedata_helper import(
	fit_Hea14,
	fit_HH20,
	fit_HH20inv,
	fit_PH12,
	fit_SE15,
	)

#import necessary dictionaries
from .dictionaries import(
	mod_params
	)

class kDistribution(object):
	__doc__='''
	Class for inputting and storing clumped isotope rate data. Currently only
	accepts D47 clumps, but will be expanded in the future as new clumped
	system data becomes available.

	Parameters
	----------
	params : array-like
		A list of the rate parameters associated with a given kDistribution.
		The values and length of this array depend on the type of model being
		implemented:

			`'Hea14'`: [ln(kc), ln(kd), ln(k2)] \n
			`'HH20'`: [mu_lam, sig_lam] \n
			`'PH12'`: [ln(k), -intercept] \n
			`'SE15'`: [ln(k1), ln(k_dif_single), [pair]_0/[pair]_eq] \n

		See discussion in each reference for parameter definitions and
		further details. All `k` values should be in units of inverse time,
		although the exact time unit can change depending on inputs.

	model : string
		The type of model associated with a given kDistribution. Options are:

			`'Hea14'` \n
			`'HH20'` \n
			`'PH12'` \n
			`'SE15'`

	lam : None or array-like
		The ln(k) values over which the rate distribution is calculated. `lam`
		only applies when `model = 'HH20'`. Defaults to `None`.

	npt : None or int
		The number of data points used in the model fit. If `model = 'Hea14'`
		or `model = 'PH12'`, then `npt` is the number of points deemed to be
		in the linear region of the curve; otherwise, it is all data points.
		Defaults to `None`.

	omega : None or scalar
		The Tikhonov omega value used for inverse regularization. `omega` only
		applies when `model = 'HH20'` and `fit_reg = True`. Defaults to `None`.

	params_std : None or array-like
		Uncertainty associated with each parameter value, as +/- 1 sigma.
		Defaults to `None`.

	rho_lam : None or array-like
		The modeled lognormal probability density function of ln(k) values.
		`rho_lam` only applies when `model = 'HH20'`. Defaults to `None`.

	rho_lam_inv : None or array-like
		The modeled inverse probability density function of ln(k) values
		calculated using Tikhonov regularization. `rho_lam_inv` only applies
		when `model = 'HH20'` and `fit_reg = True`. Defaults to `None`.

	res_inv : None or float
		The residual norm the Tikhonov regularization model-data fit. `res_inv`
		only applies when `model = 'HH20'` and `fit_reg = True`. Defaults to
		`None`.

	rgh_inv : None or float
		The roughness norm the Tikhonov regularization model-data fit. `res_inv`
		only applies when `model = 'HH20'` and `fit_reg = True`. Defaults to
		`None`.

	rmse : None or float
		The root-mean-square-error of the model-data fit. Defaults to `None`.

	Raises
	------
	ValueError
		If an unexpected keyword argument is trying to be inputted.

	TypeError
		If inputted parameters of an unacceptable type.

	ValueError
		If an unexpected model name is trying to be inputted.

	See Also
	--------
	isoclump.EDistribution
		The class for combining multiple `kDistribution` instances and
		determining the underlying activation energies.

	isoclump.HeatingExperiment
		The class containing heating experiment clumped isotope data whose
		rate data are determined.

	Examples
	--------
	Generating a bare-bones kDistribution instance without fitting any
	actual data::

		#import packages
		import isoclump as ic

		#assume some values for HH20 model parameters
		params = [-14., 5.]

		#make instance
		kd = ic.kDistribution(params, 'HH20')

	Generating a kDistribution instance by fitting some experimental D47
	data contained in a HeatingExperiment object::

		#assume some he is a HeatingExperiment object
		kd = ic.kDistribution.invert_experiment(he, model = 'PH12')

	Same as above, but now including the Tikhonov regularization inverse fit
	for 'HH20' model type::

		#assume some he is a HeatingExperiment object
		kd = ic.kDistribution.invert_experiment(
			he,
			model = 'HH20',
			fit_reg = True
			)

	To visualize these results, we can generate a plot of 'HH20' model k
	distributions::

		#import necessary packages
		import matplotlib.pyplot as plt

		#make axis
		fig, ax = plt.subplots(1,1)

		#plot data
		kd.plot(ax = ax)

	Export summary information for storing and saving::

		sum_tab = kd.summary
		sum_tab.to_csv('file_name.csv')

	**Attributes**

	lam : None or array-like
		The ln(k) values over which the rate distribution is calculated.

	model : string
		The type of model associated with a given kDistribution.

	npt : None or int
		The number of data points used in the model fit.

	omega : None or scalar
		The Tikhonov omega value used for inverse regularization.

	params : array-like
		A list of the rate parameters associated with a given kDistribution.

	params_std : None or array-like
		Uncertainty associated with each parameter value, as +/- 1 sigma.

	summary : pd.DataFrame
		DataFrame containing all the summary data.

	rho_lam : None or array-like
		The modeled lognormal probability density function of ln(k) values.

	rho_lam_inv : None or array-like
		The modeled inverse probability density function of ln(k) values
		calculated using Tikhonov regularization.

	res_inv : None or float
		The residual norm the Tikhonov regularization model-data fit.

	rgh_inv : None or float
		The roughness norm the Tikhonov regularization model-data fit.

	rmse : None or float
		The root-mean-square-error of the model-data fit.

	References
	----------
	[1] Hansen (1994) *Numerical Algorithms*, **6**, 1-35.
	[2] Forney and Rothman (2012) *J. Royal Soc. Inter.*, **9**, 2255--2267.
	[3] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
	[4] Henkes et al. (2014) *Geochim. Cosmochim. Ac.*, **139**, 362--382.
	[5] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
	[6] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.
	[7] Hemingway and Henkes (2020) *Earth Planet. Sci. Lett.*, **X**, XX--XX.
	'''

	#define all the possible attributes for __init__ using _attrs
	_kwattrs = [
		'lam', 
		'npt', 
		'omega', 
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
			A list of the rate parameters associated with a given kDistribution.
			The values and length of this array depend on the type of model
			being implemented:

				`'Hea14'`: [ln(kc), ln(kd), ln(k2)] \n
				`'HH20'`: [mu_lam, sig_lam] \n
				`'PH12'`: [ln(k), -intercept] \n
				`'SE15'`: [ln(k1), ln(k_dif_single), [pair]_0/[pair]_eq] \n

			See discussion in each reference for parameter definitions and
			further details. All `k` values should be in units of inverse time,
			although the exact time unit can change depending on inputs.

		model : string
			The type of model associated with a given kDistribution. Options
			are:

				`'Hea14'` \n
				`'HH20'` \n
				`'PH12'` \n
				`'SE15'`

		lam : None or array-like
			The ln(k) values over which the rate distribution is calculated.
			`lam` only applies when `model = 'HH20'`. Defaults to `None`.

		npt : None or int
			The number of data points used in the model fit. If `model = 'Hea14'`
			or `model = 'PH12'`, then `npt` is the number of points deemed to
			be in the linear region of the curve; otherwise, it is all data
			points. Defaults to `None`.

		omega : None or scalar
			The Tikhonov omega value used for inverse regularization. `omega`
			only applies when `model = 'HH20'` and `fit_reg = True`. Defaults
			to `None`.

		params_std : None or array-like
			Uncertainty associated with each parameter value, as +/- 1 sigma.
			Defaults to `None`.

		rho_lam : None or array-like
			The modeled lognormal probability density function of ln(k) values.
			`rho_lam` only applies when `model = 'HH20'`. Defaults to `None`.

		rho_lam_inv : None or array-like
			The modeled inverse probability density function of ln(k) values
			calculated using Tikhonov regularization. `rho_lam_inv` only applies
			when `model = 'HH20'` and `fit_reg = True`. Defaults to `None`.

		res_inv : None or float
			The residual norm the Tikhonov regularization model-data fit.
			`res_inv` only applies when `model = 'HH20'` and `fit_reg = True`.
			Defaults to `None`.

		rgh_inv : None or float
			The roughness norm the Tikhonov regularization model-data fit.
			`res_inv` only applies when `model = 'HH20'` and `fit_reg = True`.
			Defaults to `None`.

		rmse : None or float
			The root-mean-square-error of the model-data fit. Defaults to `None`.

		Returns
		-------
		kd : ic.kDistribution
			The returned `kDistribution` instance.

		Raises
		------
		ValueError
			If an unexpected keyword argument is trying to be inputted.

		TypeError
			If inputted parameters of an unacceptable type.

		ValueError
			If an unexpected model name is trying to be inputted.
		'''

		#first make everything in _attrs = None
		for k in self._kwattrs:
			setattr(self, k, None)

		#then set arguments
		self._params = params
		self._model = model

		#finally set all attributes in kwargs and raise exception if unknown
		for k, v in kwargs.items():

			if k in self._kwattrs:
				setattr(self, k, v)

			else:
				raise ValueError(
					'__init__() got an unexpected keyword argument %s' % k)

	#define classmethod for generating kDistribution instance from data
	@classmethod
	def invert_experiment(cls, he, model = 'HH20', fit_reg = False, **kwargs):
		'''
		Classmethod for generating a `kDistribution` instance directly by
		inverting a `ic.HeatingExperiment` object that contains clumped isotope
		heating experiment data.

		Parameters
		----------
		he : isoclump.HeatingExperiment
			The `ic.HeatingExperiment` instance to fit.

		model : string
			The type of model associated with a given kDistribution. Options
			are:

				`'Hea14'` \n
				`'HH20'` \n
				`'PH12'` \n
				`'SE15'`

			See the relevant documentation on each model fit function for
			details and descriptions of a given model:

				fit_PH12 \n
				fit_Hea14 \n
				fit_SE15 \n
				fit_HH20 \n
				fit_HH20inv \n

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

		See Also
		--------
		isoclump.fit_Hea14
			Fitting function for Henkes et al. (2014) model.

		isoclump.fit_HH20
			Fitting function for Hemingway and Henkes (2020) lognormal model.

		isoclump.fit_HH20inv
			Fitting function for Tikhonov regularization inversion model of
			Hemingway and Henkes (2020).

		isoclump.fit_PH12
			Fitting function for Passey and Henkes (2012) model.

		isoclump.fit_SE15
			Fitting function for Stolper and Eiler (2015) model.

		Examples
		--------
		Generating a kDistribution instance by fitting some experimental D47
		data contained in a HeatingExperiment object::

			#import packages
			import isoclump as ic

			#assume some he is a HeatingExperiment object
			kd = ic.kDistribution.invert_experiment(
				he,
				model = 'PH12',
				p0 = [-7., 0.5], #passing initial guess for model fit
				thresh = 1e-6 #passing threshold for linear region
				)

		Same as above, but now including the Tikhonov regularization inverse
		fit for 'HH20' model type::

			#assume some he is a HeatingExperiment object
			kd = ic.kDistribution.invert_experiment(
				he,
				model = 'HH20',
				fit_reg = True,
				omega = 'auto', #passing omega value for model fit
				lam_min = -30, #passing lambda bounds
				lam_max = 10
				)	

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

			#running the model in this order properly catches any nonsense
			# kwargs!

			#include regularized data if necessary
			if fit_reg is True:

				#fit the model using the inverse function
				rho_lam_inv, omega, res_inv, rgh_inv = fit_HH20inv(
					he, 
					**kwargs
					)

				#extract fit_HH20 kwargs and run Gaussian fit.
				ars = [k for k, v in inspect.signature(
					fit_HH20).parameters.items()]

				kwa = {k : kwargs[k] for k in dict(kwargs) if k in ars}

				#run Gaussian fit
				params, params_std, rmse, npt, lam, rho_lam = fit_HH20(
					he, 
					**kwa
					)

			else:

				#run Gaussian fit
				params, params_std, rmse, npt, lam, rho_lam = fit_HH20(
					he, 
					**kwargs
					)

				#this model has no rho_lam_inv and associated statistics
				rho_lam_inv = omega = res_inv = rgh_inv = None

		#raise exception if it's not an acceptable string
		elif isinstance(model, str):
			raise ValueError(
				'%s is an invalid model string. Must be one of: "PH12",'
				'"Hea14", "SE15", or "HH20"' % model)

		#raise different exception if it's not a string
		else:

			mdt = type(model).__name__

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

	#define method for plotting HH20 results
	def plot(self, ax = None, lnargs = {}, invargs = {}):
		'''
		Generates a plot of ln(k) distributions for 'HH20'-type models.

		Parameters
		----------
		ax : None or plt.axis
			Axis for plotting results; defaults to `None`.

		lnargs : dict
			Dictionary of stylistic keyword arguments to pass to `plt.plot()`
			when plotting lognormal results. Defaults to empty dict.

		invargs : dict
			Dictionary of stylistic keyword arguments to pass to `plt.plot()`
			when plotting inversion results, if they exist. Defaults to empty
			dict.

		Returns
		-------
		ax : plt.axis
			Updated axis containing results.

		Raises
		------
		ValueError
			If the `kDistribution` instance is of a model type that does not
			support plotting. Currently, only 'HH20' supports plotting.

		See Also
		--------
		matplotlib.pyplot.plot
			Underlying plotting function that is called.

		Examples
		--------
		Basic implementation, assuming `ic.kDistribution` instance `kd` exists
		and is of 'HH20' model type::

			#import modules
			import isoclump as ic
			import matplotlib.pyplot as plt

			#make figure
			fig, ax = plt.subplots(1,1)

			#plot results
			kd.plot(ax = ax)

		Similar implementation, but now putting in stylistic keyword args::

			#import modules
			import isoclump as ic
			import matplotlib.pyplot as plt

			#make figure
			fig, ax = plt.subplots(1,1)

			#define plotting style
			lnargs = {'linewidth':2, 'c':'k'}
			invargs = {'linewidth':1.5, 'c':'g'}

			#plot results
			kd.plot(ax = ax, lnargs = lnargs, invargs = invargs)
		'''

		#check if model is right
		if self.model != 'HH20':
			raise ValueError(
				'Plotting is not implemented for model type %s; only "HH20"'
				' fits can be plotted. Consider extracting k values directly'
				' from summary table instead.' % self.model)

		#make axis if necessary
		if ax is None:
			_, ax = plt.subplots(1,1)

		#plot lognormal data
		ax.plot(
			self.lam,
			self.rho_lam,
			label = 'lognormal fit',
			**lnargs
			)

		#plot inverse data if it exists
		if self.rho_lam_inv is not None:

			#make label
			invlab = r'inverse model fit ($\omega$ = %.2f)' % self.omega

			#plot data
			ax.plot(
				self.lam,
				self.rho_lam_inv,
				label = invlab,
				**invargs
				)

		#set axis labels
		ax.set_xlabel(r'$\lambda$ ($min^{-1}$)')
		ax.set_ylabel(r'$\rho(\lambda)$')

		#add legend
		ax.legend(loc = 'best')

		#return result
		return ax

	#customize __repr__ method for printing summary
	def __repr__(self):
		'''
		Sets how kDistribution is represented when called on the command line.

		Returns
		-------
		summary : str
			String representation of the summary attribute data frame.
		'''
		return str(self.summary)

	#make @property functions to ensure data format
	#ensure params is an acceptable format
	@property
	def params(self):
		return self._params
	
	@params.setter
	def params(self, value):
		#make into np.array
		p = np.array(value)

		#check dtype
		try:
			self._params = p.astype('float')

		except ValueError:
			raise TypeError(
				'Attempting to input params of type %s. Must be array-like'
				' containing int or float values.' % p.dtype.name)

	#ensure model is an acceptable string
	@property
	def model(self):
		return self._model

	@model.setter
	def model(self, value):
		#set value if it closely matches a valid model
		if value in ['Hea14','H14','hea14','Henkes14','Henkes2014','Henkes']:
			self._model = 'Hea14'

		elif value in ['HH20','hh20','Hemingway20','Hemingway2020','Hemingway']:
			self._model = 'HH20'

		elif value in ['PH12','ph12','Passey12','Passey2012','Passey']:
			self._model = 'PH12'

		elif value in ['SE15','se15','Stolper15','Stolper2015','Stolper']:
			self._model = 'SE15'

		#raise exception if it's not an acceptable string
		elif isinstance(value, str):
			raise ValueError(
				'%s is an invalid model string. Must be one of: "PH12",'
				'"Hea14", "SE15", or "HH20"' % value)

		#raise different exception if it's not a string
		else:

			mdt = type(value).__name__

			raise TypeError(
				'Unexpected model of type %s. Must be string.' % mdt)

	#make a summary table
	@property
	def summary(self):
		'''
		Prints a summary of model parameters as a `pandas.DataFrame`.

		Returns
		-------
		resdf : pd.DataFrame
			DataFrame containing summary statistics
		'''

		#extract parameters
		try:
			params = mod_params[self.model]

		except KeyError: #model not in list
			params = None

		#make summary table
		restab = {'model' : self.model,
				  'rmse' : self.rmse,
				  'npt' : self.npt,
				  'params' : params,
				  'mean' : self.params,
				  'std. dev.' : self.params_std
				  }

		resdf = pd.DataFrame(restab)

		return resdf


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