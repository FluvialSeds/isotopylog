'''
This module contains the TimeData superclass and all corresponding subclasses.
'''

#import from future for python 2 compatibility
from __future__ import(
	division,
	print_function,
	)

#set magic attributes
__docformat__ = 'restructuredtext en'
__all__ = [
	'HeatingExperiment',
	 # 'GeologicHistory'
	]

#import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import types for checking
from types import LambdaType

#import necessary isoclump timedata helper functions
from .timedata_helper import(
	_calc_G_from_D,
	_read_csv,
	# _cull_data,
	)

#import necessary isoclump dictionaries
from .dictionaries import(
	caleqs,
	clump_isos,
	)


# TODO FRIDAY 24 APRIL:
# * Update cull data function
# * Write forward_model function
# * Write plot function
# * Write docstrings


# RUNNING TODO LIST:
# * 

# DOCUMENTS TODO LIST:
# * ADD PLOT RESULT IMAGES TO NECESSARY DOCSTRINGS
# * ADD LOGO TO BANNER


class HeatingExperiment(object):
	__doc__='''
	Class for inputting, storing, and visualizing clumped isotope heating 
	experiment data. Currently only accepts D47 clumps, but will be expanded 
	in the future as new clumped system data becomes available.

	Parameters
	----------

	dex : array-like
		Array of experimental isotope values, written for each time point as
		[D, d1, d2] where D is the clumped isotope measurement (e.g., D47) and
		d1 and d2 are the corresponding major isotope values, listed from
		lowest to highest amu (e.g., d13C, d18O). Length ``ntex``.

	T : int, float, or array-like
		The equilibrium temperature at which the experiment was performed, in
		Kelvin. If array-like, must be length ``ntex``.

	tex : array-like
		Array of experimental time points, in units of time. While the exact
		time unit is flexible, all subsequent calculations will depend on
		time unit chosen (e.g., if minutes, then rates are inverse minutes).
		Length ``ntex``. 

	calibration : string or LambdaType
		The D-T calibration curve to use, either from the literature or as
		a user-inputted lambda function. If from the literature for D47
		clumps, options are: \n
			``'PH12'``: for Passey and Henkes (2012) Eq. 4 \n
			``'SE15'``: for Stolper and Eiler (2015) Fig. 3 \n
			``'Bea17'``: for Bonifacie et al. (2017) Eq. 2 \n
		If as a lambda function, must have T in Kelvin. Note that literature
		equations will be adjusted to be consistent with any reference frame,
		but lambda functions will be reference-frame-specific.
		Defaults to ``'Bea17'``.

	clumps : string
		The clumped isotope system under consideration. Currently only
		accepts 'CO47' for D47 clumped isotopes, but will include other
		isotope systems as they become more widely used and data become
		available. Defaults to ``'CO47'``.

	d : None or array-like
		Array of forward-modeled isotope values, written for each time point 
		as [D, d1, d2] where D is the clumped isotope measurement (e.g., D47) 
		and d1 and d2 are the corresponding major isotope values, listed from 
		lowest to highest amu (e.g., d13C, d18O). Shape ``nt`` x 3. Defaults  to 
		``None``.

	d_std : None or array-like
		Propagated standard deviation of forward-modeled d values.
		Shape [``nt`` x 3]. Defaults to ``None``.

	dex_std : None or array-like
		Analytical standard deviation of experimentally measured d values.
		Shape [``nt`` x 3]. Defaults to ``None``.

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
			`'Passey'``: for Passey et al. (2014) lam17\n
		Defaults to ``'Gonfiantini'``.

	ref_frame : string
		The reference frame used to calculate clumped isotope data. Options
		are:\n
			``'CDES25'``: Carbion Dioxide Equilibrium Scale acidified at 25 C.
			``'CDES90'``: Carbon Dioxide Equilibrium Scale acidified at 90 C.
			``'Ghosh'``: Heated Gas Line Reference Frame of Ghosh et al. (2006)
			acidified at 25 C.\n
		Defaults to ``'CDES90'``.

	t : None or array-like
		Array of forward-modeled time points, in the same time units as ``tex``.
		Defaults to ``None``.

	T_std : None, int, or float
		The standard deviation of experimental temperature, in Kelvin. Defaults
		to ``None``.

	Raises
	------

	TypeError
		If inputted parameters of an unacceptable type.

	ValueError
		If an unexpected keyword argument is trying to be inputted.

	ValueError
		If an unexpected 'calibration', 'clumps', 'iso_params', or 'ref_frame'
		name is trying to be inputted.

	ValueError
		If the length of inputted experimental isotope data and time arrays do
		not match.

	Notes
	-----

	If inputted ``T`` is array-like, then the average and standard deviation
	will be extracted and stored.

	If ``clumps = 'CO47'``, then all calculations assume mass-dependent d17O.

	If ``clumps = 'CO47'``, then inputted d18O data must be in permille
	relative to VPDB, not VSMOW.

	See Also
	--------

	isoclump.kDistribution
		The class for extracting and visualizing rate data from a given 
		``HeatingExperiment`` instance.

	isoclump.GeologicHistory
		The class for forward-modeling kinetic information onto a geologic
		time-temperature history.

	Examples
	--------

	Generating a bare-bones HeatingExperiment instance without fitting any
	actual data::

		#import packages
		import isoclump as ic
		import numpy as np

		#make arbitrary dex, T, and tex
		dex = np.ones(4,4)
		tex = np.arange(0,4)
		T = 450 + 273.15 #get to K

		#make instance
		he = ic.HeatingExperiment(dex, T, tex)

	Generating a HeatingExperiment instance by extracting data from a csv
	file::

		#setting a string with the file name
		file = 'string_with_file_name.csv'

		#make HeatingExperiment instance without culling data
		he = ic.HeatingExperiment.from_csv(file, culled = False)

		#or, cull the data that are too close to equilibrium (see PH12)
		he = ic.HeatingExperiment.from_csv(file, culled = True)

	Forward modeling some rate data and visualizing model results::


	References
	----------

	[1] Craig (1957) *Geochim. Cosmochim. Ac.*, **12**, 133--149.\n
	[2] Li et al. (1988) *Chin. Sci. Bull.*, **33**, 1610--1613.\n
	[3] Chang and Li (1990) *Chin. Sci. Bull.*, **35**, 290.\n
	[4] Gonfiantini (1995) *IAEA Technical Report*, 825.\n
	[5] Assonov and Brenninkmeijer (2003) *Rapid Comm. Mass Spec.*, **17**, 
	1017--1029.\n
	[6] Barkan and Luz (2005) *Rapid Comm. Mass Spec.*, **19**, 3737--3742.\n
	[7] Ghosh et al. (2006) *Geochim. Cosmochim. Ac.*, **70**, 1439--1456.\n
	[8] Brand (2010) *Pure Appl. Chem.*, **82**, 1719--1733.\n
	[9] Dennis et al. (2011) *Geochim. Cosmochim. Ac.*, **75**, 7117--7131.\n
	[10] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.\n
	[11] Passey et al. (2014) *Geochim. Cosmochim. Ac.*, **141**, 1--25.\n
	[12] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.\n
	[13] Daëron et al. (2016) *Chem. Geol.*, **442**, 83--96.\n
	[14] Bonifacie et al. (2017) *Geochim. Cosmochim. Ac.*, **200**, 255--279.
	'''

	#define all the possible attributes for __init__ using _kwattrs
	_kwattrs = {
		'calibration' : 'Bea17', 
		'clumps' : 'CO47', 
		'd' : None, 
		'd_std' : None,
		'dex_std' : None,
		'iso_params' : 'Gonfiantini',
		'ref_frame' : 'CDES90',
		't' : None,
		'T_std' : None,
		}

	#define magic methods
	#initialize the object
	def __init__(self, dex, T, tex, **kwargs):
		'''
		Initilizes the object.

		Returns
		-------
		he : isoclump.HeatingExperiment
			The ``HeatingExperiment`` object.
		'''

		#first make everything in _kwattrs equal to its default value
		for k, v in self._kwattrs.items():
			setattr(self, k, v)

		#then, set arguments
		self.tex = tex #tex first since dex setter will check length
		self.dex = dex
		self.T = T

		#finally, overwrite all attributes in kwargs and raise exception if
		# unknown
		for k, v in kwargs.items():
			if k in self._kwattrs:
				setattr(self, k, v)

			else:
				raise ValueError(
					'__init__() got an unexpected keyword argument %s' % k)

		#convert Dex to fraction remaining, Gex, and store
		self.Gex, self.Gex_std = _calc_G_from_D(
			self.dex,
			self.T,
			calibration = self.calibration,
			clumps = self.clumps,
			d_std = self.dex_std,
			ref_frame = self.ref_frame,
			)

		#if d exists, convert D to fraction remaining, G, and store
		if self.d is not None:
			self.G, self.G_std = _calc_G_from_D(
				self.d,
				self.T,
				calibration = self.calibration,
				clumps = self.clumps,
				d_std = self.d_std,
				ref_frame = self.ref_frame,
				)

	#customize __repr__ method for printing summary
	def __repr__(self):
		'''
		Sets how HeatingExperiment is represented when called on the command
		line.

		Returns
		-------

		summary : str
			String representation of the summary attribute data frame.
		'''

		#get T uncertainty
		try:
			Tstd = '%.2f' % self.T_std

		except TypeError:
			Tstd = 'None'

		attrs = {'calibration' : self.calibration,
		 		 'clumps' : self.clumps,
		 		 'iso_params' : self.iso_params,
		 		 'ref_frame' : self.ref_frame,
		 		 'T' : str(self.T) + '+/-' + Tstd
		 		}

		s = pd.Series(attrs)

		return str(s)

	#define @classmethods
	#method for generating HeatingExperiment instance from csv file 
	@classmethod
	def from_csv(cls, file, culled = True):
		'''
		Imports data from a csv file and creates a HeatingExperiment object
		from those data.

		Parameters
		----------

		Returns
		-------

		Raises
		------

		See Also
		--------

		Examples
		--------

		References
		----------
		'''

		#import experimental data
		dex, T, tex, file_attrs = _read_csv(file)

		#cull data if necessary

		#TODO: Update cull function
		if culled is True:

			dex, dex_std, tex = _cull_data(calibration,
				clumps, 
				dex, 
				dex_std,
				ref_frame,
				T,
				tex)

		#return class instance
		return cls(dex, T, tex, **file_attrs)

	def forward_model(kd):
		'''
		Forward models a given kDistribution instance to produce predicted
		evolution.
		'''

		#check which model and run it forward
		if kdistribution.model == 'PH12':

			d, d_std = _forward_PH12(kdistribution)

		elif kdistribution.model == 'Hea14':

			d, d_std = _forward_Hea14(kdistribution)

		elif kdistribution.model == 'SE15':

			d, d_std = _forward_SE15(kdistribution)

		elif kdistribution.model == 'HH20':

			d, d_std = _forward_HH20(kdistribution)

		#store attributes
		self.d = d
		self.d_std = d_std
		self.model = kdistribution.model

	def plot(self, ax = None, yaxis = 'D', logy = False, **kwargs):
		'''
		Plots results
		'''

		#make axis if necessary
		if ax is None:
			_, ax = plt.subplots(1,1)

		#plot experimental data
		if self.dex is not None:

			#get the right y data
			if yaxis == 'D':
				yex = self.dex[:,0]
				yex_std = self.dex_std[:,0]

			elif yaxis == 'G':
				yex = self.Gex
				yex_std = self.Gex_std

			ax.errorbar(self.tex, yex,
				yerr = yex_std,
				fmt = 'o',
				c = 'k',
				markeredgecolor = 'w',
				markersize = 12
				)

		#plot modeled data
		if self.d is not None:

			#get the right y data
			if yaxis == 'D':
				y = self.d[:,0]

			elif yaxis == 'G':
				y = self.G

			ax.plot(self.t, y, **kwargs)

		#add dashed line at Deq
		Deq = caleqs[self.calibration][self.ref_frame](self.T)

		if yaxis == 'D':
			ax.plot([0, self.t[-1]], [Deq, Deq],
				':r',
				linewidth = 2
				)

		elif yaxis == 'G':
			ax.plot([0, self.t[-1]], [0,0],
				':r',
				linewidth = 2
				)

		#return result
		return ax

	
	#TODO: ADD METHODS FOR CHANGING ISO_PARAMS AND REF_FRAME

	#define @property getters and setters
	@property
	def caleq(self):
		'''
		The lambda equation used for calculating T-D relationships.
		'''
		#if calibration is custom, extract directly
		if self.calibration == 'Custom':
			eq = self._caleq

		#else, extract it from the dictionary
		else:
			eq = caleqs[self.calibration][self.ref_frame]

		return eq

	@property
	def calibration(self):
		'''
		The T-D calibration equation to be used for modeling data.
		'''
		return self._calibration
	
	@calibration.setter
	def calibration(self, value):
		'''
		Setter for calibration
		'''
		#set vaue if it closely matches a valid calibration
		if value in ['Bea17','BEA17','Bonifacie','bonifacie','Bonifacie17']:
			self._calibration = 'Bea17'

		elif value in ['PH12','ph12','Passey12','Passey2012','Passey']:
			self._calibration = 'PH12'

		elif value in ['SE15','se15','Stolper15','Stolper2015','Stolper']:
			self._calibration = 'SE15'

		#raise exception if it's not an acceptable string
		elif isinstance(value, str):
			raise ValueError(
				'%s is an invalid T-D calibration. Must be one of: "Bea17",'
				'"PH12", or "SE15"' % value)

		#if it's a lambda function, store appropriately
		elif isinstance(value, LambdaType):
			self._calibration = 'Custom'
			self._caleq = value

		#raise different exception if it's not a string
		else:

			mdt = type(value).__name__

			raise TypeError(
				'Unexpected calibration of type %s. Must be string.' % mdt)

	@property
	def clumps(self):
		'''
		The clumped isotope system associated with a given experiment.
		'''
		return self._clumps

	@clumps.setter
	def clumps(self, value):
		'''
		Setter for clumps
		'''
		#ensure CO47
		if value in ['CO47','co47','Co47','D47']:
			self._clumps = 'CO47'

		#raise exception if it's not an acceptable string
		elif isinstance(value, str):
			raise ValueError(
				'%s is an invalid clumped isotope system. Currently, only'
				' accepts "CO47"' % value)

		#raise different exception if it's not a string
		else:

			mdt = type(value).__name__

			raise TypeError(
				'Unexpected clumps of type %s. Must be string.' % mdt)
	
	@property
	def d(self):
		'''
		Array containing the forward-modeled isotope data.
		'''
		return self._d
	
	@d.setter
	def d(self, value):
		'''
		Setter for d
		'''
		self._d = value

	@property
	def d_std(self):
		'''
		Array containing the forward-modeled isotope data uncertainty, as
		+/- 1 sigma.
		'''
		return self._d_std
	
	@d_std.setter
	def d_std(self, value):
		'''
		Setter for d_std
		'''
		self._d_std = value

	@property
	def dex(self):
		'''
		Array containing the measured experimental isotope data.
		'''
		return self._dex
	
	@dex.setter
	def dex(self, value):
		'''
		Setter for dex
		'''
		#check that length is right
		ntex = len(self.tex)
		ndex = len(value)

		if ndex == ntex:
			self._dex = value

		else:
			raise ValueError(
				'cannot broadcast tex of length %s and dex of length %s'
				% (ntex, ndex))

	@property
	def dex_std(self):
		'''
		Array containing the measured experimental data uncertainty, as
		+/- 1 sigma.
		'''
		return self._dex_std
	
	@dex_std.setter
	def dex_std(self, value):
		'''
		Setter for dex_std
		'''
		self._dex_std = value

	@property
	def G(self):
		'''
		Array containing the forward-modeled raction progress data.
		'''
		return self._G
	
	@G.setter
	def G(self, value):
		'''
		Setter for G
		'''
		self._G = value

	@property
	def G_std(self):
		'''
		Array containing the forward-modeled reaction progress uncertainty, as
		+/- 1 sigma.
		'''
		return self._G_std
	
	@G_std.setter
	def G_std(self, value):
		'''
		Setter for G_std
		'''
		self._G_std = value

	@property
	def Gex(self):
		'''
		Array containing the measured experimental reaction progress data.
		'''
		return self._Gex
	
	@Gex.setter
	def Gex(self, value):
		'''
		Setter for Gex
		'''
		self._Gex = value

	@property
	def Gex_std(self):
		'''
		Array containing the measured experimental reaction progress
		uncertainty, as +/- 1 sigma.
		'''
		return self._Gex_std
	
	@Gex_std.setter
	def Gex_std(self, value):
		'''
		Setter for Gex_std
		'''
		self._Gex_std = value

	@property
	def iso_params(self):
		'''
		The isotope parameters used for calculating clumped values.
		'''
		return self._iso_params

	@iso_params.setter
	def iso_params(self, value):
		'''
		Setter for iso_params
		'''
		#set value if it closely matches a valid parameter
		if value in ['Barkan','barkan']:
			self._iso_params = 'Barkan'

		elif value in ['Brand','brand','Chang + Assonov','Chang+Assonov']:
			self._iso_params = 'Brand'

		elif value in ['Chang + Li','Chang+Li','chang + li','chang+li']:
			self._iso_params = 'Chang + Li'

		elif value in ['Craig + Assonov','Craig+Assonov','craig + assonov']:
			self._iso_params = 'Craig + Assonov'

		elif value in ['Craig + Li','Craig+Li','craig + li','craig+li']:
			self._iso_params = 'Craig + Li'

		elif value in ['Gonfiantini','gonfiantini']:
			self._iso_params = 'Gonfiantini'

		elif value in ['Passey','passey']:
			self._iso_params = 'Passey'

		#raise exception if it's not an acceptable string
		elif isinstance(value, str):
			raise ValueError(
				'%s is an invalid iso_params. Must be one of: "Barkan", "Brand"'
				' "Chang+Li", "Craig+Assonov", "Craig+Li", "Gonfiantini",'
				' or "Passey".' % value)

		#raise different exception if it's not a string
		else:

			mdt = type(value).__name__

			raise TypeError(
				'Unexpected iso_params of type %s. Must be string.' % mdt)
	
	@property
	def ref_frame(self):
		'''
		The reference frame being used for experimental isotope data.
		'''
		return self._ref_frame

	@ref_frame.setter
	def ref_frame(self, value):
		'''
		Setter for ref_frame
		'''
		#set value if it closely matches a valid parameter
		if value in ['CDES25','Cdes25','cdes25']:
			self._ref_frame = 'CDES25'

		elif value in ['CDES90','Cdes90','cdes90']:
			self._ref_frame = 'CDES90'

		elif value in ['Ghosh25','ghosh25','ghosh','Ghosh']:
			self._ref_frame = 'Ghosh25'

		#raise exception if it's not an acceptable string
		elif isinstance(value, str):
			raise ValueError(
				'%s is an invalid ref_frame. Must be one of: "CDES25",'
				' "CDES90", or "Ghosh25".' % value)

		#raise different exception if it's not a string
		else:

			mdt = type(value).__name__

			raise TypeError(
				'Unexpected ref_frame of type %s. Must be string.' % mdt)
	
	@property
	def summary(self):
		'''
		DataFrame containing all the summary data.
		'''

		#extract parameters
		isos = clump_isos[self.clumps]
		iso_stds = [i + '_std' for i in isos]

		#make DataFrame of experimental data
		a = pd.DataFrame(
			self.dex,
			index = self.tex,
			columns = isos
			)

		a.index.name = 'tex'

		#make DataFrame of experimental data uncertainty
		b = pd.DataFrame(
			self.dex_std,
			index = self.tex,
			columns = iso_stds
			)

		b.index.name = 'tex'

		#combine into single DataFrame
		resdf = pd.concat([a,b], axis = 1)

		return resdf
	
	@property
	def t(self):
		'''
		Array of forward-modeled time, in same units as ``tex``.
		'''
		return self._t
	
	@t.setter
	def t(self, value):
		'''
		Setter for t
		'''
		self._t = value

	@property
	def T(self):
		'''
		The experimental temperature, in Kelvin
		'''
		return self._T

	@T.setter
	def T(self, value):
		'''
		Setter for T
		'''

		#if T is float or int, store it as is
		if type(value) in [int, float]:
			self._T = value

		#else if T is array-like, calculate mean and std dev
		elif hasattr(value, '__iter__') and not isinstance(value, str):
			self._T = np.mean(value)
			self._T_std = np.std(value)

		#raise error if some other type
		else:
			mdt = type(value).__name__

			raise TypeError(
				'Unexpected T of type %s. Must be int, float, or array-like.'
				% mdt)

	@property
	def T_std(self):
		'''
		The uncertainty on experimental temperature.
		'''
		return self._T_std

	@T_std.setter
	def T_std(self, value):
		'''
		Setter for T_std
		'''
		self._T_std = value
	


