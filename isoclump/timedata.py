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
from .exceptions import(
	LengthError,
	)

# #import helper functions
from .core_functions import(
	assert_len,
	calc_f,
	)

# from .plotting_helper import(
# 	)

# from .summary_helper import(
# 	)

# from .ratedata_helper import(
# 	)

from .timedata_helper import (
	_assert_calib,
	_assert_clumps,
	_assert_ref_frame,
	)


class TimeData(object):
	'''
	Class to store time-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, t, T, calibration = 'PH12', clumps = 'CO47', d = None,
		d_std = None, ref_frame = 'CDES90', T_std = None):
		'''
		Initializes the superclass.

		Parameters
		----------
		t : array-like
			Array of forward-modeled time points, in seconds. Length `nt`.

		T : scalar or array-like
			Array of forward-modeled temperature values, in Kelvin. Length
			`nt`.

		calibration : string or lambda function
			The D-T calibration curve to use, either from the literature or as
			a user-inputted lambda function. If from the literature for D47
			clumps, options are:

				'PH12': for Passey and Henkes (2012) Eq. 4
				'SE15': for Stolper and Eiler (2015) Fig. 3
				'Bea17': for Bonifacie et al. (2017) Eq. 2

			If as a lambda function, must have T in Kelvin. Note that literature
			equations will be adjusted to be consistent with any reference frame,
			but lambda functions will be reference-frame-specific.
			Defaults to 'PH12'.

		clumps : string
			The clumped isotope system under consideration. Currently only
			accepts 'CO47' for D47 clumped isotopes, but will include other
			isotope systems as they become more widely used and data become
			available. Defaults to 'CO47'.

		d : None or array-like
			Array of forward-modeled isotope values, written for each time
			point as [D, d1, d2] where D is the clumped isotope measurement
			(e.g., D47) and d1 and d2 are the corresponding major isotope
			values, listed from lowest to highest a.m.u. (e.g., d13C, d18O).
			Note, for 'CO47', d17O is assumed to be mass-dependent.
			Shape `nt` x 3. Defaults to `None`.

		d_std : None or array-like
			Propagated standard deviation of forward-modeled d values.
			Shape `nt` x 3. Defaults to `None`.

		ref_frame : string
			Reference frame to use, from the literature. If for 'CO47' clumps,
			options are:

				'CDES90': for Carbon Dioxide Equilibrium Scale at 90 C (Dennis
					et al. 2011).
				'Ghosh': for the reference frame of Ghosh et al. (2006)

			Defaults to 'CDES90'.

		T_std : None, scalar, or array-like
			Standard deviation of forward-modeled temperature values. Length
			`nt`. Defaults to `None`.

		Warnings
		--------

		Raises
		------

		'''

		#check and store time-temperature attributes
		nt = len(t)
		self.nt = nt
		self.t = assert_len(t, nt) #s
		self.T = assert_len(T, nt) #K

		#check and store property values (calibration, clumps, ref_frame)
		self.clumps = _assert_clumps(clumps)
		self.calibration = _assert_calib(calibration)
		self.ref_frame = _assert_ref_frame(ref_frame)

		#check and store isotope data
		if d is not None:
			self.d = assert_len(d, nt)
		else:
			self.d = None

		if d_std is not None:
			self.d_std = assert_len(d_std, nt)
		else:
			self.d_std = None

		#calculate derived attributes
		self.f, self.f_std = calc_f(
			self.d,
			clumps = clumps,
			d_std = self.d_std,
			ref_frame = ref_frame
			)

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
	t : array-like
		Array of forward-modeled time points, in seconds. Length `nt`.

	T : scalar or array-like
		Array of forward-modeled temperature values, in Kelvin. Length `nt`.

	calibration : string or lambda function
		The D-T calibration curve to use, either from the literature or as
		a user-inputted lambda function. If from the literature for D47
		clumps, options are:

			'PH12': for Passey and Henkes (2012) Eq. 4
			'SE15': for Stolper and Eiler (2015) Fig. 3
			'Bea17': for Bonifacie et al. (2017) Eq. 2

		If as a lambda function, must have T in Kelvin. Note that literature
		equations will be adjusted to be consistent with any reference frame,
		but lambda functions will be reference-frame-specific.
		Defaults to 'PH12'.

	clumps : string
		The clumped isotope system under consideration. Currently only
		accepts 'CO47' for D47 clumped isotopes, but will include other
		isotope systems as they become more widely used and data become
		available. Defaults to 'CO47'.

	d : None or array-like
		Array of forward-modeled isotope values, written for each time
		point as [D, d1, d2] where D is the clumped isotope measurement
		(e.g., D47) and d1 and d2 are the corresponding major isotope
		values, listed from lowest to highest a.m.u. (e.g., d13C, d18O).
		Note, for 'CO47', d17O is assumed to be mass-dependent.
		Shape `nt` x 3. Defaults to `None`.

	d_std : None or array-like
		Propagated standard deviation of forward-modeled d values.
		Shape `nt` x 3. Defaults to `None`.

	dex : None or array-like
		Array of experimental isotope values, written for each time point as
		[D, d1, d2] where D is the clumped isotope measurement (e.g., D47) and
		d1 and d2 are the corresponding major isotope values, listed from
		lowest to highest a.m.u. (e.g., d13C, d18O). Note, for 'CO47', d17O is
		assumed to be mass-dependent. Shape `ntex` x 3. Defaults to `None`. 

	dex_std : None or array-like
		Analytical standard deviation of experimental d values. Shape `ntex` x
		3. Defaults to `None`.

	ref_frame : string
		Reference frame to use, from the literature. If for 'CO47' clumps,
		options are:

			'CDES90': for Carbon Dioxide Equilibrium Scale at 90 C (Dennis
				et al. 2011).
			'Ghosh': for the reference frame of Ghosh et al. (2006)

		Defaults to 'CDES90'.

	tex : None or array-like
		Array of experimental time points, in seconds. Length `ntex`.
		Defaults to `None`.

	T_std : None, scalar, or array-like
		Standard deviation of forward-modeled temperature values. Length
		`nt`. Defaults to `None`.

	Raises
	------
	LengthError
		If experimental data only contain a single time point. Cannot fit any
		model if n < 2.

	TypeError
		If attempting to input experimental delta values but experimental time
		is nonetype (cannot have data with no timestamp).

	Warnings
	--------
	UserWarning
		If attempting to use non-isothermal temperature data to create a
		``HeatingExperiment`` instance. Currently, all experiments are assumed
		isothermal; consider creating a ``GeologicHistory`` instance for
		non-isothermal time-temperature histories. Also warns if T is ``None``.

	UserWarning
		If experimental data do not contain at least 3 unique points (cannot
		fit any model other than "PH12" if n < 3).

	Notes
	-----

	See Also
	--------

	Examples
	--------

	**Attributes**

	References
	----------
	[1] Ghosh et al. (2006) *Geochim. Cosmochim. Ac*, **70**, 1439--1456.
	[2] Dennis et al. (2011) *Geochim. Cosmochim. Ac*, **75**, 7117--7131.
	[3] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
	[4] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
	[5] Bonifacie et al. (2017) *Geochim. Cosmochim. Ac*, **200**, 255--279.
	'''

	def __init__(self, t, T, calibration = 'PH12', clumps = 'CO47', d = None,
		d_std = None, dex = None, dex_std = None, ref_frame = 'CDES90', 
		tex = None, T_std = None):

		#warn if T is not isothermal (all heating experiments must be for now)
		try:
			it = iter(T)

		except TypeError:
			#not iterable
			if not isinstance(T, (int, float)):
				warnings.warn(
					'T must be int, float, or isothermal array-like. Consider'
					' using a ``GeologicHistory`` instance for non-isothermal'
					' data.')
		
		else:
			#iterable
			if len(set(T)) != 1:
				warnings.warn(
					'T must be int, float, or isothermal array-like. Consider'
					' using a ``GeologicHistory`` instance for non-isothermal'
					' data.')

		#call superclass __init__ function
		super(HeatingExperiment, self).__init__(
			t,
			T,
			calibration = calibration,
			clumps = clumps,
			d = d,
			d_std = d_std,
			ref_frame = ref_frame,
			T_std = None) #force to None for HeatingExperiments

		#do additional steps:
		
		#check dex, dex_std, and tex lengths and dtypes; add to self
		if tex is not None:

			try:
				it = iter(tex)

			except TypeError:
				#not iterable; tex is scalar
				raise LengthError(
					'tex must be an array of length >1')

			#warn if length < 3
			if len(set(tex) < 3):
				warnings.warn(
					'Attempting to input experimental data with fewer than'
					' three data points. Must have at least three data points'
					' to generate a meaningful fit for any model other than'
					' "PH12"')

		#raise exception if trying to input dex but no tex
		elif tex is None and dex is not None:

			raise TypeError(
				'Cannot input dex data if tex is nonetype. Add tex data.')


		#store attributes if all are None
		if tex is None:
			self.ntex = None
			self.tex = None
			self.dex = None
			self.dex_std = None
			self.f = None
			self.f_std = None

		#store attributes if tex and dex are not None
		else:
			ntex = len(tex)
			self.ntex = ntex
			self.tex = assert_len(tex, ntex)
			self.dex = assert_len(dex, ntex)

			#include dex_std if it's not None
			if dex_std is not None:
				self.dex_std = assert_len(dex_std, ntex)

			#now include if it is None
			else:
				self.dex_std = None

			#finally, calculate derived fractional abundances
			self.fex, self.fex_std = calc_f(
				self.dex,
				clumps = clumps,
				d_std = self.dex_std,
				ref_frame = ref_frame
				)
	
	#define classmethod to import from csv file
	@classmethod
	def from_csv(cls, file, calibration = 'PH12', clumps = 'CO47',
		culled = True, ref_frame = 'CDES90'):
		'''
		Bar
		'''
		f = file

	#define method to change calibration
	def change_calibration(self, calibration):
		'''
		Baz
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
	t : array-like
		Array of forward-modeled time points, in seconds. Length `nt`.

	T : scalar or array-like
		Array of forward-modeled temperature values, in Kelvin. Length `nt`.

	calibration : string or lambda function
		The D-T calibration curve to use, either from the literature or as
		a user-inputted lambda function. If from the literature for D47
		clumps, options are:

			'PH12': for Passey and Henkes (2012) Eq. 4
			'SE15': for Stolper and Eiler (2015) Fig. 3
			'Bea17': for Bonifacie et al. (2017) Eq. 2

		If as a lambda function, must have T in Kelvin. Note that literature
		equations will be adjusted to be consistent with any reference frame,
		but lambda functions will be reference-frame-specific.
		Defaults to 'PH12'.

	clumps : string
		The clumped isotope system under consideration. Currently only
		accepts 'CO47' for D47 clumped isotopes, but will include other
		isotope systems as they become more widely used and data become
		available. Defaults to 'CO47'.

	d : None or array-like
		Array of forward-modeled isotope values, written for each time
		point as [D, d1, d2] where D is the clumped isotope measurement
		(e.g., D47) and d1 and d2 are the corresponding major isotope
		values, listed from lowest to highest a.m.u. (e.g., d13C, d18O).
		Note, for 'CO47', d17O is assumed to be mass-dependent.
		Shape `nt` x 3. Defaults to `None`.

	d_std : None or array-like
		Propagated standard deviation of forward-modeled d values.
		Shape `nt` x 3. Defaults to `None`.

	ref_frame : string
		Reference frame to use, from the literature. If for 'CO47' clumps,
		options are:

			'CDES90': for Carbon Dioxide Equilibrium Scale at 90 C (Dennis
				et al. 2011).
			'Ghosh': for the reference frame of Ghosh et al. (2006)

		Defaults to 'CDES90'.

	T_std : None, scalar, or array-like
		Standard deviation of forward-modeled temperature values. Length
		`nt`. Defaults to `None`.

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
	[1] Ghosh et al. (2006) *Geochim. Cosmochim. Ac*, **70**, 1439--1456.
	[2] Dennis et al. (2011) *Geochim. Cosmochim. Ac*, **75**, 7117--7131.
	[3] Passey and Henkes (2012) *Earth Planet. Sci. Lett.*, **351**, 223--236.
	[4] Stolper and Eiler (2015) *Am. J. Sci.*, **315**, 363--411.
	[5] Bonifacie et al. (2017) *Geochim. Cosmochim. Ac*, **200**, 255--279.
	'''

	def __init__(self, t, T, calibration = 'PH12', clumps = 'CO47', d = None,
		d_std = None, ref_frame = 'CDES90', T_std = None):

		#call superclass __init__ function
		super(GeologicHistory, self).__init__(
			t,
			T,
			calibration = calibration,
			clumps = clumps,
			d = d,
			d_std = d_std,
			ref_frame = ref_frame,
			T_std = None) #force to None for GeologicHistory

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







