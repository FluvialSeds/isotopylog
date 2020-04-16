'''
This module contains the TimeData superclass and all corresponding subclasses.
'''

#for python 2 compatibility
from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = [
	'HeatingExperiment',
	 # 'GeologicHistory'
	]

#import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import helper functions
from .timedata_helper import(
	_read_csv,
	_cull_data,
	_calc_G,
	)

from .dictionaries import(
	caleqs
	)

class HeatingExperiment(object):
	__doc__='''
	Add docstring here
	'''

	def __init__(self, t, T, calibration = 'Bea17', clumps = 'CO47', d = None,
		d_std = None, dex = None, dex_std = None, iso_params = 'Gonfiantini',
		model = None, ref_frame = 'CDES90', tex = None, T_std = None):
		'''
		Initializes the class
		'''

		#input all attributes
		self.t = t
		self.T = T
		self.calibration = calibration
		self.clumps = clumps
		self.d = d
		self.d_std = d_std
		self.dex = dex
		self.dex_std = dex_std
		self.iso_params = iso_params
		self.model = model
		self.ref_frame = ref_frame
		self.tex = tex
		self.T_std = T_std

		#convert D and Dex to fraction reaction remaining, G, and store
		if dex is not None:
			self.Gex, self.Gex_std = _calc_G(
				calibration,
				clumps, 
				dex, 
				dex_std, 
				ref_frame,
				T
				)

		if d is not None:
			self.G, self.G_std = _calc_G(
				calibration,
				clumps, 
				d, 
				d_std, 
				ref_frame,
				T
				)

	@classmethod
	def from_csv(cls, file, calibration = 'Bea17', culled = True, nt = 300):
		'''
		Imports data from a csv file
		'''

		#import experimental data (note: 'file' can be a DataFrame!)
		clumps, dex, dex_std, iso_params, ref_frame, tex, T = _read_csv(file)

		#cull data if necessary
		if culled:
			dex, dex_std, tex = _cull_data(calibration,
				clumps, 
				dex, 
				dex_std,
				ref_frame,
				T,
				tex)

		#make t and T arrays
		tex_max = np.ceil(np.max(tex)/1000)*1000 #round up to next 1000 minutes
		t = np.linspace(0, tex_max, nt)
		# T = T*np.ones(nt)

		#run __init__ and return instance
		return cls(t, T, calibration = calibration, clumps = clumps, d = None,
			d_std = None, dex = dex, dex_std = dex_std, iso_params = iso_params,
			model = None, ref_frame = ref_frame, tex = tex, T_std = None)


	def forward_model(kdistribution):
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

	def summary(self):
		'''
		Generates and prints a summary of information
		'''

		#make a summary table
		sum_vars = {
			'Clumped System' : self.clumps,
			'T' : self.T,
			'Isotope Parameters' : self.iso_params,
			'Reference Frame' : self.ref_frame,
			'T-D Calibration' : self.calibration
			}

		#make into a table
		sum_table = pd.Series(sum_vars)

		#return table
		return sum_table

