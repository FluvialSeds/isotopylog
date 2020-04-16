'''
This module contains helper functions for the RateData classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_fit_PH12',
			'_fit_Hea14',
			'_fit_SE15',
			'_fit_HH20',
			]

import numpy as np
import pandas as pd

# from .dictionaries import(
# 	caleqs
# 	)


#inverse model fitting functions
def _fit_PH12(heatingexperiment):
	'''
	Add docstring
	'''

	#simple first-order equation, ln(G) = -kt

	#convert to log space

	#loop through retained data points, calculate regression statistics

	#keep all points once RMSE improvement drops below a certain threshhold

	return k, k_std, RMSE
	
def _fit_Hea14(heatingexperiment):

	return k

def _fit_SE15(heatingexperiment):

	return k

def _fit_HH20(heatingexperiment):

	return k, pk


def f(x,m,b):
	return m*x + b

def rmse(y,yhat):
	return np.sqrt(np.sum((y-yhat)**2))