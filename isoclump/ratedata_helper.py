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

from scipy.optimize import curve_fit

# from .dictionaries import(
# 	caleqs
# 	)


#inverse model fitting functions
def _fit_PH12(he, thresh):
	'''
	Add docstring
	'''

	#simple first-order equation, ln(G) = -kt

	#convert to log space
	x = he.tex
	y = np.log(he.Gex)
	y_std = (np.log(he.Gex + he.Gex_std) - np.log(he.Gex - he.Gex_std))/2

	#loop through retained data points, calculate regression statistics
	# always retain at least 2 data points
	niter = len(x) - 1

	rmse_old = 1 #initialize "old" rmse value

	for i in range(niter):

		#sequentally drop early points
		xt = x[i:]
		yt = y[i:]
		yt_std = y_std[i:]

		#fit the data
		p0 = [1,1] #initial guess
		p, pcov = curve_fit(_flin, xt, yt, p0, 
			sigma = yt_std, 
			absolute_sigma = True
			)

		#calculate yhat an rmse
		yhat = _flin(xt, *p)
		rmse = _calc_rmse(yt, yhat)

		#check how much rmse has improved; break if below threshold
		if np.abs(rmse - rmse_old) < thresh:
			break

		else:
			rmse_old = rmse

	#calculate values to export
	k = p
	k[0] = -k[0] #since slope is -k
	k_std = np.diag(pcov)**0.5

	return k, k_std, rmse
	
def _fit_Hea14(heatingexperiment):

	return k

def _fit_SE15(heatingexperiment):

	return k

def _fit_HH20(heatingexperiment):

	return k, pk


#linear function for curve fitting
def _flin(x, m, b):
	return m*x + b

#rmse function for curve fitting
def _calc_rmse(y,yhat):
	return np.sqrt(np.sum((y-yhat)**2))