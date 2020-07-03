'''
isotopylog was created as a supplement to Hemingway and Henkes (2020) "A 
distributed activation energy model for clumped isotope bond reordering in
carbonates", *Earth and Planetary Science Letters*, **in review**, pages. It was
created by:

	Jordon D. Hemingway
	Postdoctoral Fellow, Harvard University
	jordon_hemingway@fas.harvard.edu

source code can be found at:
	
	https://github.com/FluvialSeds/isotopylog

documentation can be found at:

	http://isotopylog.readthedocs.io

Version 0.0.4 is current as of 1 July 2020 and reflects the notation used
in Hemingway and Henkes (2020).
'''

from __future__ import(
	division,
	print_function,
	)

__version__ = '0.0.4'

__docformat__ = 'restructuredtext en'


#import timedata classes
from .timedata import(
	HeatingExperiment,
	)

#import ratedata classes
from .ratedata import(
	kDistribution,
	EDistribution,
	)

#import package-level functions:
from .calc_funcs import(
	Deq_from_T,
	T_from_Deq,
	)

from .core_functions import(
	derivatize,
	geologic_history,
	)

from .ratedata_helper import(
	calc_L_curve,
	fit_Arrhenius,
	fit_Hea14,
	fit_HH20,
	fit_HH20inv,
	fit_PH12,
	fit_SE15
	)
