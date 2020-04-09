'''
This module contains helper functions for the TimeData classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_assert_calib',
			'_assert_clumps',
			'_assert_ref_frame',
			]

import numpy as np
import pandas as pd

#define hidden function for asserting calibration is okay
def _assert_calib(clumps, calibration):
	'''
	ADD DOCSTRING
	'''

#define hidden function for asserting clumped isotope system is okay
def _assert_clumps(clumps):
	'''
	ADD DOCSTRING
	'''

#define hidden function for asserting reference frame is okay
def _assert_ref_frame(clumps, ref_frame):
	'''
	ADD DOCSTRING
	'''