'''
This module contains isoclump timedata module tests
'''

import numpy as np
import os
import pandas as pd

import isoclump as ic

from nose.tools import(
	assert_almost_equal,
	assert_equal,
	assert_is_instance,
	assert_raises,
	assert_warms,
	)

from isoclump.timedata_helper import(
	)

from isoclump.exceptions import(
	)

