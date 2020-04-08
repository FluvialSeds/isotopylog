'''
This package contains all the exceptions for the ``isoclump`` package.
'''

#define a core exception class for subclassing
class ciException(Exception):
	'''
	Root Exception class for the rampedpyrox package. Do not call directly.
	'''
	pass

class ArrayError(ciException):
	'''
	Array-like object is not in the right form (e.g. strings).
	'''
	pass


class FileError(ciException):
	'''
	If a file does not contain the correct data.
	'''
	pass


class LengthError(ciException):
	'''
	Length of array is not what it should be.
	'''
	pass


class ScalarError(ciException):
	'''
	If something is not a scalar.
	'''
	pass


class StringError(ciException):
	'''
	If a string is not right.
	'''
	pass