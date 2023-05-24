from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='isotopylog',
	version='0.0.8',
	description='Clumped isotope kinetic analysis',
	long_description=readme(),
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'License :: Free for non-commercial use',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 2.7',
		'Topic :: Scientific/Engineering'
	],
	url='https://github.com/FluvialSeds/isotopylog',
	download_url='https://github.com/FluvialSeds/isotopylog/tarball/0.0.8',
	keywords=[
		'geochemistry',
		'clumped isotopes',
		'kinetics',
		'inverse modeling',
		'carbon cycle'
	],
	author='Jordon D. Hemingway',
	author_email='jordon.hemingway@erdw.ethz.ch',
	license='GNU GPL Version 3',
	packages=['isotopylog'],
	install_requires=[
		'matplotlib',
		'numpy',
		'pandas',
		'scipy'
	],
	# test_suite='nose.collector',
	# tests_require=['nose'],
	include_package_data=True,
	# zip_safe=False
	)