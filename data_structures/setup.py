from distutils.core import setup, Extension
import numpy as np

module = Extension('algorithms', sources=['algorithms.cpp'])

setup(
    name='Algorithms',
    version='1.0',
    description='Python interface for some "algorithm" library functions',
    include_dirs=[np.get_include()],
    ext_modules=[module]
)