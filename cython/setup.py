from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("dist.pyx")
)
# usage: python setup.py build_ext --inplace
