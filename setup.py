from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("FD_SDK.py", language_level="3")
)
