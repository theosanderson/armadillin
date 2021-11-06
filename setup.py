from setuptools import find_packages, setup
#import numpy as np
from Cython.Build import cythonize



if __name__ == "__main__":
    setup(ext_modules=cythonize("src/armadillin/*.pyx"))
