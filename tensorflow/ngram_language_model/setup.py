from distutils.core import setup
from Cython.Build import cythonize

# run python3 setup.py build_ext --inplace
setup(
  name = 'ngram language model',
  ext_modules = cythonize("ngram_language_model.pyx", language="c++"),
)
