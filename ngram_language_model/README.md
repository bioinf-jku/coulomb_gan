Written by Calvin Seward (calvin.seward@zalando.de)

This project defines a python and C++ API to the class NgramLanguageModel which saves ngrams efficiently and can quickly calculate statistics (like jensen shannon distance between two models). It's written in C++ for speed and has a convienient python wrapper for convienience. To compile, simply run:

python setup.py build_ext --inplace

set the python path to find the .so file, and import normally
