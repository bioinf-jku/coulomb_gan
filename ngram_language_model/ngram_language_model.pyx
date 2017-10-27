# distutils: language=c++
# distutils: sources=ngram_language_model_cpp.cpp
# distutils: extra_compile_args=["-std=c++11"]

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef extern from "ngram_language_model_cpp.h" namespace "language":
  cdef cppclass NgramLanguageModelCPP:
    NgramLanguageModelCPP(int, int) except +
    void add_sample(const vector[int]& sample)
    vector[vector[int]] get_unique_ngrams(int n)
    double js_with(const NgramLanguageModelCPP& other, int ngram_length)
    double log_likelihood(const vector[int]& ngram)
    long int get_memory()

cdef class NgramLanguageModel:
  cdef NgramLanguageModelCPP* _nglm
  def __cinit__(self, samples, int n, int m):
    self._nglm = new NgramLanguageModelCPP(n, m)
    cdef vector[int] v
    for sample in samples:
      v = np.ascontiguousarray(sample, dtype=DTYPE)
      self._nglm.add_sample(v)
      

  def __dealloc__(self):
    del self._nglm

  def unique_ngrams(self, int n):
    if n <= 0:
      raise ValueError('n must be > 0')
    cdef vector[vector[int]] v = self._nglm.get_unique_ngrams(n)
    cdef np.ndarray[DTYPE_t, ndim=2] answer = np.zeros((v.size(),n), dtype=DTYPE)
    cdef int i, j
    for i in range(v.size()):
      for j in range(n):
        answer[i,j] = v[i][j]
    return answer

  def log_likelihood(self, ngram):
    cdef vector[int] v = np.ascontiguousarray(ngram, dtype=DTYPE)
    return self._nglm.log_likelihood(v)

  def js_with(self, NgramLanguageModel other, int ngram_length):
    if ngram_length <= 0:
      raise ValueError('ngram_length must be >= 1')
    #TODO: check bounds on other end
    return self._nglm.js_with((other._nglm)[0], ngram_length)
    
  def get_memory(self):
    return self._nglm.get_memory()



