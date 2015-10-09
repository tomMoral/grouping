import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange, threadid

cdef extern from "stdio.h":
	printf(char* string)

DTYPE=np.float
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _lag_mat(np.ndarray[DTYPE_t, ndim=1] X, np.ndarray[DTYPE_t, ndim=2] res, int K):
	assert X.dtype == DTYPE
	cdef unsigned int L = X.shape[0] - K + 1
	cdef Py_ssize_t i, j, l
	cdef float tmp

	with nogil: 
		for i in prange(K, schedule='guided'):
			tmp = 0
			for l in range(L):
				tmp = tmp + X[l]*X[i+l]
			for j in range(K-i):
				res[i+j, j] = tmp/L
				res[j, i+j] = tmp/L
				tmp = tmp + X[L+j]*X[L+i+j] - X[j]*X[i+j]


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _proj_base(np.ndarray[DTYPE_t, ndim=1] sig, np.ndarray[DTYPE_t, ndim=1] base,
			  float vp):
	assert sig.dtype == DTYPE
	assert base.dtype == DTYPE
	cdef int K = base.shape[0]
	cdef int L = sig.shape[0] - K + 1

	cdef unsigned int k, j
	cdef float ac

	cdef np.ndarray[DTYPE_t, ndim=1] a = np.zeros(L, dtype=DTYPE)

	for j in range(L):
		ac = 0
		for k in range(K):
			ac += sig[j+k]*base[k]
		a[j] = ac / vp

	return a


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _heterogeneity(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] base, 
			 np.ndarray[long, ndim=1] I, np.ndarray[double, ndim=1] vp):
	assert a.dtype == DTYPE
	assert base.dtype == DTYPE
	cdef int K = base.shape[0]
	cdef int L = a.shape[1]

	if len(I) == 0:
		I = np.zeros(1, dtype=int)
	cdef int dim = len(I)

	cdef unsigned int i, j
	cdef float current = 0

	cdef np.ndarray[DTYPE_t, ndim=1] M_X = np.zeros([L+K-1], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] X = np.zeros([L+K-1], dtype=DTYPE)
	cdef float c = 0.

	for i in range(K):
		for j in range(L):
			c = 0.
			for k in range(dim):
				c += a[I[k], j] * base[i, I[k]] * vp[I[k]]
			X[i+j] += c
			M_X[i+j] += 1
	for i in range(L+K-1):
		X[i] /= M_X[i]
	return X
