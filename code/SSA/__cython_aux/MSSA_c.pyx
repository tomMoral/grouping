import numpy as np
cimport numpy as np

cimport cython

DTYPE=np.float
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _lag_mat(np.ndarray[DTYPE_t, ndim=2] X, unsigned int K):
    assert X.dtype == DTYPE
    cdef unsigned int N = X.shape[0] - K +1
    cdef unsigned int M = X.shape[1]

    cdef unsigned int i, j, n, m
    cdef float current = 0

    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros([K, K], dtype=DTYPE)
    for i in range(K):
        for j in range(i, K):
            current = 0
            for n in range(N):
                for m in range(M):
                    current += X[i+n, m]*X[j+n, m]
            res[i, j] = current
            res[j, i] = current
    return res



@cython.boundscheck(False) # turn of bounds-checking for entire function
def _lag_mat_mvar(np.ndarray[DTYPE_t, ndim=2] X, unsigned int K):
    assert X.dtype == DTYPE
    cdef unsigned int N = X.shape[0] - K +1
    cdef unsigned int M = X.shape[1]

    cdef unsigned int i, j, l, k
    cdef float current = 0

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.zeros([K, K,  M], dtype=DTYPE)
    for i in range(K):
        for j in range(i, K):
            for k in range(M):
                current = 0
                for l in range(N):
                    current += X[l+i, k]*X[l+j, k]
                res[i, j, k] = current
                res[j, i, k] = current
    return res


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _x1(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=2] U):
    cdef unsigned int i, j, l, k
    cdef unsigned int L = A.shape[0] 
    cdef unsigned int M = A.shape[1]
    cdef unsigned int N = A.shape[2]
    cdef unsigned int K = U.shape[0]
    cdef unsigned int L1 = U.shape[1]
    assert A.dtype == DTYPE
    assert L == L1
    cdef float current = 0

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.zeros([K, M, N], dtype=DTYPE)
    for k in range(K):
        for m in range(M):
            for n in range(N):
                current = 0
                for l in range(L):
                    current += U[k, l]*A[l, m, n]
                res[k, m, n] = current
    return res


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _x2(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=2] U):
    cdef unsigned int i, j, l, k
    cdef unsigned int L = A.shape[0]
    cdef unsigned int M = A.shape[1]
    cdef unsigned int N = A.shape[2]
    cdef unsigned int K = U.shape[0]
    cdef unsigned int M1 = U.shape[1]
    assert A.dtype == DTYPE
    assert M == M1
    cdef float current = 0

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.zeros([L, K, N], dtype=DTYPE)
    for l in range(L):
        for k in range(K):
            for n in range(N):
                current = 0
                for m in range(M):
                    current += U[k, m]*A[l, m, n]
                res[l, k, n] = current
    return res


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _x3(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=2] U):
    cdef unsigned int i, j, l, k
    cdef unsigned int L = A.shape[0] 
    cdef unsigned int M = A.shape[1]
    cdef unsigned int N = A.shape[2]
    cdef unsigned int K = U.shape[0]
    cdef unsigned int N1 = U.shape[1]
    assert A.dtype == DTYPE
    assert N == N1
    cdef float current = 0

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.zeros([L, M, K], dtype=DTYPE)
    for l in range(L):
        for m in range(M):
            for k in range(K):
                current = 0
                for n in range(N):
                    current += U[k, n]*A[l, m, n]
                res[l, m, k] = current
    return res


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _proj_mat(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=2] U):
    cdef unsigned int i, j, l, k
    cdef unsigned int L = A.shape[0] 
    cdef unsigned int M = A.shape[1]
    cdef unsigned int N = A.shape[2]
    cdef unsigned int K = U.shape[0]
    cdef unsigned int N1 = U.shape[1]
    assert A.dtype == DTYPE
    assert N == N1
    cdef float current = 0

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.zeros([L, M, K], dtype=DTYPE)
    for l in range(L):
        for m in range(M):
            for k in range(K):
                current = 0
                for n in range(N):
                    current += U[k, n]*A[l, m, n]
                res[l, m, k] = current
    return res


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _proj_base(np.ndarray[DTYPE_t, ndim=3] X, np.ndarray[DTYPE_t, ndim=3] base,
               np.ndarray[double, ndim=1] vp):
    assert X.dtype == DTYPE
    assert base.dtype == DTYPE
    cdef int K = base.shape[0]
    cdef int N = X.shape[0] - K + 1

    cdef unsigned int k, j

    cdef np.ndarray[DTYPE_t, ndim=2] a = np.zeros([K, N], dtype=DTYPE)

    for k in range(K):
        for j in range(N):
            a[k, j] = X[j:j+K].dot(base[:, k]) / vp[k]

    return a
