import numpy as np
cimport numpy as np

cimport cython

DTYPE=np.float64
ctypedef np.complex DTYPE_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
def cost_dict_update(np.ndarray[DTYPE_t, ndim=1] X_, np.ndarray[DTYPE_t, ndim=2] StS, np.ndarray[DTYPE_t, ndim=2] Stx, 
					 float beta, np.ndarray[DTYPE_t, ndim=2] sol=None, np.ndarray[DTYPE_t, ndim=2] Z=None):
    cdef int t, i, j, d
    cdef complex cost = 0
    cdef np.ndarray[DTYPE_t, ndim=1] grad
    cdef np.ndarray[DTYPE_t, ndim=2] hess
    cdef np.ndarray[DTYPE_t, ndim=2] Mt
    cdef np.ndarray[DTYPE_t, ndim=2] Sd
    cdef np.ndarray[DTYPE_t, ndim=1] temp


    d = len(Stx[0])
    N = len(X_)
    grad = np.zeros(d, dtype=np.complex)
    hess = np.zeros((d, d), dtype=np.complex)

    if sol is None:
        return 0, np.zeros(d)


    print('begin')

    for t in range(N):

        t_i = min(t, N-t)
        Sd = np.zeros((d, d), dtype=np.complex)
        for i in range(d):
            for j in range(i+1):
                Sd[i, j] = StS[i*(i+1)//2+j][t_i]
                Sd[j, i] = StS[i*(i+1)//2+j][t_i]

        Mt = np.linalg.inv(Sd + np.diag(sol))

        temp = Mt.dot(Stx[t_i])
        cost += abs(X_[t_i])**2 - Stx[t_i].T.dot(temp)
        grad += (temp)**2
        if Z is not None:
            hess -= 2*temp[:, None].dot(np.conj(temp).T.dot(Mt)[None, :])*Mt

    for i in range(d):
        cost = cost  - beta*sol[i, 0]
        grad[i] = grad[i] - complex(beta)
    if Z is None:
        return cost, grad

    return cost, grad, Z*hess
