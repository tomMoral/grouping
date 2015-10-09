import numpy as np

from decomposition import _Decompose
import decomposition.__cython_aux as aux


class MSSA(_Decompose):
    """docstring for SSA"""
    def __init__(self, K=10, *args, **kwargs):
        self.K = K

    def fit(self, X, y=None, *args, **kwargs):
        N, m = X.shape
        print(N, m)
        G = aux.MSSA_c._lag_mat_mvar(X, self.K)
        G /= (N-self.K+1)
        print(G.shape)
        A1 = np.concatenate([G[:, k, :] for k in range(self.K)], axis=1)
        A2 = np.concatenate([G[:, :, k].T for k in range(m)], axis=1)
        A3 = np.concatenate([G[k, :, :].T for k in range(self.K)], axis=1)
        print('Compute')
        U1, s1, _ = np.linalg.svd(A1)
        S = aux._x1(G, U1.T)
        print(S.shape)
        U2, s2, _ = np.linalg.svd(A2)
        S = aux._x2(S, U2.T)
        U3, s3, _ = np.linalg.svd(A3)
        S = aux._x3(S, U3.T)
        print('100%')

        self.dic = (S, U1, U2, U3)
        self.vp = (s1, s2, s3)
        self.G = G

    def transform(self, X, th=0.5, *args, **kwargs):
        S, U1, U2, U3 = self.dic
        U3 = U3.copy()
        U3[:, :2]
        A = aux.x_3(aux._x2(aux._x1(S, U1), U2),  U3)
        return A

    def decompose(self, A, th=0.5, strat='mat'):
        self.grouping(A, th, strat=strat)
        decomp = []
        for g in self.GI.values():
            decomp.append(aux.SSA_c._recover(A, self.dic, np.array(g), self.vp))

        return decomp
