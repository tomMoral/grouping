import numpy as np

from ._Decompose import _Decompose
from . import __cython_aux as aux


class SSA(_Decompose):
    """Singular Spectrum Analysis Algorithm

    Compute empirical dictionaries and embeddings of signals
    The dictionary elements are linked to the trend and oscilliation
    phases on the signal
    """
    def __init__(self, K=10, **kwargs):
        super(SSA, self).__init__(**kwargs)
        self.K = K

    def fit(self, X, y=None, **kwargs):
        '''Compute dictionary from the signal X

        Parameters
        ----------
        X: array-like, (T,)
            Signal to fit with the SSA
        '''
        m = 1
        if X.ndim == 2:
            print('dim 2!!!!', X.shape)
            m = X.shape[1]
            self.G = np.zeros((self.K, self.K))
            for l in range(m):
                G = np.empty((self.K, self.K))
                aux.SSA_c._lag_mat(X[:, l], G, self.K)
                self.G += G
        else:
            self.G = np.empty((self.K, self.K))
            aux.SSA_c._lag_mat(X, self.G, self.K)
        self.dic, self.vp, _ = np.linalg.svd(self.G)
        self.w = (list(range(1, self.K+1)) + [self.K]*(len(X)-2*self.K) +
                  list(range(self.K, 0, -1)))

    def transform(self, X, tau_l=0.01, **kwargs):
        '''Compute an embeddings of X on the ssa-dictionnary

        Parameters
        ----------
        X: array-like, (T,)
            Signal to represent in the ssa space
        tau_l: float, optional (default: 0.01)
            ratio to filter components, select components s.t.
                lambda_k > lambda_2*tau_l
        '''
        if X.ndim == 2:
            m = X.shape[1]
            A = aux.SSA_c._proj_base(X[:, 0], self.dic, self.vp)
            for l in range(1, m):
                A = np.c_[A.T, aux.SSA_c._proj_base(X[:, l],
                          self.dic, self.vp).T].T
        else:
            A = []
            for l in range(self.K):
                A += [np.convolve(X, self.dic[::-1, l]/self.vp[l],
                                  mode='valid')]
                if self.vp[l] < self.vp[1]*tau_l:
                    break
        return np.array(A)

    def decompose(self, X, strat='HGS', gs='first',
                  tau_l=0.01, **kwargs):
        '''Perform a signal decomposition on the ssa-dicitonary
        with automated Grouping

        Parameters
        ----------
        X: array-like, (T,)
            Signal to decompose
        strat: str, optional (default: HGS)
            Grouping strategy, choosen in {'GG1', 'GG2', 'GG3',
            'HG', 'HGS', 'AT', 'KM'}
        gs: str, optional (default: first)
            group formation strategy, in {'full', 'first'}
        tau_l: float, optional (default: 0.01)
            ratio to filter components, select components s.t.
                lambda_k > lambda_2*tau_l
        kwargs: dict, optional
            parameters for the grouping strategy, see
            ./Grouping for more info about each parameters

        Return
        ------
        dec: array-like, (n_compo, T)
        '''
        fssa = self.fit_transform(X)
        rec = []
        for k in range(self.K):
            rec += [self._compos_k(fssa[k], k)]
            if self.vp[k] < self.vp[1]*tau_l:
                break
        from .Grouping import Grouping_Algorithms
        ga = Grouping_Algorithms(rec, self.dic, self.vp)
        dec, _, GI = ga.regroup(strat, gs, **kwargs)

        self.GI = GI
        return np.array(dec)

    def detrend(self, X):
        '''Detrend the signal X

        Parameters
        ----------
        X: array-like (T,)
            Signal we wich to detrend
        '''
        dec = self.decompose(X, strat='KM')
        return X-dec[0], dec[0]

    def _compos_k(self, Ak, k):
        '''Return reconstructed component k'''
        return np.convolve(Ak, self.dic[:, k]*self.vp[k])/self.w


if __name__ == '__main__':

    # Generate dummy data
    X = np.array([1, 3, 0, -3, -2, -1], dtype=np.float)

    # Fit a SSA model
    model = SSA(K=3)
    X_ssa = model.fit_transform(X)

    round_t = lambda x: round(x, 3)
    round_m = lambda x: list(map(round_t, x))

    #Assertion to verify the model on a toy example
    assert (map(round_m, model.vp) == np.array([7.519, 5.539, 0.693])).all
    assert (abs(model.G - model.dic.dot(np.diag(model.vp)
                                        ).dot(model.dic.T)) < 1e-5).all()
    res = np.array(list(map(round_m, X_ssa)))
    print(res)
    assert (res == np.array([[-2.983, -1.063, 2.826, 3.473],
                             [-0.137, 4.09, 2.224, -0.676],
                             [-1.041, -0.375, 0.255, -1.217]]
                            )).all()

    # Plot the data and the representation
    import matplotlib.pyplot as plt
    plt.plot(X)
    plt.plot(model.recover(X_ssa))
    plt.draw()
    plt.show()
