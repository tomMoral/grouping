import numpy as np


class Grouping_Algorithms(object):
    """Class handeling grouping of SSA components"""
    def __init__(self, rec, dic, vp):
        '''Grouping Algorithms

        Parameters
        ----------
        rec: array-like
            Filtered components formed with the SSA
        dic: array-like
            Dicitonary retrieved with the SSA
        vp: array-like
            Eigenvalues of each components

        Usage
        -----
            ga = Grouping_Algorithms(rec, dic, vp)
            dec, _, _ = ga.regroup(strat='HGS', rho_0=0.7)
        '''
        self.rec = np.array(rec)
        self.dic = np.array(dic)
        self.vp = np.array(vp)
        self.K, self.T = self.rec.shape
        self.w_mat = None
        self.c_mat = None
        self.mvp = self.vp.max()

        self.w = np.array(list(range(1, self.K+1)) +
                          [self.K]*(self.T-2*self.K) +
                          list(range(self.K, 0, -1)))
        self.nr = np.sqrt((self.rec*self.rec).sum(axis=1))
        self.nrw = np.sqrt((self.rec*self.rec*self.w).sum(axis=1))

    def regroup(self, strat='HG', gs='full', **kwargs):
        '''Perform a grouping based on the given strategy

        Parameters
        ----------
        strat: str, optional (default: HG)
            Strategy for the grouping, in {'GG1', 'GG2', 'GG3',
            'HG', 'HGS', 'AT', 'KM'}
        gs: str, optional (default: full)
            group formation strategy, in {'full', 'first'}
        kwargs: dict
            hold Parameters for the strategy, see different strat for
            the parameters description

        Return
        ------
        dec: list of the retrieved components
        G: adjacency matrix
        GI: list containing the differents groups
        '''
        try:
            f_strat = self.__getattribute__('_'+strat)
        except AttributeError:
            f_strat = self._NO
            print('Fail to load strat', strat)
        G = f_strat(**kwargs)
        GI = self._group(G, gs)
        dec = []
        for g in GI:
            dec += [np.sum(self.rec[g], axis=0)]
        return dec, G, GI

    def _group(self, G, gs='full'):
        '''Given an adjacency matrix G, construct a grouping

        Return
        ------
        GI: list containing the differents groups
        '''
        K = self.K
        GI = []
        if gs == 'first':
            poss_i = list(range(K))
            while len(poss_i) > 0 and self.vp[poss_i[0]] > self.mvp*0.:  # 0005:
                i = poss_i[0]
                poss_i.pop(0)
                if G[i, i] == 0:
                    continue
                ng = [i]
                j = 0
                while j < len(poss_i):
                    if G[i, poss_i[j]] == 1:
                        ng += [poss_i[j]]
                        poss_i.pop(j)
                    else:
                        j += 1
                GI += [ng]
        else:
            poss_i = np.arange(K)
            for i in range(K):
                for j in range(i+1, K):
                    if G[i, j] == 1:
                        poss_i[j] = poss_i[i]
            for k in range(K):
                g = (poss_i == k).nonzero()[0]
                if len(g) > 0:
                    GI += [g]
        return GI

    def _NO(self, **kwargs):
        ''' No Grouping
        '''
        return np.eye(self.K)

    def _GG1(self, rho_1=0.8, rho_c=0.8, **kwargs):
        '''Grouping based on correlation between components
        Abalov_14_auto

        Parameters
        ----------
        rho_1: float, optional (default: 0.8)
            Threshold ratio to determine if the eigen triples
            are close enough
        rho_c: float, optional (default: 0.8)
            Threshold for the w-correlation

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        G = np.empty((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                G[i, j] = (self.vp[j]/self.vp[i] > rho_1)
                G[j, i] = G[i, j]

        self._correlation()
        G *= (self.c_mat > rho_c)
        return G

    def _GG2(self, rho_1=0.8, rho_2=0.05, rho_p=0.8, **kwargs):
        '''Grouping based on periodogramme study of the components
        Abalov_14_auto

        Parameters
        ----------
        rho_1: float, optional (default: 0.8)
            Threshold ratio to determine if the eigen triples
            are close enough
        rho_2: float, optional (default: 0.05)
            Threshold for to determine the fraquency of interest
        rho_p: float, optional (default: 0.8)
            Threshold for to determine if the maximas are close enough

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        F_rc = []
        for r in self.rec:
            per = abs(np.fft.fft(r))
            per /= np.sqrt(per.dot(per))
            per = per[:self.T//2]
            Krc = max(per)
            i_rc = (per >= rho_p*Krc).nonzero()[0]
            i_rc.sort()
            F_rc += [i_rc]

        G = np.empty((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                G[i, j] = self.vp[j]/self.vp[i] > rho_1
                l = min(len(F_rc[i]), len(F_rc[j]))
                G[i, j] *= (np.abs(F_rc[i][:l]-F_rc[j][:l])*2/self.T <=
                            rho_2).all()
                G[j, i] = G[i, j]
        self.G2 = G
        return G

    def _GG3(self, rho_1=0.8, rho_c=0.8, **kwargs):
        '''Grouping based on the w-correlation between the components
        Moreau_15_auto

        Parameters
        ----------
        rho_1: float, optional (default: 0.8)
            Threshold ratio to determine if the eigen triples
            are close enough
        rho_c: float, optional (default: 0.8)
            Threshold for the w-correlation

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        G = np.empty((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                G[i, j] = (self.vp[j]/self.vp[i] > rho_1)
                G[j, i] = G[i, j]

        self._w_correlation()
        G *= (self.w_mat > rho_c)
        return G

    def _HG(self, rho_0=0.8, **kwargs):
        '''Grouping based on the shared frequencies of
        the successive element of the dictionary
        Golyandina_05_auto

        Parameters
        ----------
        rho_0: float
            Threshold ratio to determine if the psd mode
            are similar

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        per_d = []
        N = len(self.dic.T[0])
        for d in self.dic.T:
            per = abs(np.fft.fft(d, n=N))
            per /= max(per)
            per_d += [per]
            per = per[:N//2]

        G = np.zeros((self.K, self.K))
        for i in range(self.K-1):
            G[i, i] = 1
            G[i, i+1] = ((per_d[i]+per_d[i+1]).max()/2 >= rho_0)
            G[i+1, i] = G[i, i+1]
        G[self.K-1, self.K-1] = 1
        self.HG = G
        return G

    def _AT(self, rho_0=0.8, s_0=1, **kwargs):
        '''Grouping based on the shared frequencies of
        the successive element of the dictionary
        Alexandrov_07_auto

        Parameters
        ----------
        rho_0: float, optional (default: 0.8)
            Threshold ratio to determine if the psd mode
            are similar
        s_0: float, optional (default: 0.8)
            Threshold ratio to determine if the psd mode
            are similar

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        per_d = []
        for d in self.dic.T:
            per = abs(np.fft.fft(d))
            per /= max(per)
            per = per[:self.T//2]
            per_d += [per]

        G = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                a = abs(per_d[i].argmax()-per_d[j].argmax()) <= s_0
                g_ij = per_d[i]+per_d[j]
                G[i, j] = a & ((g_ij[:-1]+g_ij[1:]).max()/2 >= rho_0)
                G[j, i] = G[i, j]
        self.HG = G
        return G

    def _HGS(self, rho_p=0.8, rho_f=0.001, rho_s=0.01, **kwargs):
        '''Grouping based on the frequency support of
        the dictionary elements
        Moreau_15_auto

        Parameters
        ----------
        rho_p: float, optional (default: 0.8)
            Threshold ratio to determine the support of the
            puissance spectrum density
        rho_s: float, optional (default: 0.01)
            Threshold ratio to determine the maximal width of the
            support of periodograms
        rho_f: float, optional (default: 0.001)
            Threshold ratio to determine the maximal distances
            between the pics of each periodogram

        Return
        ------
        G: adjacency matrix for the grouping'''
        supp = []
        for d in self.dic.T[:self.K]:
            supp.append(self._support(d, rho_p, rho_s))

        G = np.zeros((self.K, self.K))
        for k, (s, v) in enumerate(zip(supp, self.vp)):
            if s[1]:
                ff = s[0]
                G[k, k] = 1
                for j, (s1, v1) in enumerate(zip(supp[k:], self.vp[k:])):
                    if s1[1] and abs(s1[0]-ff) <= rho_f and\
                            v1 >= v/self.K:
                        G[k, k+j] = 1
                        G[k+j, k] = 1
        self.DF = G
        return G

    def _HGS2(self, rho_s, rho_f, **kwargs):
        '''Grouping based on the frequency support of
        the reconstructed signals

        Parameters
        ----------
        rho_p: float
            Threshold ratio to determine the support of the
            puissance spectrum density

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        supp = []
        for r in self.rec:
            supp.append(self._support(r, rho_s, 10*rho_f))

        G = np.zeros((self.K, self.K))
        for k, (s, v) in enumerate(zip(supp, self.vp)):
            if s[1]:
                ff = s[0]
                G[k, k] = 1
                for j, (s1, v1) in enumerate(zip(supp[k:], self.vp[k:])):
                    if s1[1] and abs(s1[0]-ff) <= rho_f and\
                            v1 >= v/self.K:
                        G[k, k+j] = 1
                        G[k+j, k] = 1
        self.DF = G
        return G

    def _KM(self, rho_r=0.4, **kwargs):
        '''Grouping based on the euclidean distances between
        periodograms, with K-means clustering
        Alvarez_13_auto

        Parameters
        ----------
        rho_f: float, optional (default: 0.4)
            Threshold ratio to determine the rank of the
            periodograms matrix - Used to estimate the number
            of components

        Return
        ------
        G: adjacency matrix for the grouping
        '''
        from numpy.fft import fft
        N = self.K+1000
        X = []
        for d in self.dic.T[:self.K]:
            psd = abs(fft(d, n=N))[:N//2]
            X += [psd]

        X = np.array(X)
        d = np.linalg.svd(X, compute_uv=False)
        C = max(sum((d >= max(d)*rho_r)), 1)
        from sklearn.cluster import KMeans
        km = KMeans(C, n_init=100, n_jobs=2)
        km.fit(X)
        G = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i, self.K):
                G[i, j] = km.labels_[i] == km.labels_[j]
                G[j, i] = G[i, j]
        return G

    def _correlation(self):
        '''Compute the correlation matrix between components
        '''
        if self.c_mat is None:
            G = np.empty((self.K, self.K))
            for i in range(self.K):
                for j in range(i, self.K):
                    G[i, j] = self.rec[i].dot(self.rec[j])
                    G[i, j] /= self.nr[i]*self.nr[j]
                    G[j, i] = G[i, j]
            self.c_mat = G

    def _w_correlation(self):
        '''Compute the w-correlation matrix between components
        '''
        if self.w_mat is None:
            G = np.empty((self.K, self.K))
            for i in range(self.K):
                for j in range(i, self.K):
                    G[i, j] = self.rec[i].dot(self.rec[j]*self.w)
                    G[i, j] /= self.nrw[i]*self.nrw[j]
                    G[j, i] = G[i, j]
            self.w_mat = G

    def _support(self, f, rho_p=0.5, rho_2=0.003):
        '''Auxillary function to get the support of the psd of f

        Parameters
        ----------
        f: 1D array-like
            signal fro which is retrieve the psd support
        rho_p: float, optional (default: 0.5)
            Threshold ratio to determine the support of the
            puissance spectrum density

        Return
        ------
        (freq, b, amp)
            freq is the center frequency of the support
            b is True iff the support is not spread
            amp is the maximum value of the psd
        '''
        from numpy.fft import fft
        N = len(f)+1000
        psd = abs(fft(f, n=N))[:N//2]
        a = (psd >= max(psd)*rho_p).nonzero()[0]
        if len(a) <= 1:
            return f.argmax()/N, True, max(psd)
        m = ((a < N/2)*(a+1)).nonzero()[0][-1]
        return ((a[0] + a[m])/(2*N),
                (a[m]-a[0])/2 < rho_2*N, max(psd))
