import numpy as np
from numpy import pi, sin, exp


class Sig_Gen(object):
    """Class to generate Bell-Cylinder-Funnel datasets"""
    def __init__(self):
        pass

    def gen_fun(self, ftype=0):
        '''Generate a sample of the TYPE class

        Parameters
        ----------
        ftype: int, optional (default: 0)
            Type of function to generate, in {1}
        '''

        L = int(self._random(182, 730))
        t = np.arange(L)
        compos = []
        if ftype == 0:
            P1 = self._random(5, 5+L/2)
            phi1 = self._random(-pi/2, pi/2)
            compos += [sin(2*pi*t/P1+phi1)]
        elif ftype == 1:
            P1 = self._random(5, 5+L/2)
            phi1 = self._random(-pi/2, pi/2)
            P2 = self._random(L/3, 2*L/3)
            phi2 = self._random(-pi/2, pi/2)
            compos += [sin(2*pi*t/P1+phi1)*sin(2*pi*t/P2+phi2)]
        elif ftype == 2:
            P1 = self._random(5, 5+L/2)
            phi1 = self._random(-pi/2, pi/2)
            gamma = self._random(.5, 5)
            compos += [sin(2*pi*t/P1+phi1)*exp(-gamma*t/L)]
        elif ftype == 3:
            a1 = np.random.random()
            a1 -= a1 < .5
            a2 = np.random.random()
            a2 -= a2 < .5
            b1 = self._random(0, L/2)
            b2 = self._random(0, L/2)
            c1 = self._random(0, L/3)**2
            c2 = self._random(0, L/3)**2
            compos += [a1*exp(-(b1-t)**2/c1) + a2*exp(-(b2-t)**2/c2)]

        elif ftype == 4:
            a1 = self._random(7, 9)
            P1 = self._random(5, 10)
            phi1 = self._random(-pi/2, pi/2)
            compos += [a1*sin(2*pi*t/P1+phi1)]
            a2 = self._random(1, 4)
            P2 = self._random(28, 32)
            phi2 = self._random(-pi/2, pi/2)
            compos += [a2*sin(2*pi*t/P2+phi2)]
            a3 = self._random(10, 18)
            P3 = self._random(13, 15)
            phi3 = self._random(-pi/2, pi/2)
            compos += [a3*sin(2*pi*t/P3+phi3)]
            a4 = self._random(15, 21)
            P4 = self._random(58, 62)
            phi4 = self._random(-pi/2, pi/2)
            compos += [a4*sin(2*pi*t/P4+phi4)]
        elif ftype == 5:
            L = 2*365
            a1 = self._random(9/2, 13/2)
            P1 = self._random(L, 3*L/2)
            compos += [a1*sin(2*pi*t/P1+.5)]
            a2 = self._random(1.5, 2.5)
            P2 = self._random(5, 7)
            compos += [a2*sin(2*pi*t/P2+1)]
            a3 = -self._random(3.5, 4.5)
            P31 = self._random(10, 12)
            P32 = self._random(150, 170)
            compos += [a3*sin(2*pi*t/P31+1)*sin(2*pi*t/P32)]
            a4 = self._random(.5, 1.5)
            P4 = self._random(29, 31)
            compos += [a4*sin(2*pi*t/P4+1)]
            a51 = self._random(.5, 1.5)
            a52 = self._random(.5, 1.5)
            b51 = self._random(0, L/2)
            b52 = self._random(0, L/2)
            c51 = self._random(0, L/3)**2
            c52 = self._random(0, L/3)**2
            compos += [a51*exp(-(b51-t)**2/c51) + a52*exp(-(b52-t)**2/c52)]
        elif ftype == 6:
            a = self._random(3/L, 8/L)
            compos += [a*t]
            P1 = self._random(5, 11)
            phi1 = self._random(-pi/2, pi/2)
            compos += [sin(2*pi*t/P1+phi1)]
            a2 = self._random(0, 0.5)
            P2 = self._random(L/3, L/11)
            phi2 = self._random(-pi/2, pi/2)
            compos += [a2*sin(2*pi*t/P2+phi2)]
        elif ftype == 7:
            Lm = int(self._random(20, L/10))
            inc = np.random.normal(0, 0.1, Lm)
            motif = np.cumsum(np.r_[inc, -inc[::-1]])
            cc = np.array(list(motif)*(L//(2*Lm)+1))
            compos += [cc[:L], 2*t/L]
        elif ftype == 8:
            a = self._random(2/L, 8/L)
            compos += [a*t]
            P1 = self._random(5, 11)
            phi1 = self._random(-pi/2, pi/2)
            compos += [sin(2*pi*t/P1+phi1)]
            sp = np.zeros(2000)
            sp[10:20] = 1
            sp = sp + sp[::-1]
            Lm = int(self._random(20, L/10))
            motif = np.fft.ifft(sp, Lm).real
            motif = list(motif/motif.std())
            cc = np.array(motif*(L//(Lm)+1))
            compos += [cc[:L]]

        return np.sum(compos, axis=0), compos

    def _random(self, a, b):
        return np.random.random()*(b-a)+a

    def gen_set(self, N=200, h=0, s=0.05, seed=False):
        '''Generate a dataset BCF with N samples

        Parameters
        ----------
        N: int, optional (default: 200)
            Number of signal in the dataset
        h: int, optional (default: 0)
            Type of signal generated
        s: float, optional (default=0.05)
            SNR of the generated signals
        seed: bool, optional (default: False)
            If set to True, seed the rng to get similar signals
        '''
        if seed:
            np.random.seed(1234)
        data = []
        compos = []
        for i in range(N):
            f, c = self.gen_fun(h)
            sigma_n = f.std()*s
            data += [f + sigma_n*np.random.normal(size=f.shape)]
            compos += [c]
        return data, compos

if __name__ == '__main__':
    dataset = Sig_Gen()
    dataset2 = Sig_Gen()
    db, y = dataset.gen_set(h=6, seed=True)
    db2, y2 = dataset2.gen_set(h=6, seed=True)
    db = np.array(db)
    db2 = np.array(db2)
    for d1, d2 in zip(db, db2):
        assert((d1 == d2).all())

    import matplotlib.pyplot as plt
    for k in range(5):
        plt.plot(db[k])

    plt.show()
