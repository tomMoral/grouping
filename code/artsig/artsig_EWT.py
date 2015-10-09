import numpy as np
from numpy import pi, cos, sin


class Sig_Gen(object):
    """Class to generate Bell-Cylinder-Funnel datasets"""
    def __init__(self, T=1000):
        self.T = T
        self.t = np.arange(T)/T

    def gen_fun(self, ftype=1):
        '''Generate a sample of the TYPE class

        Parameters
        ----------
        ftype: int, optional (default: 1)
            Type of function to generate, in {1}
        '''

        t = self.t
        compos = [np.zeros(t.shape)]
        if ftype == 1:
            a = np.random.randint(3, 8)
            w1 = np.random.randint(5, 11)
            b = np.random.randint(3, 7)
            compos += [a*t]
            compos += [cos(w1*pi*t)]
            compos += [0.5*cos(b*w1*pi*t)]
        elif ftype == 2:
            compos += [6*t*t]
            compos += [cos(10*pi*t*(1+t))]
            compos += [cos(pi*(80*t-15))*(t > 0.5),
                       cos(pi*60*t)*(t <= 0.5)]
        elif ftype == 3:
            compos += [1/(1.2+cos(2*pi*t))]
            compos += [1/(1.5+sin(2*pi*t))]
            compos += [cos(32*pi*t + cos(64*pi*t))]

        return np.sum(compos, axis=0), compos

    def gen_set(self, N=100, h=1, noise=3, seed=None):
        '''Generate a dataset BCF with N samples

        Parameters
        ----------
        N: int, optional (default: 100)
            Number of signal in the dataset
        rep: tuple, optional (default: unif)
            Repartition of the classes, if not unif, should
            be a 3-tuple of float containing the ratio of each class
            ('cylinder', 'bell', 'funnel')
        '''
        if seed is not None:
            np.random.seed(seed)
        data = []
        compos = []
        f, c = self.gen_fun(h)
        n = noise*np.random.random()
        data += [f + n*np.random.random(f.shape)]
        compos += [c]
        return np.array(data), compos

if __name__ == '__main__':
    dataset = EWT_Gen()
    db, y = dataset.gen_set(ftype=[1, 2, 3])

    import matplotlib.pyplot as plt
    plt.plot(db.T)
    print(len(db))

    plt.show()
