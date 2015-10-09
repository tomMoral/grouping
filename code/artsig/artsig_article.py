import numpy as np
from numpy import pi, sin, exp


class Sig_Gen(object):
    """Signal Generator for the model:
        f(t) = a_0.t^p + \sum_{i=1}^K a_i exp(-b_i t) sin(2\pi f_i t + \phi_i)
        f_i(t) = f(t) + epsilon_t

        with:
        * K in [1, 5]
        * p in [0, 5]
        * a_i in [0, 1]
        * phi_i in[-pi/2, pi/2]
        * b_i in [0, 0.5]
        * f_i in [0, 50]Hz
        and epsilon_t gaussian withe noise with sigma = s* sigma(f)"""
    def __init__(self):
        pass

    def gen_fun(self, ftype=0):
        '''Generate a sample of the ftype class

        Parameters
        ----------
        ftype: int, optional (default: 0)
            Type of function to generate, {0, 1, 2}:
             *0: a_0 = b_i = 0
             *1: b_i = 0
             *2: no constraints
        '''
        # Sampling frequency
        Fe = 100

        # Define time
        L = int(self._rand(500, 1500))
        t = np.arange(L)/Fe

        components = []

        if ftype > 0:
            a = self._rand(1)
            p = self._rand(5)
            components += [a*(t/2)**p]

        K = int(self._rand(1, 6))
        for k in range(K):
            b = self._rand(1)
            f = self._rand(50)
            phi = self._rand(-pi/2, pi/2)
            alpha = 0
            if ftype > 1:
                alpha = self._rand(0.5)
            components += [b*exp(-alpha*t)*sin(f*t + phi)]

        if ftype > 2:
            b = self._rand(10)
            a = self._rand(L)
            components += [b*t*sin(2*pi*f*t + phi)]

        return np.sum(components, axis=0), components

    def _rand(self, a, b=None):
        if b is None:
            b = a
            a = 0
        return np.random.random()*(b-a)+a

    def gen_set(self, N=1000, h=0, seed=None, s=1,
                save=None, **kwargs):
        '''Generate a dataset samples for the model of gretsi article

        Parameters
        ----------
        N: int, optional (default: 200)
            Number of signal in the dataset
        h: int, optional (default: 0)
            Type of signal generated
        s: flaot, optional (default: 0.5)
            maximal SNR
        seed: bool, optional (default: False)
            If set to True, seed the rng to get similar signals
        '''
        if seed is not None:
            np.random.seed(seed)
        data = []
        compos = []
        noise = []
        for i in range(N):
            f, c = self.gen_fun(h)
            s = self._rand(0.01, s)
            sigma_n = f.std()*s
            noise += [sigma_n*np.random.normal(size=f.shape)]
            data += [f + noise[-1]]
            compos += [c]
        if save is not None:
            np.save(save, compos)
            np.save(save+'_n', noise)
        return data, compos

if __name__ == '__main__':
    dataset = Sig_Gen()
    db, y = dataset.gen_set(h=3, seed=True)
    import matplotlib.pyplot as plt
    for k in range(5):
        plt.plot(db[k])

    plt.show()
