import numpy as np


class Wav_Gen(object):
    """Class to generate Bell-Cylinder-Funnel datasets"""
    def __init__(self, n_points=1000):
        self.n_points = n_points
        self.t = np.arange(n_points)*128/n_points

    def gen_fun(self, ftype='cylinder'):
        '''Generate a sample of the TYPE class

        Parameters
        ----------
        ftype: str, optional (default: cylinder)
            Type of function to generate, have to be in
            {'cylinder', 'bell', 'funnel'}
        '''
        a = np.random.random()*16+16
        b = np.random.random()*64+32+a
        eps = np.random.normal(size=self.n_points+1)
        n = eps[0]
        eps = eps[1:]

        t = self.t
        if ftype == 'cylinder':
            return (6+n)*(t > a)*(t < b) + eps
        elif ftype == 'bell':
            return (6+n)*(t > a)*(t < b)*(t-a)/(b-a) + eps
        else:
            return (6+n)*(t > a)*(t < b)*(b-t)/(b-a) + eps

    def gen_set(self, N=100, rep='unif'):
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
        if rep == 'unif':
            rep = (0.33, 0.33, 0.34)
        assert(abs(sum(rep)-1) <= 1e-2)
        Nd = dict(cylinder=int(rep[0]*N),
                  bell=int(rep[1]*N),
                  funnel=int(N*rep[2]))
        data = []
        for k, v in Nd.items():
            data.extend([(self.gen_fun(ftype=k), k) for i in range(v)])
        np.random.shuffle(data)
        y = [d[1] for d in data]
        data = [d[0] for d in data]
        return np.array(data), np.array(y)

if __name__ == '__main__':
    dataset = Wav_Gen()
    db, y = dataset.gen_set()

    import matplotlib.pyplot as plt
    plt.plot(db[:5].T)

    plt.show()
