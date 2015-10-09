

class _Decompose(object):
    """Abstract class for decomposition algorithm"""
    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, X, y=None, *args, **kwargs):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y, *args):
        raise DecomposeImplementationError(
            'fit not implemented for {}'
            ''.format(self.__class__.__name__))

    def transform(self, X, *args):
        raise DecomposeImplementationError(
            'transform not implemented for {}'
            ''.format(self.__class__.__name__))


class DecomposeImplementationError(Exception):
    """Error for in the implementation of a _Decomposition object"""
    def __init__(self, msg):
        super(DecomposeImplementationError, self).__init__()
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
