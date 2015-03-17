import numpy as np
import itertools
from utils.logger import Logger
log = Logger()


def _comp_S(scores):
    N, M = np.shape(scores)
    if M > 20:
        return 0
    R = 0
    for p in itertools.permutations(range(max(M, N)), N):
        Rc = 0
        for i, j in enumerate(p):
            if j < M:
                Rc += scores[i][j]
        if Rc > R:
            R = Rc
    return R/max(M, N)


def comp_R(scores):
    return np.maximum(np.max(scores, axis=1), 0).mean()


def comp_P(scores):
    return np.maximum(np.max(scores, axis=0), 0).mean()


def write_mlb_mat(name, arr):
    with open(name, 'w') as f:
        for r in arr:
            for v in r:
                f.write('{},'.format(v))
            f.write('\n')


if __name__ == '__main__':
    try:
        scores = np.load('MAt_17-03_matin.npy')
        mat = np.array([np.r_[scores[0], scores[1, 1:]],
                        np.r_[scores[2], scores[3, 1:]],
                        np.r_[scores[4], scores[5, 1:]]])
        R, Rc = [], []
        P, Pc = [], []
        S, Sc = [], []

        for k in [0, 1, 2]:
            Rc += [[]]
            for j, ss in enumerate(mat[k].T):
                if (np.array([len(ls) for ls in ss]) > 0).all():
                    lR = []
                    for i, x in enumerate(ss):
                        lR += [comp_R(x)]
                    R += [lR]
                    Rc[-1] += [lR]
        for k in [0, 1, 2]:
            Pc += [[]]
            for j, ss in enumerate(mat[k].T):
                if (np.array([len(ls) for ls in ss]) > 0).all():
                    lP = []
                    for i, x in enumerate(ss):
                        lP += [comp_P(x)]
                    log.progress(name='Precision'.format(k),
                                 iteration=12000*k+j*12+i,
                                 max_iter=36000)
                    P += [lP]
                    Pc[-1] += [lP]
        for k in [0, 1, 2]:
            Sc += [[]]
            for j, ss in enumerate(mat[k].T):
                if (np.array([len(ls) for ls in ss]) > 0).all():
                    lS = []
                    for i, x in enumerate(ss):
                        lS += [_comp_S(x)]
                        log.progress(name='Permutation'.format(k),
                                     iteration=12000*k+j*12+i,
                                     max_iter=36000)
                    S += [lS]
                    Sc[-1] += [lS]

        R0 = []
        for k in [1, 2]:
            for j, ss in enumerate(mat[k].T):
                if (np.array([len(ls) for ls in ss]) > 0).all():
                    lR = []
                    for i, x in enumerate(ss):
                        lR += [np.maximum(np.max(x[0], axis=0), 0)]
                    R0 += [lR]
        Rh = []
        for k in [0, 1]:
            for j, ss in enumerate(mat[k].T):
                if (np.array([len(ls) for ls in ss]) > 1).all():
                    lR = []
                    for i, x in enumerate(ss):
                        lR += [np.maximum(np.max(x[k:], axis=0).mean(), 0)]
                    Rh += [lR]
        Reh = []
        for k in [2]:
            for j, ss in enumerate(mat[k].T):
                if (np.array([len(ls) for ls in ss]) > 0).all():
                    lR = []
                    for i, x in enumerate(ss):
                        lR += [np.maximum(np.max(x[1:], axis=0).mean(), 0)]
                    Reh += [lR]

        R, P, S = np.array(R), np.array(P), np.array(S)
        Rc, Pc, Sc = np.array(Rc), np.array(Pc), np.array(Sc)
        R0, Rh, Reh = np.array(R0), np.array(Rh), np.array(Reh)

        impR = (R[:, 1:]-R[:, :1])/(1-R[:, :1])
        impP = (P[:, 1:]-P[:, :1])/(1-P[:, :1])
        impS = (S[:, 1:]-S[:, :1])/(1-S[:, :1])

        for met, metc, name in zip([R, P, S], [Rc, Pc, Sc], ['R', 'P', 'S']):
            for c in range(3):
                write_mlb_mat('{}_c{}.txt'.format(name, c),
                              (metc[c, :, 1:]-metc[c, :, :1]) /
                              (1-metc[c, :, :1]))
            write_mlb_mat('{}_c3.txt'.format(name),
                          (met[:, 1:]-met[:, :1]) /
                          (1-met[:, :1]))

        # Format string to generate the table
        fs = lambda i0, name: (
            name + ' - MU & ' + ' & '.join(['{'+str(i)+':.3}' if i != i0
                                            else '\\textbf{{{'+str(i)+':.3}}}'
                                            for i in range(6)]) +
            '\\\\\\hline' +
            '\n'+name+' - MH & ' + ' & '.join(['{'+str(i)+':.3}' if i != i0
                                              else '\\textbf{{{'+str(i)+':.3}}}'
                                              for i in range(6, 12)]) +

            '\\\\\\hline')

        print(fs(impR.mean(axis=0).argmax(), '$R_r$'
                 ).format(*impR.mean(axis=0)))
        print(fs(impP.mean(axis=0).argmax(), '$P_r$'
                 ).format(*impP.mean(axis=0)))
        print(fs(impS.mean(axis=0).argmax(), '$S_r$'
                 ).format(*impS.mean(axis=0)))

    finally:
        log.end()
