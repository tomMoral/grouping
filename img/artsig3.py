import matplotlib.pyplot as plt
import numpy as np


def R2(c, g):
    mu = c.mean()
    SStot = np.sum((c-mu)**2)
    SSres = np.sum((c-g)**2)
    return 1 - SSres/SStot


def score(compos, groups):
    scores = []
    scoresP = []
    for i, c in enumerate(compos):
        scores += [[]]
        scoresP += [[]]
        for g in groups:
            scores[-1] += [R2(c, g)]
            scoresP[-1] += [R2(g, c)]
        if len(scores[-1]) == 0:
            scores[-1] += [0]
        if len(scores[-1]) == 0:
            scoresP[-1] += [0]
    recall = np.maximum(np.max(scores, axis=1), 0)
    return recall

if __name__ == '__main__':

    sig = np.load('signal_img.npy')
    dec_n = np.load('dec_n.npy')
    compos = np.load('compos.npy')
    rec = np.load('rec.npy')
    print(score(dec_n[0], compos).mean())
    print(score(rec, compos).mean())
    print(score(rec, compos))
    noise = sig-compos.sum(axis=0)
    plt.figure()
    plt.subplot(6, 1, 1)
    plt.plot(sig)
    for i, d in enumerate(dec_n[0]):
        plt.subplot(6, 1, i+2)
        plt.plot(d)
    plt.subplot(6, 1, 6)
    plt.plot(sig-np.sum(dec_n[0], axis=0))
    for i in range(3, 7):
        plt.subplot(6, 1, i)
        plt.xlim((0, len(sig)))
        plt.ylim((-3, 3))
        plt.yticks([-3, 0, 3])
        if i < 6:
            plt.xticks([])
        else:
            plt.xticks(100*np.arange(7), np.arange(7))
            plt.xlabel('temps (s)')

    for i in range(2):
        plt.subplot(6, 1, i+1)
        plt.xlim((0, len(sig)))
        plt.ylim((-4, 10))
        plt.yticks([-4, 3, 10])
        plt.xticks([])
    plt.savefig('dec_artsig.eps')
    plt.figure()
    plt.plot(rec.T)
    print('SNR: {:.2}dB'.format(-10*np.log(noise.std()/sig.std())))
    plt.show()
