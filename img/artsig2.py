import numpy as np
import matplotlib.pyplot as plt


sig = np.load('artsig2.npy')
for i, yy in enumerate(sig):
    plt.subplot(6, 1, i+2)
    plt.plot(yy)
    plt.xlim((0, 1227))
    if i > 0:
        plt.ylim((-.6, .6))
        plt.yticks([-.6, 0, .6])
    else:
        plt.yticks([-2, 2, 8])
        plt.ylim((-2, 8))
plt.subplot(6, 1, 1)
plt.plot(np.sum(sig, axis=0))
plt.yticks([-2, 2, 8])
plt.ylim((-2, 8))
plt.xlim((0, 1227))
