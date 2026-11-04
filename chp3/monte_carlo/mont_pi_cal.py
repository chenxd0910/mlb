# -*- coding: utf-8 -*-
"""随机投点估计圆周率。n 为正整数或正整数列表。"""
import numpy as np
import matplotlib.pyplot as plt

def PiMonteCarlo(n, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.atleast_1d(n).astype(int)
    x = rng.uniform(-1, 1, counts.max())
    y = rng.uniform(-1, 1, counts.max())
    hits = np.cumsum(x*x+y*y <= 1)
    values = 4 * hits[counts-1] / counts
    plt.figure()
    if len(counts) == 1:
        plt.scatter(x, y, c=(x*x+y*y <= 1), s=4, cmap='coolwarm')
        theta = np.linspace(0, 2*np.pi, 300)
        plt.plot(np.cos(theta), np.sin(theta), 'k-')
        plt.axis('equal')
        plt.title(f'Estimated pi = {values[0]:.5f}')
    else:
        plt.plot(counts, values)
        plt.axhline(np.pi, color='k')
        plt.xlabel('Number of points')
        plt.ylabel('Estimated pi')
    return values[0] if np.isscalar(n) else values

if __name__ == '__main__':
    values = PiMonteCarlo(np.arange(100, 200001, 100))
    print('最终估计:', values[-1])
    plt.show()
