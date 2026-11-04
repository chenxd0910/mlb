# -*- coding: utf-8 -*-
"""平均法与投点法求积分。投点法的上下界由题目给定。"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def mont_int_avg(f, x_min, x_max, N=100000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, N)
    return (x_max-x_min) * np.mean(f(x))

def mont_int(f, x_min, x_max, N=100000, y_min=0, y_max=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, N)
    y = rng.uniform(y_min, y_max, N)
    inside = y <= f(x)
    value = (x_max-x_min) * (y_min+(y_max-y_min)*inside.mean())
    plt.figure()
    plt.scatter(x[:2000], y[:2000], c=inside[:2000], s=5, cmap='coolwarm')
    xx = np.linspace(x_min, x_max, 300)
    plt.plot(xx, f(xx), 'k-')
    return value

if __name__ == '__main__':
    def f(x):
        return np.log(x)
    a, b, N = 1, 5, 100000
    rng = np.random.default_rng(0)
    area = (b-a) * f(rng.uniform(a, b, N))
    print('数值积分:', quad(f, a, b)[0])
    print('平均法:', area.mean(), '标准误:', area.std(ddof=1)/np.sqrt(N))
    print('投点法:', mont_int(f, a, b, N, 0, np.log(5)))
    plt.show()
