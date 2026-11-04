# -*- coding: utf-8 -*-
"""三重积分：包围盒抽样与指示函数。"""
import numpy as np
from scipy.integrate import tplquad

def quad3mont(n, seed=0):
    # scipy 被积函数的参数顺序为 func(z, y, x)。
    V0, _ = tplquad(lambda z,y,x: x*y*z, 1, 2,
                    lambda x: x, lambda x: 2*x,
                    lambda x,y: x*y, lambda x,y: 2*x*y)
    rng = np.random.default_rng(seed)
    values = []
    for N in np.atleast_1d(n):
        x = rng.uniform(1, 2, N)
        y = rng.uniform(1, 4, N)
        z = rng.uniform(1, 16, N)
        inside = (y >= x) & (y <= 2*x) & (z >= x*y) & (z <= 2*x*y)
        values.append(45*np.mean(x*y*z*inside))
    return V0, values[0] if np.isscalar(n) else np.array(values)

if __name__ == '__main__':
    counts = np.array([100, 1000, 10000, 100000, 1000000])
    V0, values = quad3mont(counts)
    print('数值积分:', V0, '解析值:', 11475/64)
    for N, value in zip(counts, values):
        print(N, value, '误差:', value-V0)
