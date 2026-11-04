# -*- coding: utf-8 -*-
"""球与圆柱相交区域的体积：数值积分与随机投点。"""
import numpy as np
from scipy.integrate import dblquad

def quad2mont1(n, use_scipy=True, random_state=0):
    if use_scipy:
        value, _ = dblquad(lambda y,x: np.sqrt(max(4-x*x-y*y, 0)),
                           0, 2, lambda x: 0, lambda x: np.sqrt(max(2*x-x*x, 0)))
        V0 = 4*value
    else:
        V0 = 16*np.pi/3-64/9
    rng = np.random.default_rng(random_state)
    values = []
    for N in np.atleast_1d(n):
        x = rng.uniform(0, 2, N)
        y = rng.uniform(0, 1, N)
        z = rng.uniform(0, 2, N)
        inside = (x*x+y*y+z*z <= 4) & ((x-1)**2+y*y <= 1)
        # 包围盒体积 4，利用 y、z 对称性再乘 4。
        values.append(16*inside.mean())
    return V0, values[0] if np.isscalar(n) else np.array(values)

if __name__ == '__main__':
    counts = np.array([100, 1000, 10000, 100000, 1000000])
    V0, values = quad2mont1(counts)
    print('数值积分:', V0)
    for N, value in zip(counts, values):
        print(N, value, '误差:', value-V0)
