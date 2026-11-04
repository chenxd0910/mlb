# -*- coding: utf-8 -*-
"""对不同订货量使用同一组需求样本，比较平均利润。"""
import numpy as np
import matplotlib.pyplot as plt

p, c, v = 10, 6, 2
mu, sigma, N = 50, 10, 100000
rng = np.random.default_rng(0)
# 正态近似需求，将极少的负需求截为零。
demand = np.maximum(rng.normal(mu, sigma, N), 0)
Q_range = np.arange(30, 80)
mean_profit = []
for Q in Q_range:
    sales = np.minimum(demand, Q)
    leftover = np.maximum(Q-demand, 0)
    profit = p*sales + v*leftover - c*Q
    mean_profit.append(profit.mean())
best = np.argmax(mean_profit)
print('最优订购量:', Q_range[best])
print('最大期望利润的模拟估计:', mean_profit[best])
if __name__ == '__main__':
    plt.plot(Q_range, mean_profit)
    plt.xlabel('Order quantity')
    plt.ylabel('Mean profit')
    plt.show()
