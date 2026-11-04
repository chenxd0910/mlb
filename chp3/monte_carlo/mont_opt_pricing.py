# -*- coding: utf-8 -*-
"""无分红欧式期权：风险中性几何布朗运动的欧拉模拟。"""
import numpy as np
from scipy.stats import norm

S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1
N, n = 100000, 52
dt = T/n
rng = np.random.default_rng(0)
St = np.full(N, float(S0))
for t in range(n):
    epsilon = rng.normal(size=N)
    St = St + r*St*dt + sigma*St*np.sqrt(dt)*epsilon
call = np.exp(-r*T)*np.maximum(St-K, 0)
put = np.exp(-r*T)*np.maximum(K-St, 0)
print('欧拉模拟（含时间离散误差）')
for name, values in [('看涨', call), ('看跌', put)]:
    print(name, '价格:', values.mean(), '标准误:', values.std(ddof=1)/np.sqrt(N))

# 欧式期权只依赖终值，可以直接抽样，避免欧拉离散误差。
Z = rng.normal(size=N)
ST = S0*np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*Z)
call_exact = np.exp(-r*T)*np.maximum(ST-K, 0)
put_exact = np.exp(-r*T)*np.maximum(K-ST, 0)
d1 = (np.log(S0/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
d2 = d1-sigma*np.sqrt(T)
bs_call = S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
bs_put = K*np.exp(-r*T)*norm.cdf(-d2)-S0*norm.cdf(-d1)
print('直接抽样看涨/看跌:', call_exact.mean(), put_exact.mean())
print('解析公式对照:', bs_call, bs_put)
