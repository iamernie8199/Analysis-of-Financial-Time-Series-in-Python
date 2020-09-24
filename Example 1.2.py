import pandas as pd
import scipy.stats as stat
from math import sqrt
import numpy as np


def basicStats(x):
    return pd.Series([
        x.count(), x.isna().sum(), x.min(), x.max(),
        round(x.quantile(.25), 6),
        round(x.quantile(.75), 6),
        round(x.mean(), 6),
        round(x.median(), 6),
        round(x.sum(), 6),
        round(x.sem(), 6),
        round(stat.bayes_mvs(x, alpha=0.95)[0].minmax[0], 6),
        round(stat.bayes_mvs(x, alpha=0.95)[0].minmax[1], 6),
        round(x.var(), 6), round(x.std(), 6),
        round(x.skew(), 6), round(x.kurt(), 6)
    ], index=[
        '總計', 'NAs', '最小值', '最大值',
        '25%分位數', '75%分位數', '均值', '中位數',
        '總和', '平均值標準誤差',
        "LCL Mean", "UCL Mean",
        '方差', '標準差', '偏度', '超額峰度'
    ])


ibm = pd.read_csv('data/d-ibm3dx7008.txt', delimiter='\s+')
print(ibm.shape)
print(ibm.head(1))
sibm = ibm['rtn'] * 100
# Compute the summary statistics
print(basicStats(sibm))
# Simple tests
# Compute test statistic
s1 = sibm.skew()
t1 = s1 / sqrt(6 / sibm.count())
# t1 = stat.skewtest(sibm).statistic
print(round(t1, 6))
# Compute p-value
"""
The equivalent of the R pnorm() function is: scipy.stats.norm.cdf() with python 
The equivalent of the R qnorm() function is: scipy.stats.norm.ppf() with python
"""
pv = 2 * (1 - stat.norm.cdf(t1))
# pv = stat.skewtest(sibm).pvalue
print(round(pv, 6))

# Turn to log returns in percentages
libm = np.log(ibm['rtn'] + 1) * 100
# Test mean being zero
t2, pv2 = stat.ttest_1samp(libm, 0)
print('One sample t-test:')
print(f't = {t2}, p-value = {pv2}')

# Normality test
print('Normality test:')
Xsquared, pv3 = stat.jarque_bera(libm)
print('STATISTIC:')
print(f'X-squared: {round(Xsquared, 4)}')
print('P VALUE:')
print(f'Asymptotic p Value: {pv3}')
