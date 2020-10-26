import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from util import tsplot, tsdiag
import matplotlib.pyplot as plt

r1 = pd.read_csv('data/w-gs1yr.txt', delimiter='\s+')['rate']
r3 = pd.read_csv('data/w-gs3yr.txt', delimiter='\s+')['rate']
# add constant for fit
r = sm.add_constant(r1)
# r_3t = α + β*r_1t + e_t
res1 = sm.OLS(r3, r).fit()
print('formula = r3 ~ r1')
print(res1.summary())
tsplot(res1.resid)

c1 = round(r1.diff()[1:].reset_index(drop=True), 2)
c3 = round(r3.diff()[1:].reset_index(drop=True), 2)
# c_3t = β*c_1t + et
res2 = sm.OLS(c3, c1).fit()
print('formula = c3 ~ -1 + c1')
print(res2.summary())
tsplot(res2.resid)

res3 = sm.tsa.arima.ARIMA(c3, c1, order=(0, 0, 1)).fit()
print(res3.summary())
rsq = (c3.apply(lambda x: pow(x, 2)).sum() - res3.resid.apply(lambda x: pow(x, 2)).sum()) / c3.apply(
    lambda x: pow(x, 2)).sum()
