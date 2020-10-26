import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from scipy.stats import chi2

ibm = pd.read_csv('../data/m-ibm3dx2608.txt', delimiter='\s+')
vw = ibm['vwrtn']
m3 = ARIMA(vw, order=(3, 0, 0))
res = m3.fit()
print(res.summary())
print(
    f"intercept phi(0): {(res.polynomial_ar[0] + res.polynomial_ar[1] + res.polynomial_ar[2] + res.polynomial_ar[3]) * vw.mean()}")
# standard error of residuals
print(f"standard error of residuals: {sqrt(res.params['sigma2'])}")
# box test
print("Box-Ljung:")
box = sm.stats.acorr_ljungbox(res.resid, lags=[12], return_df=True)
pv = 1 - chi2.cdf(box['lb_stat'], 9)
print(f"pv: {pv[0]}")

m3 = ARIMA(vw, order=(3, 0, 0), enforce_stationarity=False)
with m3.fix_params({'ar.L2': 0}):
    res = m3.fit()
print(res.summary())
print(
    f"intercept phi(0): {(res.polynomial_ar[0] + res.polynomial_ar[1] + res.polynomial_ar[2] + res.polynomial_ar[3]) * vw.mean()}")
print(f"standard error of residuals: {sqrt(res.params['sigma2'])}")
print("Box-Ljung:")
box = sm.stats.acorr_ljungbox(res.resid, lags=[12], return_df=True)
print(box)
pv = 1 - chi2.cdf(box['lb_stat'], 10)
print(f"pv: {pv[0]}")