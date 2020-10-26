import pandas as pd
from util import tsplot, tsdiag
import statsmodels.api as sm

da = pd.read_csv('../data/m-deciles08.txt', delimiter='\s+')
d1 = da['CAP1RET']
tsplot(d1)
jan = da['date'].apply(lambda x: 1 if x % 10000 // 100 == 1 else 0)
mod = sm.OLS(d1, jan)
res = mod.fit()
print("formula = d1 ~ jan")
print(res.summary())

mod = sm.tsa.arima.ARIMA(d1, order=(1, 0, 0), seasonal_order=(1, 0, 1, 12))
res = mod.fit()
print(res.summary())
tsplot(res.resid)
tsdiag(res.resid, lags = 36)

mod = sm.tsa.arima.ARIMA(d1, order=(1, 0, 0), seasonal_order=(1, 0, 1, 12))
res = mod.fit()
print(res.summary())
