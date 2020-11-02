import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from util import ljungbox
import arch

da = pd.read_fwf('../data/sp500.txt')
da.plot()
plt.show()

# fit ma(3)
MA = ARIMA(da, order=(0, 0, 3)).fit()
print(MA.summary())
# set L2 to 0 according to p-value
MA3 = ARIMA(da, order=(0, 0, 3), enforce_invertibility=False)
with MA3.fix_params({'ma.L2': 0}):
    res = MA3.fit()
print(res.summary())
print(f"sigma: {res.params['sigma2'] ** 0.5}")

# fit ar(3)
ar = ARIMA(da, order=(3, 0, 0)).fit()
print(ar.summary())

# fit a GARCH(1,1) model on the residuals of the ar(3) model
garch = arch.arch_model(ar.resid, p=1, q=1).fit(update_freq=0)
print(garch.summary())
# the implied unconditional variance of a_t
variance = garch.params['omega'] / (1 - garch.params['alpha[1]'] - garch.params['beta[1]'])
print(f"unconditional variance of a_t: {round(variance, 5)}")

# dropping all AR parameters
ar0 = ARIMA(da, order=(0, 0, 0)).fit()
print(ar0.summary())
garch = arch.arch_model(ar0.resid, p=1, q=1).fit(update_freq=0)
print(garch.summary())
# the implied unconditional variance of a_t
variance = garch.params['omega'] / (1 - garch.params['alpha[1]'] - garch.params['beta[1]'])
print(f"unconditional variance of a_t: {round(variance, 5)}")
# autocorTest
stres = garch.std_resid
ljungbox(stres, lags=24)
ljungbox(stres**2, lags=24)
# alpha + beta
print(f"alpha + beta = {garch.params['alpha[1]'] + garch.params['beta[1]']}")
# Obtain 1 to 5-step predictions
print('Return:')
print(ar0.forecast(5))
print('Volatility:')
print(garch.forecast(horizon=5).variance.tail(1) ** 0.5)
