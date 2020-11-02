import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

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
