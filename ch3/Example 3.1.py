import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from math import sqrt

da = pd.read_csv('../data/m-intc7308.txt', delimiter='\s+')
intc = np.log(da['rtn'] + 1)
arch3 = arch_model(intc, vol='ARCH', q=3).fit()
print(arch3.summary())

arch1 = arch_model(intc, vol='ARCH', q=1).fit()
print(arch1.summary())

stres = arch1.std_resid
blres = acorr_ljungbox(stres, lags=10, return_df=True)
print("Box-Ljung test")
print(f"X-squared: {round(blres.tail(1)['lb_stat'].values[0], 4)}", end=", ")
print(f"df = {len(blres)}", end=", ")
print(f"p-value: {blres.tail(1)['lb_pvalue'].values[0]}")

print(arch1.arch_lm_test(lags=10, standardized=True))

# unconditional standard error
print(f"unconditional standard error: {sqrt(arch1.params[1] / (1 - arch1.params[2]))}")
# fitted volatility series
arch1.plot()
plt.show()
# Obtain 1 to 5-step predictions
predict = arch1.forecast(horizon=5)
print(predict.mean.tail(1))
print(predict.variance.tail(1))
print(predict.residual_variance.tail(1))
# Students’s t
arch1t = arch_model(intc, vol='ARCH', q=1, dist='t').fit()
print(arch1t.summary())
# fits a GARCH(1,1)
garch1 = arch_model(intc, vol='GARCH', p=1, q=1).fit()
print(garch1.summary())
# Skewed Student’s t
arch1st = arch_model(intc, vol='ARCH', q=1, dist='skewt').fit()
print(arch1st.summary())