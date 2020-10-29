import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

da = pd.read_csv('../data/m-intc7308.txt', delimiter='\s+')
intc = np.log(da['rtn'] + 1)
arch3 = arch_model(intc, vol='ARCH', p=3).fit()
print(arch3.summary())

arch1 = arch_model(intc, vol='ARCH', p=1).fit()
print(arch1.summary())

stres = arch1.std_resid
blres = acorr_ljungbox(stres, lags=10, return_df=True)
print("Box-Ljung test")
print(f"X-squared: {round(blres.tail(1)['lb_stat'].values[0], 4)}", end=", ")
print(f"df = {len(blres)}", end=", ")
print(f"p-value: {blres.tail(1)['lb_pvalue'].values[0]}")

print(arch1.arch_lm_test(lags=10, standardized=True))

arch1.plot()
plt.show()