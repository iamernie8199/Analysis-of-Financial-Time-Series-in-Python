import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

ibm = pd.read_csv('../data/m-ibm3dx2608.txt', delimiter='\s+')
print(ibm.shape)
print(ibm.head(1))
# IBM simple returns
sibm = ibm['ibmrtn']

# Ljung-Box statistic Q(5)
sresult = acorr_ljungbox(sibm, lags=5, return_df=True)
print("IBM simple returns Box-Ljung test")
print(f"X-squared: {round(sresult.tail(1)['lb_stat'].values[0], 4)}", end=" ")
print(f"p-value: {round(sresult.tail(1)['lb_pvalue'].values[0], 4)}")

# Log IBM returns
libm = np.log(sibm + 1)
lresult = acorr_ljungbox(libm, lags=5, return_df=True)
print("Log IBM returns Box-Ljung test")
print(f"X-squared: {round(lresult.tail(1)['lb_stat'].values[0], 4)}", end=" ")
print(f"p-value: {round(lresult.tail(1)['lb_pvalue'].values[0], 4)}")
