import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def ljungbox(data, lags=12):
    blres = acorr_ljungbox(data, lags=lags, return_df=True)
    print("Box-Ljung test")
    print(f"X-squared: {round(blres.tail(1)['lb_stat'].values[0], 4)}", end=", ")
    print(f"df = {len(blres)}", end=", ")
    print(f"p-value: {blres.tail(1)['lb_pvalue'].values[0]}")


def engle(data, lags=12):
    arch = het_arch(data, nlags=lags)
    print("Lagrange multiplier test")
    print(f"Chi-squared: {round(arch[0], 4)}", end=", ")
    print(f"p-value: {arch[1]}")


da = pd.read_csv('../data/m-intc7308.txt', delimiter='\s+')
intc = np.log(da['rtn'] + 1)
ljungbox(intc)

at = intc - intc.mean()
ljungbox(pow(at, 2))

# Engle’s Test
engle(intc)
