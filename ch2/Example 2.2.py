from math import log
from statsmodels.tsa.ar_model import AR

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as scs


def tsplot(y, lags=None, figsize=(10, 8), style='seaborn'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5, method='ols')
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
        plt.show()


da = pd.read_csv('../data/q-gdp4708.txt', delimiter='\s+')
da['ln(gdp)'] = da['gdp'].apply(lambda x: log(x))
da['diff-ln(gdp)'] = da['ln(gdp)'].diff()
gdp = da['ln(gdp)']
tsplot(gdp)
gdpdif = gdp.diff()[1:]
tsplot(gdpdif)

t = sm.tsa.stattools.adfuller(gdp, maxlag=10, autolag=None)
output = pd.DataFrame(index=['Lags Used', 'Dickey-Fuller', "p-value"], columns=['value'])
output['value']['Lags Used'] = t[2]
output['value']['Dickey-Fuller'] = t[0]
output['value']['p-value'] = t[1]
print(output)

da2 = pd.read_csv('../data/d-sp55008.txt', delimiter='\s+')
sp5 = da2['close'].apply(lambda x: log(x))
tsplot(sp5)
sp5dif = sp5.diff()[1:]
tsplot(sp5dif)
# order = AR(sp5dif).select_order(maxlag=20, ic='aic', trend='c', method='mle')
t = sm.tsa.stattools.adfuller(sp5, regression='ct', maxlag=15, autolag=None)
output = pd.DataFrame(index=['Lags Used', 'Dickey-Fuller', "p-value"], columns=['value'])
output['value']['Lags Used'] = t[2]
output['value']['Dickey-Fuller'] = t[0]
output['value']['p-value'] = t[1]
print(output)
