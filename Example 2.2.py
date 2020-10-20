from math import log
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

da = pd.read_csv('data/q-gdp4708.txt', delimiter='\s+')
da['ln(gdp)'] = da['gdp'].apply(lambda x: log(x))
da['diff-ln(gdp)'] = da['ln(gdp)'].diff()
da.plot(x ='year', y='ln(gdp)')
plt.show()
gdp = da['ln(gdp)']
tsplot(gdp)
gdpdif = gdp.diff()[1:]
tsplot(gdpdif)

t = sm.tsa.stattools.adfuller(gdp,maxlag=10,autolag=None)
output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used"], columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
print(output)