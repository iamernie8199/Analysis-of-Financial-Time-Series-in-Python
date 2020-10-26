import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


def basicStats(x):
    return pd.Series([
        x.count(), x.isna().sum(), x.min(), x.max(),
        round(x.quantile(.25), 6),
        round(x.quantile(.75), 6),
        round(x.mean(), 6),
        round(x.median(), 6),
        round(x.sum(), 6),
        round(x.sem(), 6),
        round(stat.bayes_mvs(x, alpha=0.95)[0].minmax[0], 6),
        round(stat.bayes_mvs(x, alpha=0.95)[0].minmax[1], 6),
        round(x.var(), 6), round(x.std(), 6),
        round(x.skew(), 6), round(x.kurt(), 6)
    ], index=[
        '總計', 'NAs', '最小值', '最大值',
        '25%分位數', '75%分位數', '均值', '中位數',
        '總和', '平均值標準誤差',
        "LCL Mean", "UCL Mean",
        '方差', '標準差', '偏度', '超額峰度'
    ])


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
