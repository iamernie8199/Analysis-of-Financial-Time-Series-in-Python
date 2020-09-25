import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pi, acos
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from numpy.polynomial.polynomial import polyroots

# Load data
gnp = pd.read_csv('data/dgnp82.txt', delimiter='\s+', header=None, names=['gnp'])
# create a time-series object
gnp = pd.DataFrame({"gnp": gnp['gnp'].to_list()}, index=pd.date_range(start='1947-05', freq='Q', periods=len(gnp)))
# plot
gnp.plot()
plt.show()
# Find the AR order
m1 = ar_select_order(gnp, maxlag=13, ic='aic')
print(f"AR order: {m1.ar_lags[-1]}")
m2 = ARIMA(gnp, order=(m1.ar_lags[-1], 0, 0))
res = m2.fit()
# Estimation
print(res.summary())

# ‘‘const’’ denotes the mean of the series.
# Therefore, the constant term is obtained below:
tmp = 1
for i in range(1, len(res.params) - 1):
    tmp -= res.params[i]
const = res.params[0] * tmp
print(f"const: {const}")
# Residual standard error
print(f"Residual standard error: {sqrt(res.params[-1])}")
# Characteristic equation
p1 = [1] + res.params[1:-1].apply(lambda x: -x).to_list()
# Find solutions
roots = polyroots(p1).tolist()
print(roots)
for r in roots:
    if not r.imag: roots.remove(r)
# the absolute values of the solutions
abs_roots = [abs(r) for r in roots]
# To compute average length of business cycles
k = 2 * pi / acos(roots[0].real / abs_roots[0].real)
print(f"k: {k}")