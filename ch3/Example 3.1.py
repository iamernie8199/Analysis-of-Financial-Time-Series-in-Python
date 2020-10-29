import pandas as pd
import numpy as np
from arch import arch_model


da = pd.read_csv('../data/m-intc7308.txt', delimiter='\s+')
intc = np.log(da['rtn'] + 1)
arch3 = arch_model(intc,vol='ARCH',p=3).fit()
print(arch3.summary())

arch1 = arch_model(intc,vol='ARCH',p=1).fit()
print(arch1.summary())