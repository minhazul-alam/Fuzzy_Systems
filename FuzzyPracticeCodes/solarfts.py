import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
#import matplotlib as plt
import matplotlib.pyplot as plt

from pyFTS.common import Util
from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Entropy
from pyFTS.models import hofts
from pyFTS.common import Membership

dataset = pd.read_csv('https://query.data.world/s/2bgegjggydd3venttp3zlosh3wpjqj', sep=';')
dataset['data'] = pd.to_datetime(dataset["data"], format='%Y-%m-%d %H:%M:%S')

train_uv = dataset['glo_avg'].values[:24505]
test_uv = dataset['glo_avg'].values[24505:]

train_mv = dataset.iloc[:24505]
test_mv = dataset.iloc[24505:]

dataset.head()

models = []
fig, ax = plt.subplots(nrows=2, ncols=1,figsize=[20,5])

ax[0].plot(train_uv[:240])
ax[1].plot(train_uv)

from statsmodels.tsa.stattools import acf

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])

ax.plot(acf(train_uv,nlags=48))
ax.set_title("Autocorrelation")
ax.set_ylabel("ACF")
ax.set_xlabel("LAG")

from itertools import product

levels = ['VL','L','M','H','VH']
sublevels = [str(k) for k in np.arange(0,7)]
names = []
for combination in product(*[levels, sublevels]):
  names.append(combination[0]+combination[1])
  
print(names)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,3])

part = Grid.GridPartitioner(data=train_uv,npart=35, names=names)

part.plot(ax)

















