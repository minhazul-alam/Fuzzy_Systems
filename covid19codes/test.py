# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pyFTS.common import Util
from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Entropy
from pyFTS.models import hofts, pwfts
from pyFTS.common import Membership

os.chdir('F:\\Minhaz\\MS@NSU\\Fuzzy\\Project_Drive_GIT\\Fuzzy_Systems\\covid19codes')

dataset = pd.read_csv('time_series_covid19_confirmed_hubei.csv', sep=',')
dataset = pd.read_csv('time_series_covid19_confirmed_global.csv', sep=',')

y = dataset.to_numpy()[0]

rows = []

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,7])
ax.plot(y, label="Original Data", color="Black")

part = Grid.GridPartitioner(data=y, npart=80)

model = hofts.HighOrderFTS(order=2, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
plt_fc = np.insert(forecasts, 0, y[0])
ax.plot(plt_fc, label="HOFTS")
rmse, mape, u = Measures.get_point_statistics(y, model)
rows.append(["HOFTS", rmse, mape, u])

model = hofts.WeightedHighOrderFTS(order=2, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
plt_fc = np.insert(forecasts, 0, y[0])
ax.plot(plt_fc, label="WHOFTS")
rmse, mape, u = Measures.get_point_statistics(y, model)
rows.append(["WHOFTS", rmse, mape, u])

model = pwfts.ProbabilisticWeightedFTS(order=2, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
plt_fc = np.insert(forecasts, 0, y[0])
ax.plot(plt_fc, label="PWFTS")
rmse, mape, u = Measures.get_point_statistics(y, model)
rows.append(["PWFTS", rmse, mape, u])

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

df = pd.DataFrame(rows, columns=['Algorithms','RMSE','MAPE','U'])
print(df)














