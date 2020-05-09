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
from pyFTS.models import hofts
from pyFTS.common import Membership

os.chdir('F:\\Minhaz\\MS@NSU\\Fuzzy\\Project_Drive_GIT\\Fuzzy_Systems\\covid19codes')

dataset = pd.read_csv('time_series_covid19_confirmed_hubei.csv', sep=',')
y = dataset.to_numpy()[0]

rows = []

#y = y[1:] - y[:-1]
# use of cumulative data vs. change
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,10])
ax.plot(y, label="Original", color="Black")

part = Grid.GridPartitioner(data=y, npart=30)
model = hofts.HighOrderFTS(order=1, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
plt_fc = np.insert(forecasts, 0, y[0])
ax.plot(plt_fc, label="npart=30", color="Red")
rmse, mape, u = Measures.get_point_statistics(y, model)
rows.append([30, rmse, mape, u])

part = Grid.GridPartitioner(data=y, npart=40)
model = hofts.HighOrderFTS(order=1, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
plt_fc = np.insert(forecasts, 0, y[0])
ax.plot(plt_fc, label="npart=40", color="Blue")
rmse, mape, u = Measures.get_point_statistics(y, model)
rows.append([40, rmse, mape, u])

part = Grid.GridPartitioner(data=y, npart=50)
model = hofts.HighOrderFTS(order=1, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
plt_fc = np.insert(forecasts, 0, y[0])
ax.plot(plt_fc, label="npart=50", color="Green")
rmse, mape, u = Measures.get_point_statistics(y, model)
rows.append([50, rmse, mape, u])

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

df = pd.DataFrame(rows, columns=['Partitions','RMSE','MAPE','U'])
print(df)

'''model = hofts.HighOrderFTS(order=2, partitioner=part)
model.fit(y)
forecasts = model.predict(y)
ax.plot(forecasts[:-1], label="Order=2", color="Blue")

model = hofts.HighOrderFTS(order=3, partitioner=part)
model.fit(y)
forecasts = model.predict(y, label="Order=3", color="Green")
ax.plot(forecasts[:-1])'''













