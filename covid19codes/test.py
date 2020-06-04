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

#dataset = pd.read_csv('time_series_covid19_confirmed_hubei.csv', sep=',')

#Apple Mobility - 13th January - 26th April
am_dataset = pd.read_csv('Mobility/mobility_apple.csv', sep=',')
#countries = list(set(dataset['region']))
#braz_data = am_dataset[am_dataset['region']=='Brazil']
braz_data = am_dataset[am_dataset['region']=='Italy']
braz_driving = np.array((braz_data[braz_data['transportation_type']=='driving'])['value'])
braz_walking = np.array((braz_data[braz_data['transportation_type']=='walking'])['value'])
#normalizing
braz_driving /= max(braz_driving)
braz_walking /= max(braz_walking)
braz_mobility = (braz_driving + braz_walking)/2.0 #same weight

#Google Mobility dataset has to be incorporated here


#Covid Infection data - 22nd January - 9th May
cv_dataset = pd.read_csv('time_series_covid19_confirmed_global.csv', sep=',')
cv_Brazil = cv_dataset[cv_dataset['Country/Region']=='Italy'].to_numpy()[0][4:]
cv_Brazil_change = cv_Brazil[1:] - cv_Brazil[:-1]
#normalizing
cv_Brazil /= max(cv_Brazil)
cv_Brazil_change /= max(cv_Brazil_change)

# hypothesizing corrrelation upto 7 days ---
# 23 - 7 = 16th January braz_mobility[3:] 
# 26 Apr + 7 = 3rd May cv_Brazil_change[:-6]

#16th Jan - 3rd May - 102 days
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])
ax.plot(braz_mobility[3:], label="Mobility Trend")
ax.plot(cv_Brazil_change[:-6], label="Corona Trend")
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

#Pearson Correlation (-1 -- +1)
corr = np.corrcoef(braz_mobility[73:], cv_Brazil_change[70:-6].astype('float64'))
print(corr)

corr = np.corrcoef(braz_driving, braz_walking)

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














