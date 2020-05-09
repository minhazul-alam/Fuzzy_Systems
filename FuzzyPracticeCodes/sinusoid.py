# -*- coding: utf-8 -*-
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

x = [k for k in np.arange(-2*np.pi, 2*np.pi, 0.1)]
y = [np.sin(k) for k in x]

rows = []

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])

ax.plot(y, label='Original',color='black')

for npart in np.arange(5,35,5):
  part = Grid.GridPartitioner(data=y, npart=npart)
  model = hofts.HighOrderFTS(order=1, partitioner=part)
  model.fit(y)
  forecasts = model.predict(y)
    
  ax.plot(forecasts[:-1], label=str(npart) + " partitions")
  
  rmse, mape, u = Measures.get_point_statistics(y, model)
  
  rows.append([npart, rmse, mape, u])
  

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

df = pd.DataFrame(rows, columns=['Partitions','RMSE','MAPE','U'])

