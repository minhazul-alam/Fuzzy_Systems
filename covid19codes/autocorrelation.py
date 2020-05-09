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

from statsmodels.tsa.stattools import acf

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])

ax.plot(acf(y,nlags=14))
ax.set_title("Autocorrelation")
ax.set_ylabel("ACF")
ax.set_xlabel("LAG")

