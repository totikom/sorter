#!/bin/python
import sys
import scipy.fft as fft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame

print("Reading data from:",sys.argv)

data = pd.read_csv(sys.argv[1], index_col=0,usecols=[0,1,2],skiprows=[1])
data_params =  pd.read_csv(sys.argv[1], usecols=[3,4],nrows=1)

spectrum = fft.rfft(data.CH1.values)

data_params.Increment[0]
frequencies = fft.rfftfreq(data.shape[0], data_params.Increment[0])
plt.plot(frequencies, np.abs(spectrum)**2)
plt.grid()
plt.show()