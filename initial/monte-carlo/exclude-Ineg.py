#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 04:19:17 2021

@author: mhasan13
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

#########
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.rc('lines', linewidth=1.5)
#########


# cadence monte carlo
df = pd.read_csv('MonteCarlo.Ineg.csv', header=None)
data = df[3].values

# plt.subplot(1,2,1)
n, bins, patches = plt.hist(data, 10, alpha = 0.3, color='g', label='Cadence')
plt.plot(np.diff(bins)/2 + bins[:-1], n, 'o-', color='g', label = 'Cadence')

# torch monte carlo
with open('exclude-Ineg.pkl', 'rb') as fp:
    data = pkl.load(fp)

# plt.subplot(1,2,2)
n_, bins_, patches_ = plt.hist(data, 10, alpha = 0.3, color='r', label='Phase Plane')
plt.plot(np.diff(bins_)/2 + bins_[:-1], n_, '*-', color='r', label='Phase Plane')
plt.legend()
plt.xlabel('Spike Frequency')
plt.ylabel('No. of Samples')

plt.show()

plt.figure()
plt.plot(np.diff(bins)/2 + bins[:-1], n, 'o-', color='g', label = 'Cadence')
plt.plot(np.diff(bins_)/2 + bins_[:-1], n_, '*-', color='r', label='Phase Plane')
plt.legend()
plt.xlabel('Spike Frequency')
plt.ylabel('No. of Samples')
plt.show()