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
plt.rc('font', family='Times')
plt.rc('font', size=20, weight='bold')
plt.rc('lines', linewidth=1.5)
plt.rc('axes', labelweight='bold')
#########


# cadence monte carlo
df = pd.read_csv('MonteCarlo.Ir.csv', header=None)
data = df[3].values

# plt.subplot(1,2,1)
n, bins, patches = plt.hist(data, 9, alpha = 0.3, color='g', label='Cadence')
plt.plot(np.diff(bins)/2 + bins[:-1], n, 'o-', color='g', label = 'Cadence')

# torch monte carlo
with open('exclude-Ir.pkl', 'rb') as fp:
    data = pkl.load(fp)

# plt.subplot(1,2,2)
n_, bins_, patches_ = plt.hist(data, 13, alpha = 0.3, color='r', label='Phase Plane')
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
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
plt.show()