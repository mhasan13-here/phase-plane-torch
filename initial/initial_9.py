#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 04:15:05 2021

@author: mhasan13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

csv_path = 'neuron_tran.csv'
file = pd.read_csv(csv_path, header=0)
data = file.values
t = torch.tensor(data[:,0])
v = torch.tensor(data[:,1])
u = torch.tensor(data[:,3])
Iv = torch.tensor(data[:,5])
Iu = torch.tensor(data[:,7])


#################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 12345
random.seed(seed)
torch.manual_seed(seed)

x1 = v[0:-1]
x2 = Iv[0:-1]
y = v[1:].unsqueeze(1).float()
X = torch.vstack((x1,x2)).T.float()
scale = 1e-3

N = X.size(0)  # num_samples_per_class
D = X.size(1)  # dimensions
C = 1  # num_classes
H = 1  # num_hidden_units



learning_rate = 1e-2

# nn package to create our linear model
# each Linear module has a weight and bias
model = nn.Sequential(
        nn.Linear(D, C, bias=False)
    )
model.to(device) # Convert to CUDA

# nn package also has different loss functions.
# we use MSE loss for our regression task
criterion = torch.nn.MSELoss()

# we use the optim package to apply
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
epoch = 100000
for i in range(epoch):

    # Feed forward to get the logits
    y_pred = model(X)
    
    # Compute the loss (MSE)
    loss = criterion(y_pred, y)
    print("[EPOCH]: %i, [LOSS or MSE]: %.10f" % (i, loss.item()))

    
    # zero the gradients before running
    # the backward pass.
    optimizer.zero_grad()
    
    # Backward pass to compute the gradient
    # of loss w.r.t our learnable params. 
    loss.backward()
    
    
    # Update params
    optimizer.step()
    
########################
t = torch.tensor(data[:,0])
dt = (t[1:]-t[0:-1])[0] # uniform timesteps
weights = model[0].weight[0]
tau = dt/(1 - weights[0]) # w1 = (1 - dt/tau)
R = weights[1]*tau/dt

y = np.zeros(v.size())
for i in range(1, t.size(0)):
    y[i] = weights[0]*y[i-1] + weights[1]*Iv[i-1]

