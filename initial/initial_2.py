# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:42:08 2021

@author: mhasan13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SpikingActivation(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad
    
def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

x = torch.tensor(1.0)
y = torch.zeros(10)
tt = torch.arange(10)
for i in range(10):
    y[i] = PoissonGen(x)


class SpikingNeuron(nn.Module):
    def __init__(self, dt=1e-6,  Tsim=2e-3, Cv=50e-12, Cu=30e-12):
        super(SpikingNeuron, self).__init__()
        
        self.dt = dt
        self.Cv = Cv
        self.Cu = Cu
        self.Tsim = Tsim
        self.num_steps = int(Tsim/dt) + 1 
        self.leak_mem = 0.95
        self.threshold = 0.3
        
    def forward(self, input):
        
        self.batch_size = input.size(0)
        self.v = torch.zeros( input.size() )
        self.spikes = torch.zeros( input.size() )
        
        self.v_t = []
        self.s_t = []
        
        for t in range(self.num_steps):
            poission_input = PoissonGen(input)
            self.v = self.leak_mem*self.v + (1-self.leak_mem)*(input)
            # self.spikes[self.v > self.threshold] = 1. # <== spikes needs to be init to 0 every timestep when using this
            self.spikes = (self.v > self.threshold) * 1
            self.v[self.v > self.threshold] = 0.
            
            self.v_t.append(self.v)
            self.s_t.append(self.spikes)
        
        return self.spikes, self.v, self.v_t, self.s_t


input = torch.tensor([[1.]])
net = SpikingNeuron(dt=1e-3, Tsim=40e-3)
s , v, v_t, s_t = net(input)

v_t = [v[0,0] for v in v_t]
s_t = [s[0,0] for s in s_t]
plt.plot(v_t)
plt.plot(s_t)


