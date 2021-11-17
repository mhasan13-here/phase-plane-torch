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
    def __init__(self, dt=1e-6, Tsim=1e-3, Cv=50e-12, Cu=30e-12, record=True):
        super(SpikingNeuron, self).__init__()
        
        self.dt = dt
        self.Cv = Cv
        self.Cu = Cu
        self.Tsim = Tsim
        self.beta = dt/Cv # assuming R=1 => tau = RC
        self.record = record
        
        self.leak_mem = 0.95
        self.threshold = 0.3

    def forward(self, input, num_steps):
        # See the autograd section for explanation of what happens here.
        
        if num_steps == 0:
            self.batch_size = input.size(0)
            self.v = torch.zeros( input.size() )
            self.spikes = torch.zeros( input.size() )
            
            if self.record:
                self.v_t = []
                self.s_t = []
        

        self.v = self.leak_mem*self.v + (1-self.leak_mem)*(input)
        # self.spikes[self.v > self.threshold] = 1. # <== spikes needs to be init to 0 every timestep when using this
        self.spikes = (self.v > self.threshold) * 1
        self.v[self.v > self.threshold] = 0.
        
        if self.record:
            self.v_t.append(self.v)
            self.s_t.append(self.spikes)
        
        return self.spikes, self.v

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'Spiking Neuron Layer'
    
    
class Net(nn.Module):
    def __init__(self, dt=1e-6, Tsim=1e-3, Cv=50e-12, Cu=30e-12, record=True):
        super(Net, self).__init__()
        self.dt = dt
        self.Tsim = Tsim
        self.num_steps = int(Tsim/dt) + 1 
        
        # define neural network layers
        self.spike_layer = SpikingNeuron(dt, Tsim, Cv, Cu, record)

    def forward(self, input):
        for t in range(self.num_steps):
            s, v = self.spike_layer(input, t)
        
        return self.spike_layer.v_t, self.spike_layer.s_t
        




input = torch.tensor([[1.]])
net = Net(dt=1e-3, Tsim=40e-3)
v_t, s_t = net(input)

v_t = [v[0,0] for v in v_t]
s_t = [s[0,0] for s in s_t]
plt.plot(v_t)
plt.plot(s_t)



