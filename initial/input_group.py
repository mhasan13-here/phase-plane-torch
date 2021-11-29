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
import pickle as pkl
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

start_time = time.time()

class NeuronMeshGrid:
    """
    Data on neuron phase plane. The neuron circuit used for phase plane 
    extraction is from this paper.
    https://ieeexplore.ieee.org/abstract/document/9184447
    """
    def __init__(self, pickle_path:str) -> None:
        with open(pickle_path, 'rb') as fp:
            itemlist = pkl.load(fp)
        
        data = torch.tensor(itemlist)
        self.u = data[0,:,:]
        self.v = data[1,:,:]
# =============================================================================
#         -ve sign has to be fixed for pmos currents now
#         as cadence introduced a -ve sign for outgoing current
# =============================================================================
        self.iCv = -data[2,:,:] - data[3,:,:] - data[6,:,:] # Ipos_feed - Ineg_feed - I_leak
        self.iCu = -data[4,:,:] - data[5,:,:] # Iw - Ir
        self.axon = data[7,:,:] # axon output
# =============================================================================
#         i=>y axis index, j=>x axis index
# =============================================================================
        self.vmax, self.vmin = torch.max(self.v), torch.min(self.v)
        self.umax, self.umin = torch.max(self.u), torch.min(self.u)
        self.j_per_x = (self.v.shape[1]-1)/(self.vmax-self.vmin)
        self.i_per_y = (self.u.shape[0]-1)/(self.umax-self.umin)
# =============================================================================
#         neuron capacitance
# =============================================================================
        self.Cv = 50e-15
        self.Cu = 30e-15
        self.Cp = 5e-15
        

    
def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

    
class InputSpikingActivation(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, v, t_period, pulse_width):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        spikes = torch.zeros(v.shape)
        spikes = (v > t_period) * 1
        v[v > t_period + pulse_width] = pulse_width
        
        ctx.save_for_backward(spikes, v)
        return spikes, v

    @staticmethod
    def backward(ctx, grad_output): ####### not implemented yet
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        spikes, v = ctx.saved_tensors
        
        return v
    

class InputGroup(nn.Module):
    """
    This module implements input spikes to the network
    """
    def __init__(self, dt=1e-6, Tsim=1e-3, record=False):
        super(InputGroup, self).__init__()

        self.spike_activation = InputSpikingActivation.apply
        self.dt = dt
        self.Tsim = Tsim
        self.pulse_width = 45e-6
        self.record = record
        
        # create recording variable
        if self.record:
            self.v_t = []
            self.s_t = []
        
        

    def forward(self, input, num_steps):
        # See the autograd section for explanation of what happens here.

        if num_steps == 0: ### this is here to initilize the variables at start
            self.batch_size = input.size(0)
            self.v = torch.zeros( input.size() )
            self.spikes = torch.zeros( input.size() )
            self.t_period = 1/(input+1e-15)
        
        self.v = self.v + self.dt
        self.spikes, self.v = self.spike_activation(self.v, self.t_period, self.pulse_width)
        
        if self.record:
            self.v_t.append(self.v)
            self.s_t.append(self.spikes)
        
        return self.spikes, self.v

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'Input Neuron Layer'
    
    
class Net(nn.Module):
    def __init__(self, dt=1e-6, Tsim=1e-3):
        super(Net, self).__init__()
        self.dt = dt
        self.Tsim = Tsim
        self.num_steps = int(Tsim/dt) + 1 
        
        # define neural network layers
        self.spike_layer = InputGroup(dt, Tsim, record=True)

    def forward(self, input):
        for t in range(self.num_steps):
            s, v = self.spike_layer(input, t)
        
        return self.spike_layer.v_t, self.spike_layer.s_t
        



Tsim = 50e-3
input = torch.tensor([[100.]])
net = Net(dt=1e-6, Tsim=Tsim)
v_t, s_t = net(input)

v_t = [v[0,0] for v in v_t]
s_t = [s[0,0] for s in s_t]
plt.plot(v_t)
plt.plot(s_t)
plt.show()
end_time = time.time()
print(end_time-start_time)





