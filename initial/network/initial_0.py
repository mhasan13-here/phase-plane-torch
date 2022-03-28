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


    
def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

    
class SpikingActivation(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # threshold = 0.
        # spikes = torch.zeros(input.shape)
        # spikes = (input > threshold) * 1
        # input[input > threshold] = 0.
        
        spikes = torch.zeros(input.shape)
        mem_thr = (input/threshold) - 1.0
        spikes = (mem_thr > 0) * 1.0
        rst = torch.zeros(input.shape)
        rst[mem_thr > 0] = threshold
        input = input - rst
        
        ctx.save_for_backward(input, spikes)
        return spikes, input

    @staticmethod
    def backward(ctx, grad_output): 
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        
        Backpropagation implemented from 
        https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time.git
        """
        spikes, input = ctx.saved_tensors
        
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        
        return grad

    

class SpikingNeuron(nn.Module):
    def __init__(self, dt=1e-6, Tsim=1e-3, Cv=50e-12, Cu=30e-12, record=True):
        super(SpikingNeuron, self).__init__()
        
        self.spike_activation = SpikingActivation.apply
        self.dt = dt
        self.Cv = Cv
        self.Cu = Cu
        self.Tsim = Tsim
        self.beta = dt/Cv # assuming R=1 => tau = RC
        self.record = record
        
        
        self.leak_mem = 0.95
        self.threshold = 0.3

    def forward(self, input, num_steps, conv, bntt):
        # See the autograd section for explanation of what happens here.

        if num_steps == 0:
            self.batch_size = input.size(0)
            self.v = torch.zeros( input.size() )
            self.spikes = torch.zeros( input.size() )
            
            if self.record:
                self.v_t = []
                self.s_t = []
                self.in_t = []
        

        # self.v = self.leak_mem*self.v + (1-self.leak_mem)*(input)
        # self.spikes, self.v = self.spike_activation(self.v)
        
        # self.v = self.leak_mem*self.v + bntt[num_steps]*conv(inpu)
        self.v = self.leak_mem*self.v + bntt*(2*input)
        self.spikes, self.v = self.spike_activation(self.v, 1.)
        
        if self.record:
            self.v_t.append(self.v)
            self.s_t.append(self.spikes)
            self.in_t.append(input)
        
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
            spike_input = PoissonGen(input)
            s, v = self.spike_layer(spike_input, t, 1., 1.)
        
        return self.spike_layer.v_t, self.spike_layer.s_t, self.spike_layer.in_t
        




input = torch.tensor([[0.5]])
net = Net(dt=1e-3, Tsim=300e-3)
v_t, s_t, in_t = net(input)

v_t = [v[0,0] for v in v_t]
s_t = [s[0,0] for s in s_t]
in_t = [i[0,0] for i in in_t]
plt.plot(v_t)
plt.plot(s_t)
plt.plot(in_t)

