# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:42:08 2021

@author: mhasan13

using exmaple from 
https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time/blob/main/model.py
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


class Net(nn.Module):
    def __init__(self, leak_mem=0.95, in_dim=1,  num_cls=1):
        super(Net, self).__init__()
        
        self.threshold = 1
        self.leak_mem = leak_mem
        self.fc = nn.Linear(in_dim, num_cls, bias=False)
        self.membrane = torch.zeros(in_dim, num_cls)
        self.a_membrane = torch.zeros(in_dim, num_cls)
    
        
    def forward(self, input):
        self.out = torch.zeros(1,1)
        self.a_membrane = self.leak_mem*self.a_membrane + (1-self.leak_mem)*(input)
        # self.membrane = self.leak_mem*self.membrane + (1-self.leak_mem)*self.fc(self.a_membrane)
        self.membrane = self.leak_mem*self.membrane + self.fc(self.a_membrane)
        self.out[self.membrane > self.threshold] = 1.
        self.membrane[self.membrane > self.threshold] = 0.
        
        return self.out, self.membrane, self.a_membrane

# input = torch.ones(1,1)
net = Net()
net.fc.weight = torch.nn.Parameter(torch.tensor([[0.5]]))
out = []
membrane = []
a_membrane = []
spike = []

for t in range(40):
    input = PoissonGen(torch.tensor(1.))
    spike.append(input.detach().numpy())
    input.unsqueeze_(0)
    input.unsqueeze_(0)
    a, b, c = net(input)
    out.append(a.detach().numpy()[0])
    membrane.append(b.detach().numpy()[0])
    a_membrane.append(c.detach().numpy()[0])
    
plt.stem(spike)
plt.plot(a_membrane)
plt.plot(membrane)
plt.plot(out)

