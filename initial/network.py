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

class SynapseMeshGrid:
    '''
    Data on synpase meshgrid
    '''
    def __init__(self, active_path:str, inactive_path:str) -> None:
        # active synapse
        with open (active_path, 'rb') as fp:
            itemlist = pkl.load(fp)
        
        data = np.array(itemlist)
        self.vfg = data[0,:,:]
        self.vd = data[1,:,:]
# =============================================================================
#         -ve sign has to be fixed for pmos currents now
#         as cadence introduced a -ve sign for outgoing current
# =============================================================================
        self.Ip_active = -data[2,:,:]
        self.In_active = data[3,:,:]
        
        with open (inactive_path, 'rb') as fp:
            itemlist = pkl.load(fp)
            
        data = np.array(itemlist)
        self.Ip_inactive = -data[2,:,:]
        self.In_inactive = data[3,:,:]
# =============================================================================
#         i=>y axis index, j=>x axis index
# =============================================================================        
        self.vfg_max, self.vfg_min = np.max(self.vfg), np.min(self.vfg)
        self.vd_max, self.vd_min = np.max(self.vd), np.min(self.vd)
        self.i_per_vfg = (self.vfg.shape[0]-1)/(self.vfg_max-self.vfg_min)
        self.j_per_vd = (self.vd.shape[1]-1)/(self.vd_max-self.vd_min)
        
class BundleSynapseMeshGrid:
    '''
    Data on bundle synapse injection current
    '''
    def __init__(self, i_bundle_path:str, i_inj_path:str) -> None:
        with open(i_bundle_path, 'rb') as fp:
            itemlist = pkl.load(fp)
            
        data = np.array(itemlist)
        self.v_leak = data[0,:,:]
        self.vd = data[1,:,:]
# =============================================================================
#         -ve sign has to be fixed for pmos currents now
#         as cadence introduced a -ve sign for outgoing current
# ============================================================================= 
        self.Ip_bundle = -(data[2,:,:]+data[3,:,:])
        self.In_bundle =   data[4,:,:]+data[5,:,:]
# =============================================================================
#         i=>y axis index, j=>x axis index
# =============================================================================    
        self.v_leak_max, self.v_leak_min = np.max(self.v_leak), np.min(self.v_leak)
        self.vd_max, self.vd_min = np.max(self.vd), np.min(self.vd)
        self.i_per_v_leak = (self.v_leak.shape[0]-1)/(self.v_leak_max-self.v_leak_min)
        self.j_per_vd = (self.vd.shape[1]-1)/(self.vd_max-self.vd_min)
        
        with open(i_inj_path, 'rb') as fp:
            itemlist = pkl.load(fp)
            
        data = np.array(itemlist)
        self.vm = data[0,:,:]
        self.v_inj = data[1,:,:]
# =============================================================================
#         -ve sign has to be fixed for pmos currents now
#         as cadence introduced a -ve sign for outgoing current
# ============================================================================= 
        self.Ip_injection = -data[2,:,:]
        self.In_injection = data[3,:,:]
# =============================================================================
#         i=>y axis index, j=>x axis index
# ============================================================================= 
        self.v_inj_max, self.v_inj_min = np.max(self.v_inj), np.min(self.v_inj)
        self.vm_max, self.vm_min = np.max(self.vm), np.min(self.vm)
        self.i_per_vm = (self.vm.shape[0]-1)/(self.vm_max-self.vm_min)
        self.j_per_v_inj = (self.v_inj.shape[1]-1)/(self.v_inj_max-self.v_inj_min)   
        
        self.Cp = 2e-15
        self.Cn = 2.5e-15

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

class SynapseLayer(nn.Module):
    """
    This layer implements the synapse from this paper
    https://ieeexplore.ieee.org/document/9184611 
    Synaptic layers in pytorch such as Linear, Conv etc. is not convenient to 
    implenent analog synapses because they are written in C. The Linear, Conv 
    layer is used to store the floating gate voltages. The output of the Linear,
    Conv layer (floating gate voltage) is used here to produce synaptic output
    current. 
    """
    def __init__(self, synapse_meshgrid, synapse_bundle_meshgrid, dt=1e-6, Tsim=1e-3, record=False):
        super(SynapseLayer, self).__init__()
        
        self.synapse_meshgrid = synapse_meshgrid
        self.dt = dt
        self.Cp = synapse_bundle_meshgrid.Cp # pmos capacitance
        self.Cn = synapse_bundle_meshgrid.Cn # nmos capacitance
        self.Tsim = Tsim
        self.record = record
        
    
class SpikingActivation(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        threshold = 0.3
        spikes = torch.zeros(input.shape)
        spikes = (input > threshold) * 1
        input[input > threshold] = 0.
        
        ctx.save_for_backward(input, spikes)
        return spikes, input

    @staticmethod
    def backward(ctx, grad_output): ####### not implemented yet
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        spikes, input = ctx.saved_tensors
        
        return input
    

class SpikingNeuron(nn.Module):
    """
    This module implements the silicon neurion cicuit from this paper
    https://ieeexplore.ieee.org/abstract/document/9184447
    
    For phase plane method this paper is followed
    https://ieeexplore.ieee.org/document/9401477
    """
    def __init__(self, neuron_meshgrid, dt=1e-6, Tsim=1e-3, record=False):
        super(SpikingNeuron, self).__init__()
        
        self.neuron_meshgrid = neuron_meshgrid
        self.spike_activation = SpikingActivation.apply
        self.dt = dt
        self.Cv = neuron_meshgrid.Cv # membrane capacitance
        self.Cu = neuron_meshgrid.Cu # auxialiary capacitance
        self.Cp = neuron_meshgrid.Cp # estimated parasitic gate capacitance
        self.Tsim = Tsim
        self.record = record
        
        # create recording variable
        if self.record:
            self.v_t = []
            self.s_t = []
        

    def forward(self, input, num_steps):
        # See the autograd section for explanation of what happens here.
        """
        Implements eulers method of integration to integrate hardware neuron equation.
        Cv*dv/dt = Iv
        Cu*du/dt = Iu
        The currents are taken from the meshgrid which is created using cadence simulation.
        """

        if num_steps == 0: ### this is here to initilize the variables at start
            self.batch_size = input.size(0)
            self.v = torch.zeros( input.size() )
            self.u = torch.zeros( input.size() )
            self.spikes = torch.zeros( input.size() )
            
        
        Iv, Iu, axon = self.fetch_current(self.v, self.u)
          
        self.v = self.v + ( (Iv + input) / (self.Cv+self.Cp) ) * self.dt
        self.u = self.u + ( Iu / (self.Cu+self.Cp) ) * self.dt
        self.spikes = axon
        # self.spikes, self.v = self.spike_activation(self.v)
        
        self.rail_out(self.v, self.u)
        
        if self.record:
            self.v_t.append(self.v)
            self.s_t.append(self.spikes)
        
        return self.spikes, self.v
    
    def fetch_current(self, v, u): 
        """
        Voltages are converted into indices which are used to fetch currents 
        from the phase plane
        """
        Iv = self.neuron_meshgrid.iCv[ (u*self.neuron_meshgrid.i_per_y).long(), (v*self.neuron_meshgrid.j_per_x).long() ]
        Iu = self.neuron_meshgrid.iCu[ (u*self.neuron_meshgrid.i_per_y).long(), (v*self.neuron_meshgrid.j_per_x).long() ]
        axon = self.neuron_meshgrid.axon[ (u*self.neuron_meshgrid.i_per_y).long(), (v*self.neuron_meshgrid.j_per_x).long() ]
        return Iv, Iu, axon

    def rail_out(self, v, u):
        self.v[v > self.neuron_meshgrid.vmax] = self.neuron_meshgrid.vmax
        self.v[v < self.neuron_meshgrid.vmin] = self.neuron_meshgrid.vmin
        self.u[u > self.neuron_meshgrid.umax] = self.neuron_meshgrid.umax
        self.u[u < self.neuron_meshgrid.umin] = self.neuron_meshgrid.umin


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'Spiking Neuron Layer'
    
    
class Net(nn.Module):
    def __init__(self, neuron_meshgrid, dt=1e-6, Tsim=1e-3):
        super(Net, self).__init__()
        self.dt = dt
        self.Tsim = Tsim
        self.num_steps = int(Tsim/dt) + 1 
        
        # define neural network layers
        self.spike_layer = SpikingNeuron(neuron_meshgrid, dt, Tsim, record=True)

    def forward(self, input):
        for t in range(self.num_steps):
            s, v = self.spike_layer(input, t)
        
        return self.spike_layer.v_t, self.spike_layer.s_t
        



Tsim = 10e-3
neuron_meshgrid = NeuronMeshGrid('neuron.pickle')
synapse_meshgrid = SynapseMeshGrid('split_files/synapse-active/synapse-active.pickle',
                                   'split_files/synapse-inactive/synapse-inactive.pickle')
synapse_bundle_meshgrid = BundleSynapseMeshGrid('split_files/synapse-bundle-current/synapse-bundle-current.pickle',
                                                'split_files/synapse-bundle-injection/synapse-bundle-injection.pickle')
input = torch.tensor([[5e-12]])
net = Net(neuron_meshgrid, dt=1e-6, Tsim=Tsim)
v_t, s_t = net(input)

v_t = [v[0,0] for v in v_t]
s_t = [s[0,0] for s in s_t]
plt.plot(v_t)
plt.plot(s_t)
plt.show()
end_time = time.time()
print(end_time-start_time)

    



