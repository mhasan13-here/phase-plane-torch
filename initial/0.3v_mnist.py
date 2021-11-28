#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:20:51 2020

@author: mhasan13
"""
from ObjectClass import *
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
import pickle as pkl
from scipy.interpolate import interp1d
import time
import brian2 as br


def weight_prep(synapse_meshgrid, vp_leak, vn_leak, weight):
    vfg = np.zeros(weight.shape)
    I_max = 5e-12
    
    I_pos = synapse_meshgrid.In_active[:,int(synapse_meshgrid.j_per_vd*0.300)] - synapse_meshgrid.Ip_active[int(synapse_meshgrid.i_per_vfg*vp_leak), int(synapse_meshgrid.j_per_vd*0.000)]
    vfg_syn = synapse_meshgrid.vfg[:, int(synapse_meshgrid.j_per_vd*0.300)]
    f_w_to_vfg = interp1d(I_pos, vfg_syn)
    vfg[weight>=0] = f_w_to_vfg(weight[weight>=0]*I_max)

    I_neg = synapse_meshgrid.In_active[int(synapse_meshgrid.i_per_vfg*vn_leak),int(synapse_meshgrid.j_per_vd*0.300)] - synapse_meshgrid.Ip_active[:, int(synapse_meshgrid.j_per_vd*0.000)]
    vfg_syn = synapse_meshgrid.vfg[:, int(synapse_meshgrid.j_per_vd*0.300)]
    f_w_to_vfg = interp1d(I_neg, vfg_syn)
    vfg[weight<0] = f_w_to_vfg(weight[weight<0]*I_max)
    
    return vfg
    
    
br.prefs.codegen.target = 'numpy'
br.start_scope()
dt = br.defaultclock.dt = 1*br.us

# =============================================================================
# # mnist data preparation
# =============================================================================
reduced_row = reduced_col = 16
mnist_file = '/nfs/users/mhasan13/linux/Desktop/iss-research_nfs/mhasan13/fc-spiking-mnist/smaller/data/mnist_test.csv'
mnist_data = np.loadtxt(mnist_file, delimiter=',')
images = mnist_data[:,1:]
labels = mnist_data[:,0]
# fetch a random digit
random_idx = random.choice(range(len(labels))) 
image = images[random_idx,:].reshape((28,28))
image = cv.resize(image,(reduced_row,reduced_row),cv.INTER_CUBIC)

# =============================================================================
# # TF weights in transposed state => #rows=input, #cols=output
# =============================================================================
weight_file = '/nfs/users/mhasan13/linux/Desktop/iss-research_nfs/mhasan13/fc-spiking-mnist/smaller/data/hidden_layer_0_weights.csv'
weight_1 = np.loadtxt(weight_file, delimiter=',')
weight_1_max = np.max(np.abs(weight_1))
weight_file = '/nfs/users/mhasan13/linux/Desktop/iss-research_nfs/mhasan13/fc-spiking-mnist/smaller/data/hidden_layer_1_weights.csv'
weight_2 = np.loadtxt(weight_file, delimiter=',')
weight_2_max = np.max(np.abs(weight_2))

# biases
bias_file = '/nfs/users/mhasan13/linux/Desktop/iss-research_nfs/mhasan13/fc-spiking-mnist/smaller/data/hidden_layer_0_biases.csv'
bias_1 = np.loadtxt(bias_file, delimiter=',')
bias_1_max = np.max(np.abs(bias_1))
bias_file = '/nfs/users/mhasan13/linux/Desktop/iss-research_nfs/mhasan13/fc-spiking-mnist/smaller/data/hidden_layer_1_biases.csv'
bias_2 = np.loadtxt(bias_file, delimiter=',')
bias_2_max = np.max(np.abs(bias_2))

# normalize weights or not
weight_max = np.max([weight_1_max, weight_2_max, bias_1_max, bias_2_max])
weight_layer_1 = weight_1#/weight_max
weight_layer_2 = weight_2#/weight_max
bias_layer_1 = bias_1#/weight_max
bias_layer_2 = bias_2#/weight_max
# weight to floating gate voltage
synapse_meshgrid = SynapseMeshGrid('../../meshgrid-generation/v3/synapse-active.pickle',
                                   '../../meshgrid-generation/v3/synapse-inactive.pickle')
#vdp_at = 000*br.mV
#vdn_at = 300*br.mV
##vp_ref = 130*br.mV
#Ip_at = synapse_meshgrid.Ip_active[:, int(synapse_meshgrid.j_per_vd*vdp_at)]
#In_at = synapse_meshgrid.In_active[:, int(synapse_meshgrid.j_per_vd*vdn_at)]
#I_syn = In_at - Ip_at
#vfg_syn = synapse_meshgrid.vfg[:, int(synapse_meshgrid.j_per_vd*vdn_at)]
#f_w_to_vfg = interp1d(I_syn, vfg_syn)
#I_max = 1200e-12
#vfg_layer_1 = f_w_to_vfg(weight_layer_1*I_max)
#vfg_layer_2 = f_w_to_vfg(weight_layer_2*I_max)
#bias_vfg_layer_1 = f_w_to_vfg(bias_layer_1*I_max)
#bias_vfg_layer_2 = f_w_to_vfg(bias_layer_2*I_max)
## =============================================================================
# meshgrid data
# =============================================================================   
neuron_meshgrid = NeuronMeshGrid('../../meshgrid-generation/v3/neuron.pickle')

@br.check_units(i=1, j=1, result=1)
def Cv_current(i:int, j:int) -> float:
    
    return neuron_meshgrid.iCv[i,j]
    
@br.check_units(i=1, j=1, result=1)
def Cu_current(i:int, j:int) -> float:
    
    return neuron_meshgrid.iCu[i,j]

synapse_meshgrid = SynapseMeshGrid('../../meshgrid-generation/v3/synapse-active.pickle',
                                   '../../meshgrid-generation/v3/synapse-inactive.pickle')

@br.check_units(i=1, j=1, result=1)
def syn_active_p(i:int, j:int) -> float:
    
    return synapse_meshgrid.Ip_active[i,j]
@br.check_units(i=1, j=1, result=1)
def syn_active_n(i:int, j:int) -> float:
    
    return synapse_meshgrid.In_active[i,j]

@br.check_units(i=1, j=1, result=1)
def syn_inactive_p(i:int, j:int) -> float:
    
    return synapse_meshgrid.Ip_inactive[i,j]
@br.check_units(i=1, j=1, result=1)
def syn_inactive_n(i:int, j:int) -> float:
    
    return synapse_meshgrid.In_inactive[i,j]

bundle_synapse_meshgrid = BundleSynapseMeshGrid('../../meshgrid-generation/v3/synapse-bundle-current.pickle',
                                                '../../meshgrid-generation/v3/synapse-bundle-injection.pickle')

@br.check_units(i=1, j=1, result=1)
def ip_bundle(i:int, j:int) -> float:
    
    return bundle_synapse_meshgrid.Ip_bundle[i,j]

@br.check_units(i=1, j=1, result=1)
def in_bundle(i:int, j:int) -> float:
    
    return bundle_synapse_meshgrid.In_bundle[i,j]

@br.check_units(i=1, jp=1 , jn=1, result=1)
def i_injection(i:int, jp:int, jn:int) -> float:
    
    return bundle_synapse_meshgrid.Ip_injection[i,jp] - bundle_synapse_meshgrid.In_injection[i,jn]

@br.check_units(Ip_bundle=br.amp, In_bundle=br.amp, Ip=br.amp, In=br.amp, vp_inj=br.volt, vn_inj=br.volt, result=1)
def debug(Ip_bundle, In_bundle, Ip, In, vp_inj, vn_inj):
#    print(Ip_bundle, In_bundle, Ip, In, vp_inj, vn_inj)
    return 0
# =============================================================================
# network preparation
# =============================================================================
L0 = InputGroupBrian(reduced_row*reduced_col)
L0.L.pulse_width = 45e-6
L0.L.frequency = image.flatten()
L0_mon = br.StateMonitor(L0.L, ('s'), record=True)
L0_spk = br.SpikeMonitor(L0.L, record=True)
# next layer
L1 = NeuronGroupBrian(neuron_meshgrid, bundle_synapse_meshgrid, 32)
L1.L.vp_leak = 70*br.mV
L1.L.vn_leak = 160*br.mV
L1_mon = br.StateMonitor(L1.L, ('v','u','Isyn','IpT','InT','vp_inj','vn_inj'), record=True)
L1_spk = br.SpikeMonitor(L1.L, record=True)
# next layer
L2 = NeuronGroupBrian(neuron_meshgrid, bundle_synapse_meshgrid, 10)
L2.L.vp_leak = 140*br.mV
L2.L.vn_leak = 80*br.mV
L2_mon = br.StateMonitor(L2.L, ('v','u','Isyn','IpT','InT','vp_inj','vn_inj'), record=True)
L2_spk = br.SpikeMonitor(L2.L, record=True)
# bias generator
B0 = InputGroupBrian(1)
B0.L.pulse_width = 45e-6
B0.L.frequency = 255
B1 = InputGroupBrian(1)
B1.L.pulse_width = 45e-6
B1.L.frequency = 255
# =============================================================================
# synapse
# =============================================================================
W1 = SynapseGroupBrian(synapse_meshgrid, L0,L1)
vfg_layer_1 = weight_prep(synapse_meshgrid, L1.L.vp_leak[0], L1.L.vn_leak[0], weight_layer_1)
W1.S.vg_p = vfg_layer_1.flatten(order='C')*br.volt 
W1.S.vg_n = vfg_layer_1.flatten(order='C')*br.volt 
W1_b = SynapseGroupBrian(synapse_meshgrid, B0,L1)
bias_vfg_layer_1 = weight_prep(synapse_meshgrid, L1.L.vp_leak[0], L1.L.vn_leak[0], bias_layer_1)
W1_b.S.vg_p = bias_vfg_layer_1.flatten(order='C')*br.volt
W1_b.S.vg_n = bias_vfg_layer_1.flatten(order='C')*br.volt
# next layer
W2 = SynapseGroupBrian(synapse_meshgrid, L1,L2)
vfg_layer_2 = weight_prep(synapse_meshgrid, L2.L.vp_leak[0], L2.L.vn_leak[0], weight_layer_2)
W2.S.vg_p = vfg_layer_2.flatten(order='C')*br.volt
W2.S.vg_n = vfg_layer_2.flatten(order='C')*br.volt
W2_b = SynapseGroupBrian(synapse_meshgrid, B1,L2)
bias_vfg_layer_2 = weight_prep(synapse_meshgrid, L2.L.vp_leak[0], L2.L.vn_leak[0], bias_layer_2)
W2_b.S.vg_p = bias_vfg_layer_2.flatten(order='C')*br.volt
W2_b.S.vg_n = bias_vfg_layer_2.flatten(order='C')*br.volt
# =============================================================================
# fix capacitor
# =============================================================================
#L1.L.Cdp_bundle = 642e-15*br.farad
#L1.L.Cdn_bundle = 642e-15*br.farad
#L2.L.Cdp_bundle = 5.5e-15*br.farad
#L2.L.Cdn_bundle = 5.5e-15*br.farad
# =============================================================================
# run and record
# =============================================================================
start_time = time.time()
sim_time = 50*br.ms
net = br.Network()
net.add(L0.L, L1.L, L2.L, B0.L, B1.L, W1.S, W2.S, W1_b.S, W2_b.S, L0_mon, L1_mon, L2_mon, L0_spk, L1_spk, L2_spk)
net.run(sim_time)
stop_time = time.time()
print('time to run() ', stop_time-start_time)
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplot(122)
plt.plot(L2_spk.i, L2_spk.t/br.ms,'.')
plt.ylabel('time (ms)'), plt.xlabel('neuron index')
plt.gca().set_xticks(range(10))
plt.grid(True)
plt.gcf().set_size_inches(6,2.5)
plt.gcf().set_tight_layout(True)
plt.figure()
plt.bar(range(10),L2_spk.count)

#plt.plot(L1_mon.t/br.ms,L1_mon.v[0])
#plt.figure()
#plt.plot(L1_mon.t/br.ms,L1_mon.Isyn[0])
plt.show()