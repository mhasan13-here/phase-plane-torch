#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:55:18 2020

@author: mhasan13

Manual Network() is necessary when NeuronGroup, Monitors are contained in other
containers such as dict etc.
https://groups.google.com/forum/#!topic/briansupport/Q1Bxs2VN3vE
"""

from ObjectClass import *
import numpy as np
import brian2 as br
import time
import matplotlib.pyplot as plt
br.prefs.codegen.target = 'numpy'

br.start_scope()
dt = br.defaultclock.dt = 0.8*br.us

    
#==============================================================================         
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
    #dummy = debug(Ip_bundle, In_bundle, Ip, In, vp_inj, vn_inj) : 1 (constant over dt)
    return 0

# =============================================================================
# run and record
# =============================================================================
#start_time = time.time()
#sim_time = 20*br.ms
#
#G = SimpleNeuronGroupBrian(neuron_meshgrid, 128)
#G.L.I = 3*br.pA
#Gmon = br.StateMonitor(G.L, ('v', 'u'), record=True)
#
#P = InputGroupBrian(256)
#P.L.frequency = 255
#P.L.pulse_width = 45e-6
#Pmon = br.StateMonitor(P.L, ('x', 's'), record=True)
#
#net = br.Network()
#net.add(G.L, Gmon, P.L, Pmon)
#net.run(sim_time)
#stop_time = time.time()
#print('time to run() ', stop_time-start_time)
#plt.plot(Gmon.t/br.ms,Gmon.v[0])


# =============================================================================
# run and record
# =============================================================================
start_time = time.time()
sim_time = 10*br.ms


#P = InputGroupBrian(1)
#P.L.frequency = np.array([1e3])
#P.L.pulse_width = 45e-6
#Pmon = br.StateMonitor(P.L, ('x', 's'), record=True)
#
#G = NeuronGroupBrian(neuron_meshgrid, bundle_synapse_meshgrid, 1)
#G.L.vp_leak = 300*br.mV
#G.L.vn_leak = 0*br.mV
#Gmon = br.StateMonitor(G.L, ('v', 'u', 'Isyn', 'IpT', 'InT', 'vp_inj', 'vn_inj'), record=True)
#
#W = SynapseGroupBrian(synapse_meshgrid, P,G)
#W.S.vg_p = 130*br.mV
#W.S.vg_n = 125*br.mV


P = InputGroupBrian(257)
P.L.frequency = np.array(np.arange(0,257)*500/256)*0
P.L.pulse_width = 45e-6
Pmon = br.StateMonitor(P.L, ('x', 's'), record=True)

G = NeuronGroupBrian(neuron_meshgrid, bundle_synapse_meshgrid, 1)
G.L.v = 00*br.mV
G.L.vp_leak = 68*br.mV
G.L.vn_leak = 170*br.mV
Gmon = br.StateMonitor(G.L, ('v', 'u', 'Isyn', 'IpT', 'InT', 'vp_inj', 'vn_inj'), record=True)

W = SynapseGroupBrian(synapse_meshgrid, P,G)
W.S.vg_p = 130*br.mV
W.S.vg_n = 110*br.mV



net = br.Network()
net.add(G.L, Gmon, P.L, Pmon, W.S)
net.run(sim_time)
stop_time = time.time()
print('time to run() ', stop_time-start_time)
plt.plot(Gmon.t/br.ms,Gmon.v[0])
plt.figure()
plt.plot(Gmon.t/br.ms,Gmon.Isyn[0])
plt.show()
#plt.plot(Pmon.t/br.ms,Pmon.s[0]/(Pmon.s[0]+1e-15))