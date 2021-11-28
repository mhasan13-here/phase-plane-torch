#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 03:23:55 2020

@author: mhasan13
"""

import pickle as pkl
import numpy as np
import brian2 as br

class NeuronMeshGrid:
    '''
    Data on neuron phase plane
    '''
    def __init__(self, pickle_path:str) -> None:
        with open(pickle_path, 'rb') as fp:
            itemlist = pkl.load(fp)
        
        data = np.array(itemlist)
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
        self.vmax, self.vmin = np.max(self.v), np.min(self.v)
        self.umax, self.umin = np.max(self.u), np.min(self.u)
        self.j_per_x = (self.v.shape[1]-1)/(self.vmax-self.vmin)
        self.i_per_y = (self.u.shape[0]-1)/(self.umax-self.umin)
        
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
        
class InputGroupBrian:
    '''
    Input spike generation from frequency
    '''
    def __init__(self, n:int) -> None:
        self.dt = br.defaultclock.dt        
        
        self.input_neuron_model='''
                            dx/dt = 1/second : 1
                            s : 1
                            frequency : 1
                            t_period = 1/(frequency+1e-15) : 1
                            pulse_width : 1
                            '''
        self.input_spike_event_action = '''
                                    s += 1
                                    '''
        self.input_reset_event_action = '''
                                    x = pulse_width
                                    s = 0
                                    '''
        self.input_neuron_events={
                            'spike':'s==1',
                            'spike_event':'x>t_period',
                            'resetting':'x>t_period+pulse_width', 
                            'reset_event':'x<t_period'
                            } # threshold='s==1' also works
        
        self.L = br.NeuronGroup(n,
                                model=self.input_neuron_model, 
                                events=self.input_neuron_events,
                                dt=self.dt)
        
        self.L.run_on_event('spike_event',self.input_spike_event_action)
        self.L.run_on_event('resetting',self.input_reset_event_action)
        
class NeuronGroupBrian:
    '''
    Pack all the components of brian NeuronGroup 
    '''
    def __init__(self, neuron_meshgrid:NeuronMeshGrid, bundle_synapse_meshgrid:BundleSynapseMeshGrid, n:int) -> None:
        self.dt = br.defaultclock.dt
        
        self.model = '''
                    i_per_u : 1
                    j_per_v : 1
                    vmax : volt
                    vmin : volt
                    umax : volt
                    umin : volt
                    Cv : farad
                    Cu : farad
                    Cp : farad
                    Cdp_bundle : farad
                    Cdn_bundle : farad
                    
                    vp_leak : volt
                    vn_leak : volt
                    i_per_v_leak : 1
                    j_per_vd : 1
                    i_per_vm : 1
                    j_per_v_inj : 1
                    
                    IpT : amp
                    InT : amp
                    IpB = ip_bundle( int(i_per_v_leak*vp_leak/volt), int(j_per_vd*vp_inj/volt) )*amp : amp (constant over dt)
                    InB = in_bundle( int(i_per_v_leak*vn_leak/volt), int(j_per_vd*vn_inj/volt) )*amp : amp (constant over dt)
                    dvp_inj/dt = (IpB - InT)/Cdn_bundle : volt
                    dvn_inj/dt = (IpT - InB)/Cdp_bundle : volt
                    
                   
                    
                    Isyn = i_injection( int(i_per_vm*v/volt), int(j_per_v_inj*vp_inj/volt), int(j_per_v_inj*vn_inj/volt) )*amp : amp (constant over dt) 
                    dv/dt = dvdt : volt
                    dvdt=( Cv_current(int(i_per_u*u/volt),int(j_per_v*v/volt))*amp + Isyn )/(Cv+Cp) : amp/farad (constant over dt) 
                    du/dt = dudt : volt
                    dudt=Cu_current(int(i_per_u*u/volt),int(j_per_v*v/volt))*amp/(Cu+Cp) : amp/farad (constant over dt) 
                    s : 1
                    '''
        self.spike_event_action = '''
                                  s += 1
                                  '''
        self.reset_event_action = '''
                                  s = 0
                                  '''
        self.neuron_events={
                           'vdd_rail':'v>vmax',
                           'vss_rail':'v<vmin',
                           'udd_rail':'u>umax',
                           'uss_rail':'u<umin',
                           'vp_inj_rail_up':'vp_inj>vmax',
                           'vp_inj_rail_down':'vp_inj<vmin',
                           'vn_inj_rail_up':'vn_inj>vmax',
                           'vn_inj_rail_down':'vn_inj<vmin',
                           't_step':'t>0*second',
                           'spike':'s==1',
                           'spike_event':'v>200*mV',
                           'reset_event':'v<200*mV'
                           }
        
        self.L = br.NeuronGroup(n, 
                    model=self.model, 
                    events=self.neuron_events,
                    dt=self.dt
                    )
        
        self.L.vmax = neuron_meshgrid.vmax*br.volt
        self.L.vmin = neuron_meshgrid.vmin*br.volt
        self.L.umax = neuron_meshgrid.umax*br.volt
        self.L.umin = neuron_meshgrid.umin*br.volt
        self.L.i_per_u = neuron_meshgrid.i_per_y
        self.L.j_per_v = neuron_meshgrid.j_per_x
        self.L.Cv = 50e-15*br.farad
        self.L.Cu = 30e-15*br.farad
        self.L.Cp = 5e-15*br.farad 
        self.L.Cdp_bundle = 2e-15*br.farad
        self.L.Cdn_bundle = 2.5e-15*br.farad
        self.L.i_per_v_leak = bundle_synapse_meshgrid.i_per_v_leak
        self.L.j_per_vd = bundle_synapse_meshgrid.j_per_vd
        self.L.i_per_vm = bundle_synapse_meshgrid.i_per_vm
        self.L.j_per_v_inj = bundle_synapse_meshgrid.j_per_v_inj
        self.L.vp_inj = 300*br.mV # set initial value
        self.L.vn_inj = 0*br.mV # set initial value
        
        self.L.run_on_event('vdd_rail','v=vmax')
        self.L.run_on_event('vss_rail','v=vmin')
        self.L.run_on_event('udd_rail','u=umax')
        self.L.run_on_event('uss_rail','u=umin')
        self.L.run_on_event('vp_inj_rail_up','vp_inj=vmax')
        self.L.run_on_event('vp_inj_rail_down','vp_inj=vmin')
        self.L.run_on_event('vn_inj_rail_up','vn_inj=vmax')
        self.L.run_on_event('vn_inj_rail_down','vn_inj=vmin')
        self.L.run_on_event('spike_event',self.spike_event_action)
        self.L.run_on_event('reset_event',self.reset_event_action)
       
        
class SynapseGroupBrian:
    '''
    Pack all the components of synapse
    '''
    def __init__(self, synapse_meshgrid:SynapseMeshGrid, pre_group:NeuronGroupBrian, post_group:NeuronGroupBrian) -> None:
        self.syn_model = '''
                    i_per_vg_syn : 1
                    j_per_vd_syn : 1
                    
                    vg_p : volt
                    vg_n : volt
                    Isyn_active_p = syn_active_p( int(i_per_vg_syn*vg_p/volt), int(j_per_vd_syn*vn_inj/volt) )*amp  : amp (constant over dt)
                    Isyn_active_n = syn_active_n( int(i_per_vg_syn*vg_n/volt), int(j_per_vd_syn*vp_inj/volt) )*amp  : amp (constant over dt)
                    Isyn_inactive_p = syn_inactive_p( int(i_per_vg_syn*vg_p/volt), int(j_per_vd_syn*vn_inj/volt) )*amp  : amp (constant over dt)
                    Isyn_inactive_n = syn_inactive_n( int(i_per_vg_syn*vg_n/volt), int(j_per_vd_syn*vp_inj/volt) )*amp  : amp (constant over dt)
                    Ip_syn_previous_t_step : amp
                    In_syn_previous_t_step : amp
                    '''
# =============================================================================
# I += Isyn will keep increasing I for the duration of spike. but this is wrong.
# i need to keep I same as Isyn for the duration of spike. 
# with I_syn_previous_t_step variable previous timestep current can be subtracted
# from I before adding new timestep current and thus prevents I from increasing
# =============================================================================
        self.syn_active_action = '''
                            IpT -= Ip_syn_previous_t_step 
                            InT -= In_syn_previous_t_step 
                            IpT += Isyn_active_p
                            InT += Isyn_active_n
                            Ip_syn_previous_t_step = Isyn_active_p
                            In_syn_previous_t_step = Isyn_active_n
                            '''
        self.syn_inactive_action = '''
                            IpT -= Ip_syn_previous_t_step 
                            InT -= In_syn_previous_t_step 
                            IpT += Isyn_inactive_p
                            InT += Isyn_inactive_n
                            Ip_syn_previous_t_step = Isyn_inactive_p
                            In_syn_previous_t_step = Isyn_inactive_n
                            '''
        self.on_pre_action={
                        'syn_active_path':self.syn_active_action,
                        'syn_inactive_path':self.syn_inactive_action,
                        }
        self.event_assignment={
                        'syn_active_path':'spike_event',
                        'syn_inactive_path':'reset_event',
                        }
        self.S = br.Synapses(pre_group.L, post_group.L, 
                self.syn_model, 
                on_pre=self.on_pre_action, 
                on_event=self.event_assignment
                )
        self.S.connect()
        self.S.i_per_vg_syn = synapse_meshgrid.i_per_vfg
        self.S.j_per_vd_syn = synapse_meshgrid.j_per_vd
        # set drain capacitance of the bundle synapse
        # += because bais synpase is added seperately
        post_group.L.Cdp_bundle += 0.5e-15*br.farad*pre_group.L.N
        post_group.L.Cdn_bundle += 1.05e-15*br.farad*pre_group.L.N

class SimpleNeuronGroupBrian:
    '''
    Pack all the components of brian NeuronGroup 
    '''
    def __init__(self, neuron_meshgrid:NeuronMeshGrid, n:int) -> None:
        self.dt = br.defaultclock.dt
        
        self.model = '''
                    i_per_u : 1
                    j_per_v : 1
                    vmax : volt
                    vmin : volt
                    umax : volt
                    umin : volt
                    Cv : farad
                    Cu : farad
                    Cp : farad
                    
                    I : amp
                    dv/dt = dvdt : volt
                    dvdt=( Cv_current(int(i_per_u*u/volt),int(j_per_v*v/volt))*amp + I )/(Cv+Cp) : amp/farad (constant over dt) 
                    du/dt = dudt : volt
                    dudt=Cu_current(int(i_per_u*u/volt),int(j_per_v*v/volt))*amp/(Cu+Cp/2) : amp/farad (constant over dt) 
                    s : 1
                    '''
        self.spike_event_action = '''
                                  s += 1
                                  '''
        self.reset_event_action = '''
                                  s = 0
                                  '''
        self.neuron_events={
                           'vdd_rail':'v>vmax',
                           'vss_rail':'v<vmin',
                           'udd_rail':'u>umax',
                           'uss_rail':'u<umin',
                           't_step':'t>0*second',
                           'spike':'s==1',
                           'spike_event':'v>200*mV',
                           'reset_event':'v<200*mV'
                           }
        
        self.L = br.NeuronGroup(n, 
                    model=self.model, 
                    events=self.neuron_events,
                    dt=self.dt
                    )
        
        self.L.vmax = neuron_meshgrid.vmax*br.volt
        self.L.vmin = neuron_meshgrid.vmin*br.volt
        self.L.umax = neuron_meshgrid.umax*br.volt
        self.L.umin = neuron_meshgrid.umin*br.volt
        self.L.i_per_u = neuron_meshgrid.i_per_y
        self.L.j_per_v = neuron_meshgrid.j_per_x
        self.L.Cv = 50e-15*br.farad
        self.L.Cu = 30e-15*br.farad
        self.L.Cp = 5e-15*br.farad 

        
        self.L.run_on_event('vdd_rail','v=vmax')
        self.L.run_on_event('vss_rail','v=vmin')
        self.L.run_on_event('udd_rail','u=umax')
        self.L.run_on_event('uss_rail','u=umin')
        self.L.run_on_event('spike_event',self.spike_event_action)
        self.L.run_on_event('reset_event',self.reset_event_action)