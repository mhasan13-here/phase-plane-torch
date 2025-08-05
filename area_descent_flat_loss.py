#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 20:22:07 2025

@author: croc
"""

# y = 2 - 0.5*np.cos(2*np.pi*(x-3))*np.exp(-np.abs(x-3)/1)

 # x = X-3
 # L = -0.8*np.cos(2*np.pi*x) * np.exp(-np.abs(x)/1)
 # L = L + 2

import numpy as np
import matplotlib.pyplot as plt

def loss_function(X):
    x = X - 1.5 # offset
    L = np.heaviside((np.abs(x)-1), 1) + np.heaviside((np.abs(x)-2), 1) + np.heaviside((np.abs(x)-3), 1)
    L = L + 3 # offset
    return L

def wm_dynamics(w, tau, dt, wo, wm, Lo, Lm):
    # move wm towards minizing area
    w_t = -(wm - wo)*(Lm - Lo) * (dt/tau) + w
    return w_t
    
def wo_dynamics(w, tau, dt, wo, wm, Lo, Lm):
    # move wo towards wm
    w_t = (wm - wo) * (dt/tau) + w
    return w_t
    
def set_wm_wo(wo, wm, Lo, Lm):
    # return as order (wo, wm) <- (Lo > Lm)
    if Lm > Lo:
        return wm, wo, Lm, Lo # swtich roles
    else:
        return wo, wm, Lo, Lm # stay same

x = np.linspace(-5,5,500)
L = loss_function(x)

# Loss function plot
fig = plt.figure()
plt.gca().plot(x,L)
plt.gca().axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.gca().axvline(0, color='k', linestyle='--', linewidth=0.5)

# set initial points
wo = -3
wm = 5
p_wo = plt.gca().plot(wo, loss_function(wo), 'g.', markersize=10)
p_wm = plt.gca().plot(wm, loss_function(wm), 'r.', markersize=10)
#plt.gca().vlines(wo, ymin=0, ymax=loss_function(wo), color='g', linestyle='--', linewidth=1)
#plt.gca().vlines(wm, ymin=0, ymax=loss_function(wm), color='g', linestyle='--', linewidth=1)

# set dynamics parameters
dt = 1e-6
tau_wo = 1*1e-4 # wo is faster
tau_wm = 2*1e-4 # wm is slower

for i in range(1000):
    # set wm as the smaller loss
    Lo, Lm = loss_function(wo), loss_function(wm)
    wo, wm, Lo, Lm = set_wm_wo(wo, wm, Lo, Lm)
    
    # move to next value
    wo, wm = wo_dynamics(wo, tau_wo, dt, wo, wm, Lo, Lm), wm_dynamics(wm, tau_wm, dt, wo, wm, Lo, Lm)
    
    # plot
    p_wo[0].set_data([wo], [loss_function(wo)])
    p_wm[0].set_data([wm], [loss_function(wm)])

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    
    # plt.pause(1e-2)



