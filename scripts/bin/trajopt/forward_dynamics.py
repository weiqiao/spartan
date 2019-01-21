from __future__ import absolute_import, division, print_function
from itertools import islice, chain
from collections import namedtuple
import time
import numpy as np
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver
import pydrake.symbolic as sym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import pickle

# Given current state and control, output the next state for the carrot-gripper system
g = 9.8

def forward_dynamics(X,F,params):
	x,y,theta,x_dot,y_dot,theta_dot,d = X
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia

	x_next = x+x_dot*dt
	y_next = y+y_dot*dt
	theta_next = theta+theta_dot*dt

	x_ddot = (F1*np.sin(theta) + Ft) / m
	x_ddot += (F1_tp*np.cos(theta) - F1_tm*np.cos(theta)) / m # friction on finger 1
	x_ddot += - F2*np.sin(phi) / m # F2, angle = pi/2 - phi
	x_ddot += (F2_tp*np.cos(phi) - F2_tm*np.cos(phi)) / m # angle = phi 
	y_ddot = (-F1*np.cos(theta) + Fn - m*g) / m
	y_ddot += (F1_tp*np.sin(theta) - F1_tm*np.sin(theta)) / m # friction on finger 1
	y_ddot += F2*np.cos(phi) / m
	y_ddot += (F2_tp*np.sin(phi) - F2_tm*np.sin(phi)) / m
	x_dot_next = x_dot + x_ddot*dt
	y_dot_next = y_dot + y_ddot*dt

	tor_F1 = F1*d 
	tor_Fn = -Fn*(DistanceCentroidToCoM*np.sin(theta))
	tor_Ft = Ft*(r-DistanceCentroidToCoM*np.cos(theta))
	tor_F1t = -F1_tp*DistanceCentroidToCoM+F1_tm*DistanceCentroidToCoM
	tor_F2 = F2*DistanceCentroidToCoM*np.sin(phi-theta)
	tor_F2t = (F2_tp-F2_tm)*(r-DistanceCentroidToCoM*np.cos(theta-phi))
	theta_ddot = (tor_F1 + tor_Fn + tor_Ft) / I
	theta_ddot += tor_F1t / I
	theta_ddot += tor_F2 / I 
	theta_ddot += tor_F2t / I
	theta_dot_next = theta_dot + theta_ddot*dt

	d_next = d-v1*dt

	X_next = np.array([x_next, y_next, theta_next, x_dot_next, y_dot_next, theta_dot_next, d_next])

	return X_next


