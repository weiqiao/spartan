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

import carrot_linearization_1 as calin1 
import carrot_linearization_2 as calin2
import carrot_linearization_3 as calin3

SAVE_OUTPUT = 1

def tvlqr(A,B,Q,R,P0,T):
	# A, B, P, Q are 3d tensors, a tuple (At, Bt, Qt, Rt) at time t for t from 0 to T-1
	n = A.shape[0]
	m = B.shape[1]
	K = np.zeros((m,n,T))
	Pt = P0
	for t in range(T-1,-1,-1):
		At = A[:,:,t]
		Bt = B[:,:,t]
		Qt = Q[:,:,t]
		Rt = R[:,:,t]
		Kt = -np.linalg.inv(Rt+Bt.T.dot(Pt).dot(Bt)).dot(Bt.T).dot(Pt).dot(At)
		Ct = At + Bt.dot(Kt)
		Pt = Qt + (Kt.T).dot(Rt).dot(Kt) + (Ct.T).dot(Pt).dot(Ct)
		K[:,:,t] = Kt

	return K

if __name__=="__main__":
	file_name = "trajopt_example8_T100_phialwaysincreasing"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]
	'''
		params_save = np.array([11])                                                                    # 0: idx
		params_save = np.append(params_save,np.array([mass, inertia, DistanceCentroidToCoM, r, dt, 
			DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel]))  # 1~idx
		params_save = np.append(params_save,StateBound[0,:])											# idx+1~idx+6
		params_save = np.append(params_save,StateBound[1,:])											# idx+7~idx+12
		params_save = np.append(params_save,pos_init)												    # idx+13~idx+19
		params_save = np.append(params_save,pos_final)												 	# idx+20~idx+26
		params_save = np.append(params_save,[T])														# idx+27
	'''
	idx = int(params[0])
	T = int(params[idx+27])
	#t0 = int(T/4*2)
	t0 = 0
	list_of_cells = []
	n = 7
	m = 12
	AT = np.zeros((n+1,n+1,T-t0))
	BT = np.zeros((n+1,m,T-t0))
	QT = np.zeros((n+1,n+1,T-t0))
	RT = np.zeros((m,m,T-t0))
	P0 = np.eye(n+1)
	for t in range(t0,T):
		v1 = F_over_time[t,4]
		# if np.abs(v1) < 0.001:
		# 	A,B,c,H,h = calin1.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		# elif v1 > 0.001:
		# 	A,B,c,H,h = calin2.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		# else:
		# 	A,B,c,H,h = calin3.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		if v1 > 0:
			A,B,c,H,h = calin2.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		else:
			A,B,c,H,h = calin3.linearize(pos_over_time[t,:], F_over_time[t,:], params)

		AT[:n,:n,t-t0] = A 
		AT[:n,n,t-t0] = c[:,0]
		AT[n,n,t-t0] = 1
		BT[:n,:m,t-t0] = B 
		QT[:n,:n,t-t0] = np.eye(n)
		RT[:,:,t-t0] = np.eye(m)
	K = tvlqr(AT,BT,QT,RT,P0,T-t0)

	if SAVE_OUTPUT:
		output = {"K": K}
		pickle.dump( output, open(file_name+"_tvlqr_output.p","wb"))
