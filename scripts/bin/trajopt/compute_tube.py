import pickle
import numpy as np
import scipy as sp
from gurobipy import Model,GRB,LinExpr
import time as time
from pypolycontain.lib.containment_encodings import subset_LP,subset_zonotopes
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope

import carrot_linearization_1 as calin1 
import carrot_linearization_2 as calin2
import carrot_linearization_3 as calin3
import poly_trajectory as polytraj 
from pwa_system import system, linear_cell
import pickle


SAVE_OUTPUT = 1
if __name__=="__main__":
	file_name = "trajopt_example11_latest"
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
	t0 = int(T/4*2)
	#t0 = 0
	list_of_cells2 = []
	for t in range(t0,T):
		v1 = F_over_time[t,4]
		if v1 > 0:
			A,B,c,H,h = calin2.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		else:
			A,B,c,H,h = calin3.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		list_of_cells2.append(linear_cell(A,B,c,polytope(H,h)))
		
	pos_init = pos_over_time[t0,:]
	x0 = pos_init.reshape((-1,1))
	pos_final = params[idx+20:idx+27]
	n = 7
	epsilon = 0.01
	goal=zonotope(pos_final.reshape(-1,1),np.eye(n)*epsilon)

	(x2,u2,G2,theta2)= polytraj.polytopic_trajectory_given_modes(x0,list_of_cells2,goal,eps=1,order=1)



	list_of_cells1 = []
	for t in range(t0):
		v1 = F_over_time[t,4]
		if v1 > 0:
			A,B,c,H,h = calin2.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		else:
			A,B,c,H,h = calin3.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		list_of_cells1.append(linear_cell(A,B,c,polytope(H,h)))

	pos_init = pos_over_time[0,:]
	x0 = pos_init.reshape((-1,1))
	pos_final = pos_over_time[t0,:]
	goal=zonotope(pos_final.reshape(-1,1),G2[0])
	(x1,u1,G1,theta1)= polytraj.polytopic_trajectory_given_modes(x0,list_of_cells1,goal,eps=1,order=1)

	x1.pop()
	G1.pop()
	x = x1 + x2
	u = u1 + u2
	G = G1 + G2 
	theta = theta1 + theta2
	list_of_cells = list_of_cells1 + list_of_cells2

	epsilon2 = 0.001
	g_add = epsilon2 * np.identity(n)
	G_inv = []
	for g0 in G:
		g0 += g_add
		G_inv.append(np.linalg.inv(g0))


	if SAVE_OUTPUT:
		output = {"x": x, "u":u, "G": G, "theta": theta, "G_inv":G_inv, "list_of_cells":list_of_cells}
		pickle.dump( output, open(file_name+"_tube_output.p","wb"))
